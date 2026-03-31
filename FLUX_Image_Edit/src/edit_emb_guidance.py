"""
FLUX Image Editing with Embedding-level Guidance.

Applies guidance in the embedding space before the model:
    guided_txt = (1 + alpha) * cond_txt - alpha * uncond_txt
"""

import os
import re
import time
import argparse
from glob import iglob

import torch
import numpy as np
from einops import rearrange
from PIL import ExifTags, Image

from flux.sampling import get_schedule, prepare, unpack, denoise
from flux.sampling_emb_guidance import denoise_emb_guidance
from flux.util import configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5


@torch.inference_mode()
def encode(init_image, torch_device, ae):
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0)
    init_image = init_image.to(torch_device)
    init_image = ae.encode(init_image.to()).to(torch.bfloat16)
    return init_image


@torch.inference_mode()
def main(args):
    name = args.name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    torch.set_grad_enabled(False)

    num_steps = args.num_steps or (4 if name == "flux-schnell" else 25)
    offload = args.offload

    print(f"=== FLUX Embedding Guidance ===")
    print(f"Alpha: {args.emb_guidance_alpha}, Guidance: {args.guidance}")
    print(f"Source: {args.source_prompt}")
    print(f"Target: {args.target_prompt}")
    print(f"Inject: {args.inject}, Steps: {num_steps}")

    # Load models
    t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)

    if offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.encoder.to(torch_device)

    # Load and encode source image
    init_image = np.array(Image.open(args.source_img_dir).convert('RGB'))
    shape = init_image.shape
    new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
    new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
    init_image = init_image[:new_h, :new_w, :]
    width, height = init_image.shape[0], init_image.shape[1]
    init_image = encode(init_image, torch_device, ae)

    if offload:
        ae = ae.cpu()
        torch.cuda.empty_cache()
        t5, clip = t5.to(torch_device), clip.to(torch_device)

    # Prepare embeddings for source, target, and empty (unconditional)
    inp_source = prepare(t5, clip, init_image, prompt=args.source_prompt)
    inp_target = prepare(t5, clip, init_image, prompt=args.target_prompt)
    inp_uncond = prepare(t5, clip, init_image, prompt="")

    if offload:
        t5, clip = t5.cpu(), clip.cpu()
        torch.cuda.empty_cache()
        model = model.to(torch_device)

    # Setup info
    info = {
        'feature_path': args.feature_path,
        'feature': {},
        'inject_step': args.inject,
    }
    if not os.path.exists(args.feature_path):
        os.makedirs(args.feature_path, exist_ok=True)

    # Step 1: Inversion with source prompt (no guidance)
    timesteps = get_schedule(num_steps, inp_source["img"].shape[1], shift=(name != "flux-schnell"))
    z, info = denoise(model, **inp_source, timesteps=timesteps, guidance=1, inverse=True, info=info)

    # Step 2: Denoise with embedding guidance
    inp_target["img"] = z
    timesteps = get_schedule(num_steps, inp_target["img"].shape[1], shift=(name != "flux-schnell"))

    x, _ = denoise_emb_guidance(
        model,
        img=inp_target["img"],
        img_ids=inp_target["img_ids"],
        cond_txt=inp_target["txt"],
        uncond_txt=inp_uncond["txt"],
        txt_ids=inp_target["txt_ids"],
        cond_vec=inp_target["vec"],
        uncond_vec=inp_uncond["vec"],
        timesteps=timesteps,
        inverse=False,
        info=info,
        guidance=args.guidance,
        emb_guidance_alpha=args.emb_guidance_alpha,
    )

    if offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.decoder.to(x.device)

    # Decode
    batch_x = unpack(x.float(), width, height)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for xi in batch_x:
        xi = xi.unsqueeze(0)
        with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
            xi = ae.decode(xi)

        t1 = time.perf_counter()
        xi = xi.clamp(-1, 1)
        xi = embed_watermark(xi.float())
        xi = rearrange(xi[0], "c h w -> h w c")
        img = Image.fromarray((127.5 * (xi + 1.0)).cpu().byte().numpy())

        fn = os.path.join(output_dir, f"alpha{args.emb_guidance_alpha}_g{args.guidance}_inj{args.inject}.jpg")
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux;emb_guidance"
        exif_data[ExifTags.Base.ImageDescription] = args.target_prompt
        img.save(fn, exif=exif_data, quality=95, subsampling=0)
        print(f"Saved: {fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FLUX RF-Edit with Embedding Guidance')
    parser.add_argument('--name', default='flux-dev', type=str)
    parser.add_argument('--source_img_dir', required=True, type=str)
    parser.add_argument('--source_prompt', type=str, default='')
    parser.add_argument('--target_prompt', type=str, required=True)
    parser.add_argument('--feature_path', type=str, default='feature_emb')
    parser.add_argument('--guidance', type=float, default=2.0)
    parser.add_argument('--num_steps', type=int, default=25)
    parser.add_argument('--inject', type=int, default=4)
    parser.add_argument('--output_dir', default='output_emb', type=str)
    parser.add_argument('--offload', action='store_true')
    parser.add_argument('--emb-guidance-alpha', type=float, default=0.5)
    args = parser.parse_args()
    main(args)
