"""
Run FLUX attention guidance on PIE-Bench++ dataset.

Usage:
    python run_pie_bench_attn_guidance.py --dataset_dir /path/to/pie_bench_pp --output_dir /path/to/output
    python run_pie_bench_attn_guidance.py --attn-k 32 --attn-alpha 0.5 --dataset_dir ... --output_dir ...
"""

import os
import sys
import json
import time
import argparse

import torch
import numpy as np
import pandas as pd
from PIL import Image
from einops import rearrange

from flux.sampling import denoise, get_schedule, prepare, unpack
from flux.sampling_attn_guidance import denoise_attn_guidance
from flux.util import configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5


@torch.inference_mode()
def encode(init_image, torch_device, ae):
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0)
    init_image = init_image.to(torch_device)
    init_image = ae.encode(init_image.to()).to(torch.bfloat16)
    return init_image


@torch.inference_mode()
def edit_single_image(
    model, ae, t5, clip, torch_device,
    source_image_pil, source_prompt, target_prompt,
    inject, guidance, num_steps, name,
    offload, attn_k=32, attn_alpha=1.0, overlap_target="uncond",
):
    """Edit a single image with attention guidance."""
    init_image = np.array(source_image_pil.convert('RGB'))
    h, w = init_image.shape[:2]
    new_h = h if h % 16 == 0 else h - h % 16
    new_w = w if w % 16 == 0 else w - w % 16
    init_image = init_image[:new_h, :new_w, :]
    width, height = init_image.shape[0], init_image.shape[1]

    if offload:
        ae.encoder.to(torch_device)
    init_latent = encode(init_image, torch_device, ae)
    if offload:
        ae.encoder.cpu()
        torch.cuda.empty_cache()
        t5.to(torch_device)
        clip.to(torch_device)

    # Prepare embeddings
    inp_source = prepare(t5, clip, init_latent, prompt=source_prompt)
    inp_target = prepare(t5, clip, init_latent, prompt=target_prompt)
    inp_uncond = prepare(t5, clip, init_latent, prompt="")

    if offload:
        t5.cpu()
        clip.cpu()
        torch.cuda.empty_cache()
        model.to(torch_device)

    # Inversion
    info = {'feature': {}, 'inject_step': inject, 'feature_path': '/tmp/flux_feat_attn'}
    os.makedirs(info['feature_path'], exist_ok=True)
    timesteps = get_schedule(num_steps, inp_source["img"].shape[1], shift=(name != "flux-schnell"))
    z, info = denoise(model, **inp_source, timesteps=timesteps, guidance=1, inverse=True, info=info)

    inp_target["img"] = z
    timesteps = get_schedule(num_steps, inp_target["img"].shape[1], shift=(name != "flux-schnell"))

    # Denoising with attention guidance
    x, _ = denoise_attn_guidance(
        model,
        img=inp_target["img"], img_ids=inp_target["img_ids"],
        cond_txt=inp_target["txt"], uncond_txt=inp_uncond["txt"],
        txt_ids=inp_target["txt_ids"],
        cond_vec=inp_target["vec"], uncond_vec=inp_uncond["vec"],
        timesteps=timesteps, inverse=False, info=info,
        guidance=guidance,
        attn_k=attn_k, attn_alpha=attn_alpha, overlap_target=overlap_target,
    )

    if offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.decoder.to(x.device)

    # Decode
    batch_x = unpack(x.float(), width, height)
    xi = batch_x[0].unsqueeze(0)
    with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
        xi = ae.decode(xi)
    if offload:
        ae.decoder.cpu()
        torch.cuda.empty_cache()

    xi = xi.clamp(-1, 1)
    xi = rearrange(xi[0], "c h w -> h w c")
    edited_img = Image.fromarray((127.5 * (xi + 1.0)).cpu().byte().numpy())
    return edited_img, init_image


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    torch.set_grad_enabled(False)
    name = args.name
    num_steps = args.num_steps or (4 if name == "flux-schnell" else 25)

    print(f"=== PIE-Bench Attention Guidance ===")
    print(f"k={args.attn_k}, alpha={args.attn_alpha}, overlap={args.overlap_target}")
    print(f"Guidance: {args.guidance}, Inject: {args.inject}, Steps: {num_steps}")

    # Load models once
    t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if args.offload else torch_device)
    ae = load_ae(name, device="cpu" if args.offload else torch_device)

    if args.offload:
        model.cpu(); torch.cuda.empty_cache()

    # Load dataset
    dataset_dir = args.dataset_dir
    categories = sorted([d for d in os.listdir(dataset_dir)
                        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')])

    if args.category:
        categories = [c for c in categories if c == args.category]

    method_name = f"attn_k{args.attn_k}_a{args.attn_alpha}"
    total_processed = 0
    total_errors = 0

    for cat in categories:
        pq_path = os.path.join(dataset_dir, cat, "V1-00000-of-00001.parquet")
        if not os.path.exists(pq_path):
            continue

        df = pd.read_parquet(pq_path)
        n = min(len(df), args.max_samples) if args.max_samples else len(df)
        print(f"\n=== Category: {cat} ({n}/{len(df)} samples) | {method_name} ===")

        out_dir = os.path.join(args.output_dir, method_name, cat)
        os.makedirs(out_dir, exist_ok=True)

        for idx in range(n):
            row = df.iloc[idx]
            source_prompt = row['source_prompt']
            target_prompt = row['target_prompt'].replace('[', '').replace(']', '')
            sample_id = row.get('id', idx)

            img_data = row['image']
            if isinstance(img_data, dict) and 'bytes' in img_data:
                import io
                source_img = Image.open(io.BytesIO(img_data['bytes']))
            elif isinstance(img_data, dict) and 'path' in img_data:
                source_img = Image.open(img_data['path'])
            else:
                continue

            out_path = os.path.join(out_dir, f"{sample_id}.jpg")
            if os.path.exists(out_path):
                total_processed += 1
                continue

            t0 = time.time()
            try:
                edited_img, source_np = edit_single_image(
                    model, ae, t5, clip, torch_device,
                    source_img, source_prompt, target_prompt,
                    args.inject, args.guidance, num_steps, name,
                    args.offload, args.attn_k, args.attn_alpha, args.overlap_target,
                )
                edited_img.save(out_path, quality=95)
                source_img.save(os.path.join(out_dir, f"{sample_id}_source.jpg"), quality=95)

                meta = {
                    'source_prompt': source_prompt,
                    'target_prompt': target_prompt,
                    'category': cat,
                    'sample_id': str(sample_id),
                    'mode': 'attn_guidance',
                    'attn_k': args.attn_k,
                    'attn_alpha': args.attn_alpha,
                    'overlap_target': args.overlap_target,
                    'guidance': args.guidance,
                    'inject': args.inject,
                }
                with open(out_path.replace('.jpg', '_meta.json'), 'w') as f:
                    json.dump(meta, f, indent=2)

                elapsed = time.time() - t0
                total_processed += 1
                print(f"  [{idx+1}/{n}] {sample_id}: {elapsed:.1f}s")
            except Exception as e:
                total_errors += 1
                print(f"  [{idx+1}/{n}] ERROR: {e}")

    print(f"\n=== Done! Processed: {total_processed}, Errors: {total_errors} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PIE-Bench evaluation with FLUX attention guidance')
    parser.add_argument('--dataset_dir', required=True, help='Path to PIE-Bench++ dataset')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--name', default='flux-dev', type=str)
    parser.add_argument('--guidance', type=float, default=2.0)
    parser.add_argument('--num_steps', type=int, default=25)
    parser.add_argument('--inject', type=int, default=4)
    parser.add_argument('--offload', action='store_true')
    parser.add_argument('--category', type=str, default=None)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--attn-k', type=int, default=32, help='Top-k for overlap detection')
    parser.add_argument('--attn-alpha', type=float, default=1.0, help='Attention guidance strength')
    parser.add_argument('--overlap-target', default='uncond', choices=['uncond', 'cond'])
    args = parser.parse_args()
    main(args)
