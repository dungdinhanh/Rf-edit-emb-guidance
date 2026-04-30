"""
Visualize attention maps per layer with/without single-layer CFG.

For each layer, overlays the image-to-text attention on the source image,
comparing cond-only vs single-layer CFG. Shows what each layer's guidance
focuses on spatially.
"""

import os, sys, io, json, math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class AttentionCaptureHook:
    """Captures image-to-text attention weights from a specific block."""
    def __init__(self):
        self.img_to_txt_attn = None  # [B, H, img_len, txt_len]

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, *args, **kwargs):
        residual = hidden_states
        if hidden_states.ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h*w).transpose(1, 2)
        if encoder_hidden_states.ndim == 4:
            b, c, h, w = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(b, c, h*w).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]
        inner_dim = attn.to_q.out_features
        head_dim = inner_dim // attn.heads
        H = attn.heads

        img_q = attn.to_q(hidden_states).view(batch_size, -1, H, head_dim).transpose(1, 2)
        img_k = attn.to_k(hidden_states).view(batch_size, -1, H, head_dim).transpose(1, 2)
        img_v = attn.to_v(hidden_states).view(batch_size, -1, H, head_dim).transpose(1, 2)
        txt_q = attn.add_q_proj(encoder_hidden_states).view(batch_size, -1, H, head_dim).transpose(1, 2)
        txt_k = attn.add_k_proj(encoder_hidden_states).view(batch_size, -1, H, head_dim).transpose(1, 2)
        txt_v = attn.add_v_proj(encoder_hidden_states).view(batch_size, -1, H, head_dim).transpose(1, 2)

        img_len = img_q.shape[2]
        full_q = torch.cat([img_q, txt_q], dim=2)
        full_k = torch.cat([img_k, txt_k], dim=2)
        full_v = torch.cat([img_v, txt_v], dim=2)

        # Compute attention weights explicitly
        scale = head_dim ** -0.5
        attn_logits = torch.matmul(full_q, full_k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_logits, dim=-1)

        # Store image-to-text portion, averaged over heads
        self.img_to_txt_attn = attn_weights[:, :, :img_len, img_len:].detach().cpu().float().mean(dim=1)  # [B, img_len, txt_len]

        # Standard attention output
        out = F.scaled_dot_product_attention(full_q, full_k, full_v)
        out = out.transpose(1, 2).reshape(batch_size, -1, inner_dim).to(hidden_states.dtype)

        hidden_out = out[:, :residual.shape[1]]
        enc_out = out[:, residual.shape[1]:]
        hidden_out = attn.to_out[0](hidden_out)
        hidden_out = attn.to_out[1](hidden_out)
        if not attn.context_pre_only:
            enc_out = attn.to_add_out(enc_out)
        return hidden_out, enc_out


def run_and_capture(pipe, latents, timestep, enc, pooled, capture_layers):
    """Run forward with attention capture on specified layers."""
    original_procs = {}
    captures = {}
    for li in capture_layers:
        block = pipe.transformer.transformer_blocks[li]
        original_procs[li] = block.attn.processor
        cap = AttentionCaptureHook()
        captures[li] = cap
        block.attn.processor = cap

    with torch.no_grad():
        pipe.transformer(
            hidden_states=latents, timestep=timestep,
            encoder_hidden_states=enc, pooled_projections=pooled,
            return_dict=False)

    attn_maps = {}
    for li in capture_layers:
        attn_maps[li] = captures[li].img_to_txt_attn
        pipe.transformer.transformer_blocks[li].attn.processor = original_procs[li]

    return attn_maps


def plot_attention_overlay(source_img, attn_map, img_h, img_w, title, ax):
    """Overlay attention heatmap on source image."""
    # attn_map: [img_len, txt_len] → sum over txt tokens → [img_len]
    importance = attn_map.sum(dim=-1).numpy()  # [img_len]
    importance = importance.reshape(img_h, img_w)

    # Normalize
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)

    # Resize to image size
    importance_resized = np.array(Image.fromarray((importance * 255).astype(np.uint8)).resize(
        source_img.size, Image.BILINEAR)) / 255.0

    # Overlay
    img_array = np.array(source_img).astype(float) / 255.0
    heatmap = plt.cm.jet(importance_resized)[:, :, :3]
    overlay = 0.5 * img_array + 0.5 * heatmap

    ax.imshow(overlay)
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-3-medium-diffusers')
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--sample_idx', type=int, default=1)
    parser.add_argument('--cfg_scale', type=float, default=1.2)
    args = parser.parse_args()

    device = torch.device("cuda")
    print("Loading pipeline...")
    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # Load sample
    cat_dir = os.path.join(args.dataset_dir, "1_change_object_80")
    df = pd.read_parquet(os.path.join(cat_dir, "V1-00000-of-00001.parquet"))
    row = df.iloc[args.sample_idx]
    source_img = Image.open(io.BytesIO(row['image']['bytes']))
    target_prompt = row['target_prompt'].replace('[', '').replace(']', '')
    source_prompt = row['source_prompt']

    print(f"Source: '{source_prompt}'")
    print(f"Target: '{target_prompt}'")

    w, h = source_img.size
    w, h = w - w % 16, h - h % 16
    source_img = source_img.resize((w, h))

    # Encode
    (cond_e, neg_e, cond_p, neg_p) = pipe.encode_prompt(
        prompt=target_prompt, prompt_2=target_prompt, prompt_3=target_prompt,
        negative_prompt="", negative_prompt_2="", negative_prompt_3="",
        do_classifier_free_guidance=True, device=device)

    # Compute guided embeddings for each layer using full-block approach
    # We need the norm_txt after AdaLayerNorm for cond and uncond
    # For simplicity, just capture attention with cond vs uncond text

    # Encode image
    image_tensor = pipe.image_processor.preprocess(source_img)
    image_tensor = image_tensor.to(device=device, dtype=pipe.vae.dtype)
    latents = pipe.vae.encode(image_tensor).latent_dist.sample()
    latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    latents = latents.to(dtype=cond_e.dtype)

    pipe.scheduler.set_timesteps(25, device=device)
    timesteps, _ = pipe.get_timesteps(25, 0.7, device)
    noise = torch.randn_like(latents)
    latents = pipe.scheduler.scale_noise(latents, timesteps[:1], noise)
    t = timesteps[len(timesteps) // 2].unsqueeze(0)

    patch_size = pipe.transformer.config.patch_size
    img_h = h // (pipe.vae_scale_factor * patch_size)
    img_w = w // (pipe.vae_scale_factor * patch_size)

    all_layers = list(range(24))

    # Capture attention with COND text
    print("Capturing cond attention...")
    attn_cond = run_and_capture(pipe, latents, t, cond_e, cond_p, all_layers)

    # Capture attention with UNCOND text
    print("Capturing uncond attention...")
    attn_uncond = run_and_capture(pipe, latents, t, neg_e, neg_p, all_layers)

    os.makedirs(args.output_dir, exist_ok=True)
    source_img.save(os.path.join(args.output_dir, "source.jpg"))

    # === Figure 1: All 24 layers, cond attention overlay ===
    print("Plotting all layers cond attention...")
    fig, axes = plt.subplots(4, 6, figsize=(24, 16))
    fig.suptitle(f"Cond Attention Overlay per Layer\nTarget: '{target_prompt}'", fontsize=14)
    for li in range(24):
        row_idx, col_idx = li // 6, li % 6
        attn = attn_cond[li][0]  # [img_len, txt_len]
        plot_attention_overlay(source_img, attn, img_h, img_w,
                              f"Layer {li}", axes[row_idx, col_idx])
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "all_layers_cond_attn.png"), dpi=150)
    plt.close()

    # === Figure 2: Cond - Uncond difference (what CFG amplifies) ===
    print("Plotting cond-uncond difference...")
    fig, axes = plt.subplots(4, 6, figsize=(24, 16))
    fig.suptitle(f"Cond - Uncond Attention Difference per Layer (what CFG amplifies)\nTarget: '{target_prompt}'", fontsize=14)
    for li in range(24):
        row_idx, col_idx = li // 6, li % 6
        diff = attn_cond[li][0] - attn_uncond[li][0]  # [img_len, txt_len]
        diff_importance = diff.sum(dim=-1).numpy().reshape(img_h, img_w)

        # Show positive (red) and negative (blue)
        vmax = max(abs(diff_importance.min()), abs(diff_importance.max()))
        if vmax < 1e-8:
            vmax = 1.0
        diff_norm = diff_importance / vmax  # [-1, 1]

        # Overlay on image
        img_array = np.array(source_img).astype(float) / 255.0
        diff_resized = np.array(Image.fromarray(
            ((diff_norm + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        ).resize(source_img.size, Image.BILINEAR)) / 255.0

        heatmap = plt.cm.RdBu_r(diff_resized)[:, :, :3]
        overlay = 0.4 * img_array + 0.6 * heatmap
        axes[row_idx, col_idx].imshow(overlay)
        axes[row_idx, col_idx].set_title(f"Layer {li}", fontsize=9)
        axes[row_idx, col_idx].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "all_layers_diff_attn.png"), dpi=150)
    plt.close()

    # === Figure 3: Top helpful layers detailed view ===
    helpful_layers = [1, 7, 8, 11, 12, 13, 18, 22, 23]
    print(f"Plotting helpful layers: {helpful_layers}")

    fig, axes = plt.subplots(3, len(helpful_layers), figsize=(3*len(helpful_layers), 10))
    fig.suptitle(f"Helpful Layers — Cond / Uncond / Difference\nTarget: '{target_prompt}'", fontsize=14)

    for col, li in enumerate(helpful_layers):
        # Row 0: Cond attention
        attn_c = attn_cond[li][0]
        plot_attention_overlay(source_img, attn_c, img_h, img_w,
                              f"L{li} Cond", axes[0, col])

        # Row 1: Uncond attention
        attn_u = attn_uncond[li][0]
        plot_attention_overlay(source_img, attn_u, img_h, img_w,
                              f"L{li} Uncond", axes[1, col])

        # Row 2: Difference
        diff = attn_c - attn_u
        diff_imp = diff.sum(dim=-1).numpy().reshape(img_h, img_w)
        vmax = max(abs(diff_imp.min()), abs(diff_imp.max()))
        if vmax < 1e-8: vmax = 1.0
        diff_norm = diff_imp / vmax
        img_array = np.array(source_img).astype(float) / 255.0
        diff_resized = np.array(Image.fromarray(
            ((diff_norm + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        ).resize(source_img.size, Image.BILINEAR)) / 255.0
        heatmap = plt.cm.RdBu_r(diff_resized)[:, :, :3]
        overlay = 0.4 * img_array + 0.6 * heatmap
        axes[2, col].imshow(overlay)
        axes[2, col].set_title(f"L{li} Diff", fontsize=9)
        axes[2, col].axis('off')

    axes[0, 0].set_ylabel("Cond", fontsize=12)
    axes[1, 0].set_ylabel("Uncond", fontsize=12)
    axes[2, 0].set_ylabel("Cond-Uncond", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "helpful_layers_detail.png"), dpi=150)
    plt.close()

    # === Figure 4: Token-level analysis for top layers ===
    print("Plotting token importance for helpful layers...")
    fig, axes = plt.subplots(2, len(helpful_layers), figsize=(3*len(helpful_layers), 6))
    fig.suptitle(f"Per-Token Importance: Cond vs Uncond", fontsize=14)

    for col, li in enumerate(helpful_layers):
        attn_c = attn_cond[li][0]   # [img_len, txt_len]
        attn_u = attn_uncond[li][0]

        # Token importance = sum of attention from all image tokens
        cond_imp = attn_c.sum(dim=0).numpy()  # [txt_len]
        uncond_imp = attn_u.sum(dim=0).numpy()

        x = np.arange(len(cond_imp))
        axes[0, col].bar(x, cond_imp, color='blue', alpha=0.7, width=1.0)
        axes[0, col].set_title(f"L{li} Cond", fontsize=8)
        axes[0, col].set_xlim(0, len(cond_imp))

        diff_imp = cond_imp - uncond_imp
        colors = ['red' if d > 0 else 'blue' for d in diff_imp]
        axes[1, col].bar(x, diff_imp, color=colors, alpha=0.7, width=1.0)
        axes[1, col].set_title(f"L{li} Diff", fontsize=8)
        axes[1, col].axhline(y=0, color='black', linewidth=0.3)
        axes[1, col].set_xlim(0, len(cond_imp))

    axes[0, 0].set_ylabel("Token importance", fontsize=10)
    axes[1, 0].set_ylabel("Cond - Uncond", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "helpful_layers_tokens.png"), dpi=150)
    plt.close()

    print(f"\nAll visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
