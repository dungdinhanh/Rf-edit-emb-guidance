"""
Visualize attention maps per layer across multiple timesteps.

Uses consistent colormap (hot) for all plots. Shows absolute attention
strength for cond/uncond and absolute difference magnitude.
"""

import os, sys, io, json, math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline
import pandas as pd


class AttentionCaptureHook:
    def __init__(self):
        self.img_to_txt_attn = None

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

        scale = head_dim ** -0.5
        attn_logits = torch.matmul(full_q, full_k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_logits, dim=-1)
        self.img_to_txt_attn = attn_weights[:, :, :img_len, img_len:].detach().cpu().float().mean(dim=1)

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


def attn_to_spatial(attn_map, img_h, img_w):
    """Convert [img_len, txt_len] → [img_h, img_w] by summing over txt tokens."""
    importance = attn_map.sum(dim=-1).numpy()
    return importance.reshape(img_h, img_w)


def overlay_heatmap(source_img, heatmap, vmin=None, vmax=None):
    """Overlay normalized heatmap on source image using 'hot' colormap."""
    if vmin is None: vmin = heatmap.min()
    if vmax is None: vmax = heatmap.max()
    if vmax - vmin < 1e-8: vmax = vmin + 1.0

    heatmap_norm = (heatmap - vmin) / (vmax - vmin)
    heatmap_norm = np.clip(heatmap_norm, 0, 1)

    heatmap_resized = np.array(Image.fromarray(
        (heatmap_norm * 255).astype(np.uint8)
    ).resize(source_img.size, Image.BILINEAR)) / 255.0

    img_array = np.array(source_img).astype(float) / 255.0
    colormap = plt.cm.hot(heatmap_resized)[:, :, :3]
    return 0.4 * img_array + 0.6 * colormap


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-3-medium-diffusers')
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--sample_idx', type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda")
    print("Loading pipeline...")
    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    cat_dir = os.path.join(args.dataset_dir, "1_change_object_80")
    df = pd.read_parquet(os.path.join(cat_dir, "V1-00000-of-00001.parquet"))
    row = df.iloc[args.sample_idx]
    source_img = Image.open(io.BytesIO(row['image']['bytes']))
    target_prompt = row['target_prompt'].replace('[', '').replace(']', '')
    source_prompt = row['source_prompt']
    print(f"Source: '{source_prompt}' → Target: '{target_prompt}'")

    w, h = source_img.size
    w, h = w - w % 16, h - h % 16
    source_img = source_img.resize((w, h))

    (cond_e, neg_e, cond_p, neg_p) = pipe.encode_prompt(
        prompt=target_prompt, prompt_2=target_prompt, prompt_3=target_prompt,
        negative_prompt="", negative_prompt_2="", negative_prompt_3="",
        do_classifier_free_guidance=True, device=device)

    image_tensor = pipe.image_processor.preprocess(source_img)
    image_tensor = image_tensor.to(device=device, dtype=pipe.vae.dtype)
    latents_clean = pipe.vae.encode(image_tensor).latent_dist.sample()
    latents_clean = (latents_clean - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    latents_clean = latents_clean.to(dtype=cond_e.dtype)

    pipe.scheduler.set_timesteps(25, device=device)
    timesteps, _ = pipe.get_timesteps(25, 0.7, device)

    patch_size = pipe.transformer.config.patch_size
    img_h = h // (pipe.vae_scale_factor * patch_size)
    img_w = w // (pipe.vae_scale_factor * patch_size)

    os.makedirs(args.output_dir, exist_ok=True)
    source_img.save(os.path.join(args.output_dir, "source.jpg"))

    all_layers = list(range(24))
    # Select timesteps: early (step 1), early-mid (step 5), mid (step 9), late-mid (step 13), late (step 17)
    total_steps = len(timesteps)
    step_indices = [0, total_steps // 4, total_steps // 2, 3 * total_steps // 4, total_steps - 1]
    step_names = ["early", "early_mid", "mid", "late_mid", "late"]

    for si, (step_idx, step_name) in enumerate(zip(step_indices, step_names)):
        t = timesteps[step_idx]
        t_val = t.item()
        print(f"\n=== Step {step_idx} ({step_name}, t={t_val:.0f}) ===")

        # Add noise at this timestep level
        noise = torch.randn_like(latents_clean)
        latents = pipe.scheduler.scale_noise(latents_clean, t.unsqueeze(0), noise)

        t_input = t.unsqueeze(0)

        # Capture cond and uncond attention
        attn_cond = run_and_capture(pipe, latents, t_input, cond_e, cond_p, all_layers)
        attn_uncond = run_and_capture(pipe, latents, t_input, neg_e, neg_p, all_layers)

        # === Figure: All layers at this timestep (3 rows: cond, uncond, |diff|) ===
        fig, axes = plt.subplots(3, 24, figsize=(72, 9))
        fig.suptitle(f"Step {step_idx} ({step_name}, t={t_val:.0f}) — Target: '{target_prompt}'", fontsize=16)

        # Compute global vmin/vmax for consistent coloring within each row
        all_cond_spatial = []
        all_uncond_spatial = []
        all_diff_spatial = []
        for li in range(24):
            cs = attn_to_spatial(attn_cond[li][0], img_h, img_w)
            us = attn_to_spatial(attn_uncond[li][0], img_h, img_w)
            ds = np.abs(cs - us)
            all_cond_spatial.append(cs)
            all_uncond_spatial.append(us)
            all_diff_spatial.append(ds)

        cond_vmax = max(cs.max() for cs in all_cond_spatial)
        uncond_vmax = max(us.max() for us in all_uncond_spatial)
        attn_vmax = max(cond_vmax, uncond_vmax)
        diff_vmax = max(ds.max() for ds in all_diff_spatial)

        for li in range(24):
            # Row 0: Cond attention
            overlay = overlay_heatmap(source_img, all_cond_spatial[li], vmin=0, vmax=attn_vmax)
            axes[0, li].imshow(overlay)
            axes[0, li].set_title(f"L{li}", fontsize=7)
            axes[0, li].axis('off')

            # Row 1: Uncond attention
            overlay = overlay_heatmap(source_img, all_uncond_spatial[li], vmin=0, vmax=attn_vmax)
            axes[1, li].imshow(overlay)
            axes[1, li].axis('off')

            # Row 2: |Cond - Uncond| (absolute difference)
            overlay = overlay_heatmap(source_img, all_diff_spatial[li], vmin=0, vmax=diff_vmax)
            axes[2, li].imshow(overlay)
            axes[2, li].axis('off')

        axes[0, 0].set_ylabel("Cond", fontsize=10, rotation=0, labelpad=40)
        axes[1, 0].set_ylabel("Uncond", fontsize=10, rotation=0, labelpad=40)
        axes[2, 0].set_ylabel("|Diff|", fontsize=10, rotation=0, labelpad=40)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(args.output_dir, f"step{step_idx}_{step_name}_all_layers.png"), dpi=100)
        plt.close()

        # === Figure: Helpful layers detail at this timestep ===
        helpful = [1, 7, 8, 11, 12, 13, 18, 22, 23]
        fig, axes = plt.subplots(3, len(helpful), figsize=(3 * len(helpful), 9))
        fig.suptitle(f"Helpful Layers — Step {step_idx} ({step_name}, t={t_val:.0f})", fontsize=14)

        for col, li in enumerate(helpful):
            cs = all_cond_spatial[li]
            us = all_uncond_spatial[li]
            ds = all_diff_spatial[li]

            axes[0, col].imshow(overlay_heatmap(source_img, cs, vmin=0, vmax=attn_vmax))
            axes[0, col].set_title(f"L{li} Cond", fontsize=9)
            axes[0, col].axis('off')

            axes[1, col].imshow(overlay_heatmap(source_img, us, vmin=0, vmax=attn_vmax))
            axes[1, col].set_title(f"L{li} Uncond", fontsize=9)
            axes[1, col].axis('off')

            axes[2, col].imshow(overlay_heatmap(source_img, ds, vmin=0, vmax=diff_vmax))
            axes[2, col].set_title(f"L{li} |Diff|", fontsize=9)
            axes[2, col].axis('off')

        axes[0, 0].set_ylabel("Cond", fontsize=10, rotation=0, labelpad=35)
        axes[1, 0].set_ylabel("Uncond", fontsize=10, rotation=0, labelpad=35)
        axes[2, 0].set_ylabel("|Diff|", fontsize=10, rotation=0, labelpad=35)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(os.path.join(args.output_dir, f"step{step_idx}_{step_name}_helpful.png"), dpi=150)
        plt.close()

    # === Summary figure: Diff magnitude across layers and timesteps ===
    print("\nPlotting summary heatmap...")

    # Re-collect diff magnitudes for all steps and layers
    diff_magnitudes = np.zeros((len(step_indices), 24))
    for si, step_idx in enumerate(step_indices):
        t = timesteps[step_idx]
        noise = torch.randn_like(latents_clean)
        latents = pipe.scheduler.scale_noise(latents_clean, t.unsqueeze(0), noise)
        t_input = t.unsqueeze(0)

        attn_c = run_and_capture(pipe, latents, t_input, cond_e, cond_p, all_layers)
        attn_u = run_and_capture(pipe, latents, t_input, neg_e, neg_p, all_layers)

        for li in range(24):
            cs = attn_to_spatial(attn_c[li][0], img_h, img_w)
            us = attn_to_spatial(attn_u[li][0], img_h, img_w)
            diff_magnitudes[si, li] = np.abs(cs - us).mean()

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    im = ax.imshow(diff_magnitudes, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Timestep")
    ax.set_xticks(range(24))
    ax.set_yticks(range(len(step_indices)))
    ax.set_yticklabels([f"Step {si} ({sn})" for si, sn in zip(step_indices, step_names)])
    ax.set_title("Mean |Cond-Uncond| Attention Difference — Layer × Timestep")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "summary_diff_heatmap.png"), dpi=150)
    plt.close()

    print(f"\nAll visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
