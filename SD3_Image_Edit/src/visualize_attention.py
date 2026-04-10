"""
Visualize attention maps with and without embedding guidance.

Hooks into SD3 transformer blocks to extract image-to-text attention weights,
then visualizes them as heatmaps comparing standard vs guided embeddings.
"""

import os
import json
import io
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline
import pandas as pd


class AttnCaptureProcessor:
    """Replaces JointAttnProcessor2_0 to capture attention weights."""

    def __init__(self):
        self.attn_weights = None  # Will store [B, H, L, L]

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, *args, **kwargs):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]
        inner_dim = attn.to_q.out_features
        head_dim = inner_dim // attn.heads
        H = attn.heads

        # Image projections
        query = attn.to_q(hidden_states).view(batch_size, -1, H, head_dim).transpose(1, 2)
        key = attn.to_k(hidden_states).view(batch_size, -1, H, head_dim).transpose(1, 2)
        value = attn.to_v(hidden_states).view(batch_size, -1, H, head_dim).transpose(1, 2)

        # Text projections
        txt_q = attn.add_q_proj(encoder_hidden_states).view(batch_size, -1, H, head_dim).transpose(1, 2)
        txt_k = attn.add_k_proj(encoder_hidden_states).view(batch_size, -1, H, head_dim).transpose(1, 2)
        txt_v = attn.add_v_proj(encoder_hidden_states).view(batch_size, -1, H, head_dim).transpose(1, 2)

        img_len = query.shape[2]
        txt_len = txt_q.shape[2]

        # Full joint Q, K, V
        full_q = torch.cat([query, txt_q], dim=2)
        full_k = torch.cat([key, txt_k], dim=2)
        full_v = torch.cat([value, txt_v], dim=2)

        # Compute attention weights explicitly
        scale = head_dim ** -0.5
        attn_logits = torch.matmul(full_q, full_k.transpose(-2, -1)) * scale
        attn_w = F.softmax(attn_logits, dim=-1)

        # Store: image-to-text attention [B, H, img_len, txt_len]
        self.attn_weights = attn_w[:, :, :img_len, img_len:].detach().cpu()

        # Compute output
        out = torch.matmul(attn_w, full_v)
        out = out.transpose(1, 2).reshape(batch_size, -1, inner_dim).to(hidden_states.dtype)

        hidden_states_out = out[:, :residual.shape[1]]
        encoder_hidden_states_out = out[:, residual.shape[1]:]

        hidden_states_out = attn.to_out[0](hidden_states_out)
        hidden_states_out = attn.to_out[1](hidden_states_out)
        if not attn.context_pre_only:
            encoder_hidden_states_out = attn.to_add_out(encoder_hidden_states_out)

        if input_ndim == 4:
            hidden_states_out = hidden_states_out.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states_out = encoder_hidden_states_out.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states_out, encoder_hidden_states_out


def norm_emb_guidance(cond, uncond, alpha):
    cond_f = cond.float()
    uncond_f = uncond.float()
    v = (1.0 + alpha) * cond_f - alpha * uncond_f
    cond_norm = cond_f.norm(dim=-1, keepdim=True)
    v_norm = v.norm(dim=-1, keepdim=True) + 1e-8
    return (v * (cond_norm / v_norm)).to(cond.dtype)


def run_and_capture(pipe, latents, timestep, encoder_hidden_states, pooled_projections,
                    capture_layers):
    """Run one denoising step with attention capture on specified layers."""
    # Install capture processors
    original_processors = {}
    capture_procs = {}
    for idx in capture_layers:
        block = pipe.transformer.transformer_blocks[idx]
        original_processors[idx] = block.attn.processor
        cap = AttnCaptureProcessor()
        capture_procs[idx] = cap
        block.attn.processor = cap

    # Forward pass
    with torch.no_grad():
        _ = pipe.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            return_dict=False,
        )

    # Collect attention maps
    attn_maps = {}
    for idx in capture_layers:
        attn_maps[idx] = capture_procs[idx].attn_weights

    # Restore
    for idx in capture_layers:
        pipe.transformer.transformer_blocks[idx].attn.processor = original_processors[idx]

    return attn_maps


def visualize_comparison(attn_cond, attn_guided, layer_idx, save_path,
                         prompt, emb_alpha, img_hw=None):
    """Create side-by-side attention heatmaps."""
    # Average over heads: [B, img_len, txt_len]
    cond_avg = attn_cond[0].float().mean(dim=0).numpy()  # [img_len, txt_len]
    guided_avg = attn_guided[0].float().mean(dim=0).numpy()

    # Also compute per-token importance (sum over image tokens)
    cond_importance = cond_avg.sum(axis=0)  # [txt_len]
    guided_importance = guided_avg.sum(axis=0)
    diff_importance = guided_importance - cond_importance

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"Layer {layer_idx} | Prompt: '{prompt}' | α={emb_alpha}", fontsize=14)

    # Row 1: Full attention maps
    im0 = axes[0, 0].imshow(cond_avg, aspect='auto', cmap='hot', interpolation='nearest')
    axes[0, 0].set_title("Cond only (no guidance)")
    axes[0, 0].set_xlabel("Text tokens")
    axes[0, 0].set_ylabel("Image tokens")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(guided_avg, aspect='auto', cmap='hot', interpolation='nearest')
    axes[0, 1].set_title(f"Norm emb guidance α={emb_alpha}")
    axes[0, 1].set_xlabel("Text tokens")
    axes[0, 1].set_ylabel("Image tokens")
    plt.colorbar(im1, ax=axes[0, 1])

    diff = guided_avg - cond_avg
    vmax = max(abs(diff.min()), abs(diff.max()))
    im2 = axes[0, 2].imshow(diff, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                             interpolation='nearest')
    axes[0, 2].set_title("Difference (guided - cond)")
    axes[0, 2].set_xlabel("Text tokens")
    axes[0, 2].set_ylabel("Image tokens")
    plt.colorbar(im2, ax=axes[0, 2])

    # Row 2: Per-text-token importance
    x = np.arange(len(cond_importance))
    axes[1, 0].bar(x, cond_importance, color='blue', alpha=0.7)
    axes[1, 0].set_title("Token importance (cond)")
    axes[1, 0].set_xlabel("Text token index")
    axes[1, 0].set_ylabel("Sum of attention from all image tokens")

    axes[1, 1].bar(x, guided_importance, color='red', alpha=0.7)
    axes[1, 1].set_title(f"Token importance (guided α={emb_alpha})")
    axes[1, 1].set_xlabel("Text token index")

    colors = ['green' if d > 0 else 'red' for d in diff_importance]
    axes[1, 2].bar(x, diff_importance, color=colors, alpha=0.7)
    axes[1, 2].set_title("Importance change (guided - cond)")
    axes[1, 2].set_xlabel("Text token index")
    axes[1, 2].axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def visualize_spatial(attn_cond, attn_guided, layer_idx, save_path,
                      prompt, emb_alpha, img_h, img_w):
    """Visualize spatial attention pattern for top text tokens."""
    cond_avg = attn_cond[0].float().mean(dim=0).numpy()  # [img_len, txt_len]
    guided_avg = attn_guided[0].float().mean(dim=0).numpy()

    # Find top-5 text tokens with most change
    diff_importance = guided_avg.sum(axis=0) - cond_avg.sum(axis=0)
    top_changed = np.argsort(np.abs(diff_importance))[-5:][::-1]

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle(f"Layer {layer_idx} — Spatial attention for top-5 changed tokens | α={emb_alpha}", fontsize=14)

    for i, tok_idx in enumerate(top_changed):
        # Reshape to spatial grid
        cond_spatial = cond_avg[:, tok_idx].reshape(img_h, img_w)
        guided_spatial = guided_avg[:, tok_idx].reshape(img_h, img_w)

        axes[0, i].imshow(cond_spatial, cmap='hot')
        axes[0, i].set_title(f"Cond | tok {tok_idx}\nΔ={diff_importance[tok_idx]:.4f}")
        axes[0, i].axis('off')

        axes[1, i].imshow(guided_spatial, cmap='hot')
        axes[1, i].set_title(f"Guided α={emb_alpha}")
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel("Cond only", fontsize=12)
    axes[1, 0].set_ylabel("Norm emb guided", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-3-medium-diffusers')
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output_dir', default='attention_vis')
    parser.add_argument('--sample_idx', type=int, default=1)
    parser.add_argument('--emb_alpha', type=float, default=0.8)
    parser.add_argument('--layers', type=str, default='0,5,11,17,23',
                        help='Comma-separated layer indices to visualize')
    parser.add_argument('--timestep_idx', type=int, default=5,
                        help='Which timestep to capture (0=first, higher=later)')
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

    # Resize
    w, h = source_img.size
    w, h = w - w % 16, h - h % 16
    source_img = source_img.resize((w, h))

    # Encode
    (cond_embeds, neg_embeds,
     cond_pooled, neg_pooled) = pipe.encode_prompt(
        prompt=target_prompt, prompt_2=target_prompt, prompt_3=target_prompt,
        negative_prompt="", negative_prompt_2="", negative_prompt_3="",
        do_classifier_free_guidance=True,
        device=device,
    )

    # Guided embeddings
    guided_embeds = norm_emb_guidance(cond_embeds, neg_embeds, args.emb_alpha)
    guided_pooled = norm_emb_guidance(cond_pooled, neg_pooled, args.emb_alpha)

    # Encode image
    image_tensor = pipe.image_processor.preprocess(source_img)
    image_tensor = image_tensor.to(device=device, dtype=pipe.vae.dtype)
    latents = pipe.vae.encode(image_tensor).latent_dist.sample()
    latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    latents = latents.to(dtype=cond_embeds.dtype)

    # Setup timesteps
    pipe.scheduler.set_timesteps(25, device=device)
    timesteps, _ = pipe.get_timesteps(25, 0.7, device)

    # Add noise
    noise = torch.randn_like(latents)
    latents = pipe.scheduler.scale_noise(latents, timesteps[:1], noise)

    # Pick timestep
    t_idx = min(args.timestep_idx, len(timesteps) - 1)
    t = timesteps[t_idx]
    timestep_tensor = t.expand(latents.shape[0])

    capture_layers = [int(x) for x in args.layers.split(',')]
    print(f"Capturing layers: {capture_layers}, timestep {t_idx} (t={t.item():.1f})")

    # Compute image spatial dimensions (after VAE + patchify)
    patch_size = pipe.transformer.config.patch_size
    img_h = h // (pipe.vae_scale_factor * patch_size)
    img_w = w // (pipe.vae_scale_factor * patch_size)
    print(f"Image tokens: {img_h}x{img_w} = {img_h * img_w}")

    # Run with cond only
    print("\nCapturing cond-only attention...")
    attn_cond = run_and_capture(pipe, latents, timestep_tensor, cond_embeds, cond_pooled, capture_layers)

    # Run with guided embeddings
    print(f"Capturing guided (α={args.emb_alpha}) attention...")
    attn_guided = run_and_capture(pipe, latents, timestep_tensor, guided_embeds, guided_pooled, capture_layers)

    # Visualize
    os.makedirs(args.output_dir, exist_ok=True)

    # Save source image
    source_img.save(os.path.join(args.output_dir, "source.jpg"))

    for layer_idx in capture_layers:
        # Full heatmap comparison
        visualize_comparison(
            attn_cond[layer_idx], attn_guided[layer_idx],
            layer_idx,
            os.path.join(args.output_dir, f"attn_layer{layer_idx}_t{t_idx}.png"),
            target_prompt, args.emb_alpha,
        )
        # Spatial attention for top changed tokens
        visualize_spatial(
            attn_cond[layer_idx], attn_guided[layer_idx],
            layer_idx,
            os.path.join(args.output_dir, f"spatial_layer{layer_idx}_t{t_idx}.png"),
            target_prompt, args.emb_alpha,
            img_h, img_w,
        )

    print(f"\nDone! Visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
