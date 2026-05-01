"""
Compare hidden state trajectories: Standard CFG vs Full-block LayerCFG
across multiple denoising timesteps.

For each timestep, uses the ACTUAL intermediate latent from the CFG
denoising trajectory as input, then runs both methods and compares
the hidden states at every block.
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


def capture_block_hiddens_cfg(pipe, latents, timestep, cond_e, neg_e, cond_p, neg_p, cfg_scale=7.0):
    """Run standard CFG and capture per-block hidden states for the COND path."""
    transformer = pipe.transformer

    # Batch=2 input
    latent_input = torch.cat([latents, latents])
    timestep_input = timestep.expand(2)
    enc_input = torch.cat([neg_e, cond_e])
    pooled_input = torch.cat([neg_p, cond_p])

    block_hiddens = []
    hooks = []

    for i, block in enumerate(transformer.transformer_blocks):
        def make_hook(idx):
            def hook(module, args, kwargs, output):
                if isinstance(output, tuple) and len(output) == 2:
                    enc_out, hidden_out = output
                    # hidden_out is [2, ...] — uncond then cond
                    # Take cond path (index 1)
                    block_hiddens.append(hidden_out[1:2].detach().float().cpu())
                return output
            return hook
        h = block.register_forward_hook(make_hook(i), with_kwargs=True)
        hooks.append(h)

    with torch.no_grad():
        noise_pred = transformer(
            hidden_states=latent_input, timestep=timestep_input,
            encoder_hidden_states=enc_input, pooled_projections=pooled_input,
            return_dict=False)[0]

    for h in hooks:
        h.remove()

    # CFG combination happens at prediction level
    uncond_pred, cond_pred = noise_pred.chunk(2)
    cfg_pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)

    return block_hiddens, cfg_pred


def capture_block_hiddens_layercfg(pipe, latents, timestep, cond_e, neg_e, cond_p,
                                    layer_scales):
    """Run full-block LayerCFG and capture per-block hidden states."""
    transformer = pipe.transformer
    hidden_states = latents
    height, width = hidden_states.shape[-2:]
    hidden_states = transformer.pos_embed(hidden_states)
    temb = transformer.time_text_embed(timestep, cond_p)
    enc_cond = transformer.context_embedder(cond_e)
    enc_uncond = transformer.context_embedder(neg_e)

    block_hiddens = []

    for idx, block in enumerate(transformer.transformer_blocks):
        scale_i = layer_scales[idx]
        if block.context_pre_only:
            _, hidden_states = block(hidden_states=hidden_states,
                encoder_hidden_states=enc_cond, temb=temb)
        elif scale_i == 1.0:
            enc_cond, hidden_states = block(hidden_states=hidden_states,
                encoder_hidden_states=enc_cond, temb=temb)
        else:
            enc_cond_out, hidden_cond = block(hidden_states=hidden_states,
                encoder_hidden_states=enc_cond, temb=temb)
            enc_uncond_out, hidden_uncond = block(hidden_states=hidden_states,
                encoder_hidden_states=enc_uncond, temb=temb)
            hidden_states = hidden_uncond + scale_i * (hidden_cond - hidden_uncond)
            enc_cond = enc_cond_out
            enc_uncond = enc_uncond_out

        block_hiddens.append(hidden_states.detach().float().cpu())

    # Get prediction
    hidden_states = transformer.norm_out(hidden_states, temb)
    hidden_states = transformer.proj_out(hidden_states)
    patch_size = transformer.config.patch_size
    h_out = height // patch_size
    w_out = width // patch_size
    hidden_states = hidden_states.reshape(
        hidden_states.shape[0], h_out, w_out,
        patch_size, patch_size, transformer.out_channels)
    hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
    lcfg_pred = hidden_states.reshape(
        hidden_states.shape[0], transformer.out_channels,
        h_out * patch_size, w_out * patch_size)

    return block_hiddens, lcfg_pred


@torch.no_grad()
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-3-medium-diffusers')
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--sample_idx', type=int, default=1)
    parser.add_argument('--cfg_scale', type=float, default=7.0)
    parser.add_argument('--lcfg_scale', type=float, default=1.2)
    args = parser.parse_args()

    device = torch.device("cuda")
    print("Loading pipeline...")
    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    num_layers = len(pipe.transformer.transformer_blocks)

    # Load sample
    cat_dir = os.path.join(args.dataset_dir, "1_change_object_80")
    df = pd.read_parquet(os.path.join(cat_dir, "V1-00000-of-00001.parquet"))
    row = df.iloc[args.sample_idx]
    source_img = Image.open(io.BytesIO(row['image']['bytes']))
    target_prompt = row['target_prompt'].replace('[', '').replace(']', '')
    print(f"Target: '{target_prompt}'")

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
    noise = torch.randn_like(latents_clean)
    latents_start = pipe.scheduler.scale_noise(latents_clean, timesteps[:1], noise)

    # LayerCFG scales
    layer_scales = [args.lcfg_scale] * num_layers
    layer_scales[23] = 1.0  # context_pre_only

    os.makedirs(args.output_dir, exist_ok=True)

    # Run ACTUAL CFG denoising to get intermediate latents
    print("Running CFG denoising to collect intermediate latents...")
    cfg_latents = latents_start.clone()
    cfg_intermediates = {}

    total_steps = len(timesteps)
    capture_steps = [0, total_steps // 4, total_steps // 2, 3 * total_steps // 4, total_steps - 1]
    step_names = ["early", "early_mid", "mid", "late_mid", "late"]

    for i, t in enumerate(timesteps):
        if i in capture_steps:
            cfg_intermediates[i] = cfg_latents.clone()

        latent_input = torch.cat([cfg_latents, cfg_latents])
        timestep_input = t.expand(2)
        enc_input = torch.cat([neg_e, cond_e])
        pooled_input = torch.cat([neg_p, cond_p])
        noise_pred = pipe.transformer(
            hidden_states=latent_input, timestep=timestep_input,
            encoder_hidden_states=enc_input, pooled_projections=pooled_input,
            return_dict=False)[0]
        uncond_pred, cond_pred = noise_pred.chunk(2)
        guided_pred = uncond_pred + args.cfg_scale * (cond_pred - uncond_pred)
        cfg_latents = pipe.scheduler.step(guided_pred, t, cfg_latents, return_dict=False)[0]

    # Compare at each captured timestep
    print("Comparing hidden states at each timestep...")

    all_stats = {}

    for si, (step_idx, step_name) in enumerate(zip(capture_steps, step_names)):
        t = timesteps[step_idx]
        t_val = t.item()
        input_latents = cfg_intermediates[step_idx]

        print(f"\n=== Step {step_idx} ({step_name}, t={t_val:.0f}) ===")

        # Run standard CFG
        cfg_hiddens, cfg_pred = capture_block_hiddens_cfg(
            pipe, input_latents, t.unsqueeze(0), cond_e, neg_e, cond_p, neg_p, args.cfg_scale)

        # Run LayerCFG with same input
        lcfg_hiddens, lcfg_pred = capture_block_hiddens_layercfg(
            pipe, input_latents, t.unsqueeze(0), cond_e, neg_e, cond_p, layer_scales)

        # Compare predictions
        pred_diff = (cfg_pred.float().cpu() - lcfg_pred.float().cpu()).norm().item()
        pred_cos = F.cosine_similarity(
            cfg_pred.float().cpu().flatten().unsqueeze(0),
            lcfg_pred.float().cpu().flatten().unsqueeze(0)).item()
        print(f"  Prediction diff: {pred_diff:.2f}, cosine: {pred_cos:.6f}")

        # Compare per-block hidden states
        step_stats = []
        for li in range(min(len(cfg_hiddens), len(lcfg_hiddens))):
            ch = cfg_hiddens[li]
            lh = lcfg_hiddens[li]
            l2 = (ch - lh).norm().item()
            cos = F.cosine_similarity(ch.flatten().unsqueeze(0), lh.flatten().unsqueeze(0)).item()
            cn = ch.norm().item()
            ln = lh.norm().item()
            step_stats.append({
                'l2_diff': l2,
                'cosine': cos,
                'cfg_norm': cn,
                'lcfg_norm': ln,
                'norm_ratio': ln / (cn + 1e-8),
                'relative_diff': l2 / (cn + 1e-8),
            })

        all_stats[step_idx] = {
            'step_name': step_name,
            't': t_val,
            'layers': step_stats,
            'pred_diff': pred_diff,
            'pred_cosine': pred_cos,
        }

    # Write report
    report_path = os.path.join(args.output_dir, "cfg_vs_layercfg_comparison.md")
    with open(report_path, 'w') as f:
        f.write("# CFG vs LayerCFG Hidden State Comparison\n\n")
        f.write(f"CFG scale={args.cfg_scale}, LayerCFG scale={args.lcfg_scale}\n")
        f.write(f"Same input latents (from CFG trajectory) for both methods.\n\n")

        for step_idx in capture_steps:
            s = all_stats[step_idx]
            f.write(f"\n## Step {step_idx} ({s['step_name']}, t={s['t']:.0f})\n\n")
            f.write(f"Prediction diff: {s['pred_diff']:.2f}, cosine: {s['pred_cosine']:.6f}\n\n")

            f.write("| Layer | L2 Diff | Relative Diff | Cosine Sim | CFG Norm | LCFG Norm | Norm Ratio |\n")
            f.write("|:-----:|:-------:|:-------------:|:----------:|:--------:|:---------:|:----------:|\n")
            for li, ls in enumerate(s['layers']):
                f.write(f"| {li:2d} | {ls['l2_diff']:.1f} | {ls['relative_diff']:.4f} | {ls['cosine']:.6f} | {ls['cfg_norm']:.1f} | {ls['lcfg_norm']:.1f} | {ls['norm_ratio']:.4f} |\n")

    # Plot: relative diff across layers x timesteps
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    rel_diffs = np.zeros((len(capture_steps), num_layers))
    cosines = np.zeros((len(capture_steps), num_layers))
    for si, step_idx in enumerate(capture_steps):
        for li, ls in enumerate(all_stats[step_idx]['layers']):
            rel_diffs[si, li] = ls['relative_diff']
            cosines[si, li] = 1.0 - ls['cosine']

    im0 = axes[0].imshow(rel_diffs, aspect='auto', cmap='hot', interpolation='nearest')
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Timestep")
    axes[0].set_xticks(range(0, num_layers, 2))
    axes[0].set_yticks(range(len(capture_steps)))
    axes[0].set_yticklabels([f"Step {si} ({sn})" for si, sn in zip(capture_steps, step_names)])
    axes[0].set_title("Relative L2 Diff (||cfg - lcfg|| / ||cfg||)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(cosines, aspect='auto', cmap='hot', interpolation='nearest')
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Timestep")
    axes[1].set_xticks(range(0, num_layers, 2))
    axes[1].set_yticks(range(len(capture_steps)))
    axes[1].set_yticklabels([f"Step {si} ({sn})" for si, sn in zip(capture_steps, step_names)])
    axes[1].set_title("1 - Cosine Similarity (0=identical)")
    plt.colorbar(im1, ax=axes[1])

    plt.suptitle(f"CFG vs LayerCFG Trajectory Divergence\nCFG scale={args.cfg_scale}, LCFG scale={args.lcfg_scale}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "cfg_vs_layercfg_heatmap.png"), dpi=150)
    plt.close()

    # Line plot: per-layer diff at each timestep
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for si, (step_idx, step_name) in enumerate(zip(capture_steps, step_names)):
        diffs = [all_stats[step_idx]['layers'][li]['relative_diff'] for li in range(num_layers)]
        ax.plot(range(num_layers), diffs, label=f"Step {step_idx} ({step_name})", marker='o', markersize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Relative L2 Diff")
    ax.set_title("Where CFG and LayerCFG trajectories diverge")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "cfg_vs_layercfg_lines.png"), dpi=150)
    plt.close()

    print(f"\nReport: {report_path}")
    print(f"Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
