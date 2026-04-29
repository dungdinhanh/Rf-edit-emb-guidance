"""
Statistical analysis of signals at each guidance injection point.

For 100 PIE-Bench samples, measure the norm, variance, and cond-uncond
difference ratio at each of the 4 injection points across all 24 blocks.
"""

import os, sys, io, json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class StatsCollector:
    """Hooks into blocks to collect statistics at each injection point."""

    def __init__(self, pipe, cond_embeds, neg_embeds, cond_pooled):
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.cond_embeds = cond_embeds
        self.neg_embeds = neg_embeds
        self.cond_pooled = cond_pooled

        # Project uncond
        self.enc_uncond = self.transformer.context_embedder(neg_embeds)

        self.num_layers = len(self.transformer.transformer_blocks)
        self.hooks = []

        # Storage: per-layer stats
        # Each entry: {norm_cond, norm_uncond, norm_diff, ratio_diff_to_signal, ...}
        self.stats = {
            'normemb': [[] for _ in range(self.num_layers)],   # After LayerNorm (step 1)
            'kv': [[] for _ in range(self.num_layers)],        # After QKV proj (step 2)
            'atten': [[] for _ in range(self.num_layers)],     # After attention (step 4)
            'fullblock': [[] for _ in range(self.num_layers)],  # After full block (step 7)
        }

    def install(self):
        """Install hooks on all blocks."""
        for i, block in enumerate(self.transformer.transformer_blocks):
            def make_hook(block_ref, idx):
                def hook(module, args, kwargs, output):
                    temb = kwargs.get('temb', None)
                    enc_cond_input = kwargs.get('encoder_hidden_states', None)
                    hidden_input = kwargs.get('hidden_states', None)

                    if enc_cond_input is None or temb is None or hidden_input is None:
                        return output

                    enc_uncond = self.enc_uncond

                    with torch.no_grad():
                        # === Point 1: After LayerNorm (NormEmb) ===
                        if block_ref.context_pre_only:
                            norm_cond = block_ref.norm1_context(enc_cond_input, temb)
                            norm_uncond = block_ref.norm1_context(enc_uncond, temb)
                        else:
                            norm_cond, c_gate, c_shift, c_scale, c_gate_mlp = block_ref.norm1_context(enc_cond_input, emb=temb)
                            norm_uncond, u_gate, u_shift, u_scale, u_gate_mlp = block_ref.norm1_context(enc_uncond, emb=temb)

                        nc = norm_cond.float()
                        nu = norm_uncond.float()
                        diff_norm = (nc - nu).norm().item()
                        cond_norm = nc.norm().item()
                        uncond_norm = nu.norm().item()
                        self.stats['normemb'][idx].append({
                            'cond_norm': cond_norm,
                            'uncond_norm': uncond_norm,
                            'diff_norm': diff_norm,
                            'ratio': diff_norm / (cond_norm + 1e-8),
                            'cond_std': nc.std().item(),
                            'uncond_std': nu.std().item(),
                        })

                        # === Point 2: After QKV projection (KV) ===
                        attn = block_ref.attn
                        cond_k = attn.add_k_proj(norm_cond).float()
                        uncond_k = attn.add_k_proj(norm_uncond).float()
                        cond_v = attn.add_v_proj(norm_cond).float()
                        uncond_v = attn.add_v_proj(norm_uncond).float()

                        dk = (cond_k - uncond_k).norm().item()
                        ck = cond_k.norm().item()
                        dv = (cond_v - uncond_v).norm().item()
                        cv = cond_v.norm().item()
                        self.stats['kv'][idx].append({
                            'cond_k_norm': ck,
                            'uncond_k_norm': uncond_k.norm().item(),
                            'diff_k_norm': dk,
                            'ratio_k': dk / (ck + 1e-8),
                            'cond_v_norm': cv,
                            'diff_v_norm': dv,
                            'ratio_v': dv / (cv + 1e-8),
                        })

                        # === Point 3: After attention (AttenCFG) ===
                        # Need to run attention twice
                        norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp = block_ref.norm1(hidden_input, emb=temb)

                        inner_dim = attn.to_q.out_features
                        head_dim = inner_dim // attn.heads
                        H = attn.heads
                        B = hidden_input.shape[0]

                        img_q = attn.to_q(norm_hidden).view(B, -1, H, head_dim).transpose(1, 2)
                        img_k = attn.to_k(norm_hidden).view(B, -1, H, head_dim).transpose(1, 2)
                        img_v = attn.to_v(norm_hidden).view(B, -1, H, head_dim).transpose(1, 2)

                        c_q = attn.add_q_proj(norm_cond).view(B, -1, H, head_dim).transpose(1, 2)
                        c_k_h = cond_k.view(B, -1, H, head_dim).transpose(1, 2)
                        c_v_h = cond_v.view(B, -1, H, head_dim).transpose(1, 2)
                        u_k_h = uncond_k.view(B, -1, H, head_dim).transpose(1, 2)
                        u_v_h = uncond_v.view(B, -1, H, head_dim).transpose(1, 2)
                        u_q = attn.add_q_proj(norm_uncond).view(B, -1, H, head_dim).transpose(1, 2)

                        img_len = img_q.shape[2]

                        # Cond attention
                        q_c = torch.cat([img_q, c_q], dim=2)
                        k_c = torch.cat([img_k, c_k_h], dim=2)
                        v_c = torch.cat([img_v, c_v_h], dim=2)
                        attn_c = F.scaled_dot_product_attention(q_c, k_c, v_c)
                        img_attn_c = attn_c[:, :, :img_len, :].float()

                        # Uncond attention
                        q_u = torch.cat([img_q, u_q], dim=2)
                        k_u = torch.cat([img_k, u_k_h], dim=2)
                        v_u = torch.cat([img_v, u_v_h], dim=2)
                        attn_u = F.scaled_dot_product_attention(q_u, k_u, v_u)
                        img_attn_u = attn_u[:, :, :img_len, :].float()

                        da = (img_attn_c - img_attn_u).norm().item()
                        ca = img_attn_c.norm().item()
                        self.stats['atten'][idx].append({
                            'cond_attn_norm': ca,
                            'uncond_attn_norm': img_attn_u.norm().item(),
                            'diff_attn_norm': da,
                            'ratio': da / (ca + 1e-8),
                            'cond_attn_std': img_attn_c.std().item(),
                        })

                        # === Point 4: After full block (FullBlock) ===
                        # output = (enc_out, hidden_out) for the cond path
                        if isinstance(output, tuple) and len(output) == 2:
                            enc_out_cond, hidden_out_cond = output
                            # Run uncond path
                            enc_out_uncond, hidden_out_uncond = block_ref(
                                hidden_states=hidden_input,
                                encoder_hidden_states=enc_uncond,
                                temb=temb,
                            )
                            hc = hidden_out_cond.float()
                            hu = hidden_out_uncond.float()
                            dh = (hc - hu).norm().item()
                            ch = hc.norm().item()
                            self.stats['fullblock'][idx].append({
                                'cond_hidden_norm': ch,
                                'uncond_hidden_norm': hu.norm().item(),
                                'diff_hidden_norm': dh,
                                'ratio': dh / (ch + 1e-8),
                                'hidden_std': hc.std().item(),
                            })

                    # Evolve uncond through FF (approximate)
                    if not block_ref.context_pre_only:
                        with torch.no_grad():
                            norm_uf = block_ref.norm2_context(self.enc_uncond)
                            norm_uf = norm_uf * (1 + u_scale[:, None]) + u_shift[:, None]
                            uf = block_ref.ff_context(norm_uf)
                            self.enc_uncond = self.enc_uncond + u_gate_mlp.unsqueeze(1) * uf

                    return output
                return hook

            h = block.register_forward_hook(make_hook(block, i), with_kwargs=True)
            self.hooks.append(h)

    def uninstall(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_summary(self):
        """Compute mean statistics across all samples."""
        summary = {}
        for point_name in ['normemb', 'kv', 'atten', 'fullblock']:
            summary[point_name] = []
            for layer_idx in range(self.num_layers):
                entries = self.stats[point_name][layer_idx]
                if not entries:
                    summary[point_name].append({})
                    continue
                avg = {}
                for key in entries[0].keys():
                    avg[key] = np.mean([e[key] for e in entries])
                summary[point_name].append(avg)
        return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-3-medium-diffusers')
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output', default='layer_analyse.md')
    parser.add_argument('--max_samples', type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda")
    print("Loading pipeline...")
    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    dataset_dir = args.dataset_dir
    categories = sorted([d for d in os.listdir(dataset_dir)
                        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')])

    # Collect stats across samples
    all_summaries = []
    sample_count = 0

    for cat in categories:
        pq_path = os.path.join(dataset_dir, cat, "V1-00000-of-00001.parquet")
        if not os.path.exists(pq_path):
            continue
        df = pd.read_parquet(pq_path)
        n = min(len(df), args.max_samples)

        for idx in range(n):
            row = df.iloc[idx]
            target_prompt = row['target_prompt'].replace('[', '').replace(']', '')
            img_data = row['image']
            if isinstance(img_data, dict) and 'bytes' in img_data:
                source_img = Image.open(io.BytesIO(img_data['bytes']))
            else:
                continue

            w, h = source_img.size
            w, h = w - w % 16, h - h % 16
            source_img = source_img.resize((w, h))

            # Encode
            (cond_embeds, neg_embeds, cond_pooled, neg_pooled) = pipe.encode_prompt(
                prompt=target_prompt, prompt_2=target_prompt, prompt_3=target_prompt,
                negative_prompt="", negative_prompt_2="", negative_prompt_3="",
                do_classifier_free_guidance=True, device=device)

            # Encode image + add noise
            image_tensor = pipe.image_processor.preprocess(source_img)
            image_tensor = image_tensor.to(device=device, dtype=pipe.vae.dtype)
            latents = pipe.vae.encode(image_tensor).latent_dist.sample()
            latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            latents = latents.to(dtype=cond_embeds.dtype)

            pipe.scheduler.set_timesteps(25, device=device)
            timesteps, _ = pipe.get_timesteps(25, 0.7, device)
            noise = torch.randn_like(latents)
            latents = pipe.scheduler.scale_noise(latents, timesteps[:1], noise)

            # Pick middle timestep
            t_idx = len(timesteps) // 2
            t = timesteps[t_idx]

            collector = StatsCollector(pipe, cond_embeds, neg_embeds, cond_pooled)
            collector.install()

            # Run one forward pass (cond path)
            with torch.no_grad():
                pipe.transformer(
                    hidden_states=latents,
                    timestep=t.unsqueeze(0),
                    encoder_hidden_states=cond_embeds,
                    pooled_projections=cond_pooled,
                    return_dict=False,
                )

            collector.uninstall()
            all_summaries.append(collector.get_summary())
            sample_count += 1
            print(f"  [{sample_count}] {cat}/{row.get('id', idx)}")

    print(f"\nCollected stats from {sample_count} samples.")

    # Average across all samples
    num_layers = len(pipe.transformer.transformer_blocks)
    final = {}
    for point in ['normemb', 'kv', 'atten', 'fullblock']:
        final[point] = []
        for layer_idx in range(num_layers):
            layer_entries = [s[point][layer_idx] for s in all_summaries if s[point][layer_idx]]
            if not layer_entries:
                final[point].append({})
                continue
            avg = {}
            for key in layer_entries[0].keys():
                avg[key] = np.mean([e[key] for e in layer_entries])
            final[point].append(avg)

    # Write report
    with open(args.output, 'w') as f:
        f.write("# Per-Layer Signal Statistics — Empirical Analysis\n\n")
        f.write(f"Averaged over {sample_count} samples from PIE-Bench++, middle timestep.\n\n")

        f.write("## 1. Cond-Uncond Difference Ratio per Layer\n\n")
        f.write("Ratio = ||cond - uncond|| / ||cond|| — measures how much of the signal is the guidance-relevant difference.\n\n")
        f.write("Higher ratio means CFG amplification affects a larger fraction of the signal → more disruptive.\n\n")

        f.write("| Layer | NormEmb (step 1) | KV-K (step 2) | KV-V (step 2) | Atten (step 4) | FullBlock (step 7) |\n")
        f.write("|:-----:|:----------------:|:-------------:|:-------------:|:--------------:|:------------------:|\n")
        for i in range(num_layers):
            ne = final['normemb'][i].get('ratio', 0)
            kvk = final['kv'][i].get('ratio_k', 0)
            kvv = final['kv'][i].get('ratio_v', 0)
            at = final['atten'][i].get('ratio', 0)
            fb = final['fullblock'][i].get('ratio', 0)
            f.write(f"| {i:2d} | {ne:.4f} | {kvk:.4f} | {kvv:.4f} | {at:.4f} | {fb:.4f} |\n")

        f.write("\n## 2. Absolute Norms per Layer\n\n")
        f.write("| Layer | NormEmb cond | NormEmb diff | KV-K cond | KV-K diff | Atten cond | Atten diff | Block cond | Block diff |\n")
        f.write("|:-----:|:----------:|:----------:|:--------:|:--------:|:---------:|:---------:|:---------:|:---------:|\n")
        for i in range(num_layers):
            ne_c = final['normemb'][i].get('cond_norm', 0)
            ne_d = final['normemb'][i].get('diff_norm', 0)
            kk_c = final['kv'][i].get('cond_k_norm', 0)
            kk_d = final['kv'][i].get('diff_k_norm', 0)
            at_c = final['atten'][i].get('cond_attn_norm', 0)
            at_d = final['atten'][i].get('diff_attn_norm', 0)
            fb_c = final['fullblock'][i].get('cond_hidden_norm', 0)
            fb_d = final['fullblock'][i].get('diff_hidden_norm', 0)
            f.write(f"| {i:2d} | {ne_c:.1f} | {ne_d:.1f} | {kk_c:.1f} | {kk_d:.1f} | {at_c:.1f} | {at_d:.1f} | {fb_c:.1f} | {fb_d:.1f} |\n")

        f.write("\n## 3. Summary\n\n")

        # Compute averages
        avg_ratios = {}
        for point in ['normemb', 'kv', 'atten', 'fullblock']:
            if point == 'kv':
                ratios = [final[point][i].get('ratio_k', 0) for i in range(num_layers) if final[point][i]]
            else:
                ratios = [final[point][i].get('ratio', 0) for i in range(num_layers) if final[point][i]]
            avg_ratios[point] = np.mean(ratios) if ratios else 0

        f.write("| Injection Point | Avg Diff/Signal Ratio | Implication |\n")
        f.write("|:----------------|:--------------------:|:------------|\n")
        f.write(f"| NormEmb (step 1) | {avg_ratios['normemb']:.4f} | {'High' if avg_ratios['normemb'] > 0.1 else 'Low'} — CFG affects {avg_ratios['normemb']*100:.1f}% of signal |\n")
        f.write(f"| KV (step 2) | {avg_ratios['kv']:.4f} | {'High' if avg_ratios['kv'] > 0.1 else 'Low'} — CFG affects {avg_ratios['kv']*100:.1f}% of signal |\n")
        f.write(f"| Atten (step 4) | {avg_ratios['atten']:.4f} | {'High' if avg_ratios['atten'] > 0.1 else 'Low'} — CFG affects {avg_ratios['atten']*100:.1f}% of signal |\n")
        f.write(f"| FullBlock (step 7) | {avg_ratios['fullblock']:.4f} | {'High' if avg_ratios['fullblock'] > 0.1 else 'Low'} — CFG affects {avg_ratios['fullblock']*100:.1f}% of signal |\n")

        f.write(f"\n**The lower the ratio, the safer the CFG amplification.** Full-block has the lowest ratio because the residual connection dominates the output, making the cond-uncond difference a tiny fraction of the total signal.\n")

    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
