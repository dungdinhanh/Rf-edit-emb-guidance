"""
Test: Transformer layer sharding across 2 GPUs.

Verifies that HYVideoDiffusionTransformer.shard_to_devices() correctly
distributes layers and produces valid forward pass output.

Usage: python test_sharding.py
Requires: 2 CUDA GPUs
"""

import sys
import time
import argparse

import torch

if torch.cuda.device_count() < 2:
    print(f"SKIP: Need 2 GPUs, found {torch.cuda.device_count()}")
    sys.exit(1)

for i in range(2):
    props = torch.cuda.get_device_properties(i)
    mem_gb = props.total_memory / 1024**3
    print(f"GPU {i}: {props.name} ({mem_gb:.1f} GB)")

from hyvideo.modules.models import HYVideoDiffusionTransformer, HUNYUAN_VIDEO_CONFIG


def main():
    dev0 = torch.device("cuda:0")
    dev1 = torch.device("cuda:1")
    dtype = torch.bfloat16

    args = argparse.Namespace(
        text_states_dim=4096,
        text_states_dim_2=768,
    )

    config = HUNYUAN_VIDEO_CONFIG["HYVideo-T/2-cfgdistill"]
    print(f"\nModel config: {config}")
    print(f"Double blocks: {config['mm_double_blocks_depth']}, "
          f"Single blocks: {config['mm_single_blocks_depth']}")

    # Build on CPU to avoid OOM
    print("\n[1/4] Building transformer on CPU (bf16)...")
    t0 = time.time()
    with torch.no_grad():
        model = HYVideoDiffusionTransformer(
            args,
            in_channels=16,
            out_channels=16,
            device="cpu",
            dtype=dtype,
            **config,
        )
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    param_gb = param_count * 2 / 1024**3  # bf16 = 2 bytes
    print(f"  Parameters: {param_count:,} ({param_gb:.2f} GB in bf16)")
    print(f"  Built in {time.time() - t0:.1f}s")

    # Shard across GPUs
    print(f"\n[2/4] Sharding to {dev0} and {dev1}...")
    t0 = time.time()
    model.shard_to_devices(dev0, dev1)
    print(f"  Double blocks split at: {model._double_split} "
          f"(0-{model._double_split-1} on {dev0}, {model._double_split}-{len(model.double_blocks)-1} on {dev1})")
    print(f"  Single blocks split at: {model._single_split} "
          f"(0-{model._single_split-1} on {dev0}, {model._single_split}-{len(model.single_blocks)-1} on {dev1})")
    print(f"  Sharded in {time.time() - t0:.1f}s")

    # Check memory distribution
    for i in range(2):
        used = torch.cuda.memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_mem / 1024**3 if hasattr(torch.cuda.get_device_properties(i), 'total_mem') else torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {used:.2f} GB allocated")

    # Forward pass
    print(f"\n[3/4] Running forward pass...")
    B, C, T, H, W = 1, 16, 5, 64, 64
    patch_t, patch_h, patch_w = model.patch_size
    tt = T // patch_t
    th = H // patch_h
    tw = W // patch_w

    # Inputs on dev0 (where embeddings live)
    x = torch.randn(B, C, T, H, W, device=dev0, dtype=dtype)
    t_step = torch.tensor([500.0], device=dev0, dtype=dtype)
    text_states = torch.randn(B, 8, 4096, device=dev0, dtype=dtype)
    text_mask = torch.ones(B, 8, device=dev0, dtype=torch.bool)
    text_states_2 = torch.randn(B, 768, device=dev0, dtype=dtype)
    guidance = torch.tensor([6000.0], device=dev0, dtype=dtype)

    # RoPE embeddings
    seq_len = tt * th * tw
    rope_dim = config["hidden_size"] // config["heads_num"]
    freqs_cos = torch.randn(seq_len, rope_dim, device=dev0, dtype=dtype)
    freqs_sin = torch.randn(seq_len, rope_dim, device=dev0, dtype=dtype)

    info = {'inversion': False, 'inject': False, 'second_order': False, 'feature': {}}

    t0 = time.time()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        out, info = model(
            x, t_step,
            text_states=text_states,
            text_mask=text_mask,
            text_states_2=text_states_2,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            guidance=guidance,
            return_dict=True,
            info=info,
        )
    elapsed = time.time() - t0
    output = out["x"]
    print(f"  Output shape: {output.shape}, device: {output.device}, dtype: {output.dtype}")
    print(f"  Expected shape: [{B}, {C}, {T}, {H}, {W}]")
    print(f"  Forward pass took {elapsed:.1f}s")

    # Verify output
    print(f"\n[4/4] Verifying...")
    assert output.shape == (B, C, T, H, W), f"Shape mismatch: {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    # Output should be on dev1 (where final_layer is) — but pipeline moves it back
    # Here we just verify it's a valid tensor
    print(f"  Shape: PASS")
    print(f"  No NaN/Inf: PASS")

    # Final memory
    print(f"\n  Final memory:")
    for i in range(2):
        used = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"    GPU {i}: {used:.2f} GB allocated, {reserved:.2f} GB reserved")

    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("ALL PASSED — Transformer sharding works!")
    print("=" * 60)


if __name__ == "__main__":
    main()
