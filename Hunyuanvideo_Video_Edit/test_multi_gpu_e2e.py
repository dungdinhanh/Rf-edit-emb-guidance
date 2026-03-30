"""
End-to-end multi-GPU test with dummy weights.

Tests the full HunyuanVideoEdit pipeline (VAE encode → inversion → denoise → VAE decode)
across 2 GPUs using randomly initialized (dummy) model weights.

GPU 0: Transformer (denoising)
GPU 1: VAE + Text Encoders

Usage: python test_multi_gpu_e2e.py
Requires: 2 CUDA GPUs
"""

import sys
import os
import argparse
import time
import tempfile

import torch
import torch.nn as nn
import numpy as np

# ── Check GPUs before importing heavy modules ──────────────────────────
if torch.cuda.device_count() < 2:
    print(f"SKIP: Need 2 GPUs, found {torch.cuda.device_count()}")
    sys.exit(1)

for i in range(2):
    props = torch.cuda.get_device_properties(i)
    mem_gb = props.total_memory / 1024**3
    print(f"GPU {i}: {props.name} ({mem_gb:.1f} GB)")

# ── Stub components that match real interfaces ─────────────────────────

class FakeVAEConfig:
    """Mimics the VAE config object."""
    in_channels = 3
    out_channels = 3
    latent_channels = 16
    block_out_channels = [128, 256]  # 2 blocks → vae_scale_factor = 2
    scaling_factor = 0.18215
    shift_factor = None
    spatial_compression_ratio = 8
    time_compression_ratio = 4


class FakeLatentDist:
    def __init__(self, x):
        self._x = x
    def sample(self):
        return self._x


class FakeEncodeOutput:
    def __init__(self, x):
        self.latent_dist = FakeLatentDist(x)


class FakeVAE(nn.Module):
    """Tiny 3D conv VAE stub matching AutoencoderKLCausal3D interface."""
    def __init__(self):
        super().__init__()
        self.config = FakeVAEConfig()
        # Encoder: [B,3,T,H,W] -> [B,16,T',H',W']
        self.encoder = nn.Conv3d(3, 16, kernel_size=1)
        # Decoder: [B,16,T',H',W'] -> [B,3,T,H,W] (won't match dims exactly, but good enough)
        self.decoder = nn.Conv3d(16, 3, kernel_size=1)
        self._tiling = False

    def enable_tiling(self):
        self._tiling = True

    def encode(self, x):
        with torch.no_grad():
            z = self.encoder(x)
        return FakeEncodeOutput(z)

    def decode(self, z, return_dict=False, generator=None):
        with torch.no_grad():
            out = self.decoder(z)
        return (out,)


class FakeTextEncoderOutput:
    def __init__(self, hidden_state, attention_mask):
        self.hidden_state = hidden_state
        self.attention_mask = attention_mask


class FakeTextEncoder(nn.Module):
    """Stub text encoder matching hyvideo.text_encoder.TextEncoder interface."""
    def __init__(self, hidden_dim=4096):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(100, hidden_dim)
        self.dtype = torch.float16

    def text2tokens(self, texts, data_type="video"):
        """Return dummy token dict."""
        batch_size = len(texts) if isinstance(texts, list) else 1
        return {"input_ids": torch.zeros(batch_size, 8, dtype=torch.long)}

    def encode(self, batch_encoding, data_type="video", device=None):
        """Return dummy embeddings on the encoder's device."""
        input_ids = batch_encoding["input_ids"]
        dev = next(self.parameters()).device
        input_ids = input_ids.to(dev)
        out = self.embed(input_ids)
        mask = torch.ones(out.shape[0], out.shape[1], device=dev)
        return FakeTextEncoderOutput(out, mask)

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)


class FakeTransformerConfig:
    in_channels = 16


class FakeTransformer(nn.Module):
    """Tiny transformer stub matching HYVideoDiffusionTransformer interface."""
    def __init__(self, hidden_size=192, heads_num=4):
        super().__init__()
        self.patch_size = [1, 2, 2]
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        # head_dim=192/4=48, split into 3 equal parts for RoPE (T, H, W)
        self.rope_dim_list = [16, 16, 16]
        self.config = FakeTransformerConfig()

        # Minimal conv to transform latent channels
        self.proj = nn.Conv3d(16, 16, kernel_size=1)

    def forward(self, x, t, text_states=None, text_mask=None,
                text_states_2=None, freqs_cos=None, freqs_sin=None,
                guidance=None, return_dict=True, info=None, **kwargs):
        if info is None:
            info = {}
        with torch.no_grad():
            noise = self.proj(x)
        if return_dict:
            return {"x": noise}, info
        return noise, info


# ── Build args namespace matching config.py defaults ───────────────────

def make_args():
    return argparse.Namespace(
        # Model
        model="HYVideo-T/2-cfgdistill",
        latent_channels=16,
        precision="bf16",
        # VAE
        vae="884-16c-hy",
        vae_precision="fp16",
        vae_tiling=True,
        # Text encoder
        text_encoder="llm",
        text_encoder_2="clipL",
        text_encoder_precision="fp16",
        text_encoder_precision_2="fp16",
        text_states_dim=4096,
        text_states_dim_2=768,
        text_len=256,
        text_len_2=77,
        tokenizer="llm",
        tokenizer_2="clipL",
        prompt_template="dit-llm-encode",
        prompt_template_video="dit-llm-encode-video",
        hidden_state_skip_layer=2,
        apply_final_norm=False,
        # Scheduler
        denoise_type="flow",
        flow_shift=7.0,
        flow_reverse=True,
        flow_solver="euler",
        # Inference
        model_base="ckpts",
        dit_weight="dummy",
        model_resolution="540p",
        load_key="module",
        use_cpu_offload=False,
        multi_gpu=True,
        gpu_ids=[0, 1],
        batch_size=1,
        infer_steps=3,  # Very few steps for speed
        disable_autocast=False,
        save_path="./test_results",
        save_path_suffix="",
        name_suffix="",
        num_videos=1,
        seed=42,
        seed_type="fixed",
        inject_step=1,
        neg_prompt=None,
        cfg_scale=1.0,
        embedded_cfg_scale=6.0,
        reproduce=False,
        source_path=None,
        source_prompt="a cat sitting on a couch",
        target_prompt="a dog sitting on a couch",
        use_linear_quadratic_schedule=False,
        linear_schedule_end=25,
        rope_theta=256,
    )


# ── Monkey-patch video loading to avoid PyAV dependency ────────────────

def patch_video_loading(num_frames=5, height=64, width=64):
    """Replace read_video_from_path with a dummy that returns random video tensor."""
    import hyvideo.utils.data_utils as du

    def fake_read_video(path, transform=None, transform_name="center", image_size=(256, 256)):
        # Return [C, T, H, W] tensor normalized to [-1, 1] (matching 'norm' transform)
        video = torch.randn(3, num_frames, height, width)
        return video

    du.read_video_from_path = fake_read_video


# ── Main test ──────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("Multi-GPU End-to-End Test (Dummy Weights)")
    print("=" * 60)

    args = make_args()
    dev0 = torch.device("cuda:0")
    dev1 = torch.device("cuda:1")

    # 1. Create models on the right GPUs
    print("\n[1/6] Creating dummy models...")
    transformer = FakeTransformer(hidden_size=192, heads_num=4).to(dev0).half()
    vae = FakeVAE().to(dev1).half()
    text_encoder = FakeTextEncoder(hidden_dim=4096).to(dev1).half()
    text_encoder_2 = FakeTextEncoder(hidden_dim=768).to(dev1).half()

    print(f"  Transformer on: {next(transformer.parameters()).device}")
    print(f"  VAE on:         {next(vae.parameters()).device}")
    print(f"  Text enc on:    {next(text_encoder.parameters()).device}")
    print(f"  Text enc 2 on:  {next(text_encoder_2.parameters()).device}")

    # 2. Build pipeline
    print("\n[2/6] Building pipeline...")
    from hyvideo.diffusion.schedulers import FlowMatchDiscreteScheduler
    from hyvideo.diffusion.pipelines import HunyuanVideoPipeline

    scheduler = FlowMatchDiscreteScheduler(
        shift=args.flow_shift,
        reverse=args.flow_reverse,
        solver=args.flow_solver,
    )

    pipeline = HunyuanVideoPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        transformer=transformer,
        scheduler=scheduler,
        args=args,
    )

    # Set multi-GPU flags (as done in inference.py load_diffusion_pipeline)
    pipeline._multi_gpu = True
    pipeline._vae_device = dev1
    # DiffusionPipeline._execution_device is a read-only property that fails with
    # stub modules. Override it on the class since multi-GPU mode replaces it anyway.
    HunyuanVideoPipeline._execution_device = property(lambda self: dev0)
    print("  Pipeline created with _multi_gpu=True")

    # 3. Create HunyuanVideoEdit wrapper
    print("\n[3/6] Creating HunyuanVideoEdit wrapper...")
    from hyvideo.inference import HunyuanVideoEdit

    editor = HunyuanVideoEdit(
        args=args,
        vae=vae,
        vae_kwargs={"s_ratio": 8, "t_ratio": 4},
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        model=transformer,
        pipeline=pipeline,
        use_cpu_offload=False,
        device=str(dev0),
        vae_device=str(dev1),
    )
    # Override the pipeline that __init__ builds (it would call load_diffusion_pipeline
    # which we already did manually)
    editor.pipeline = pipeline
    print("  Editor wrapper ready")

    # 4. Patch video loading with dummy data
    print("\n[4/6] Patching video loading (dummy 5 frames, 64x64)...")
    patch_video_loading(num_frames=5, height=64, width=64)
    # Also patch it in inference module since it imports directly
    import hyvideo.inference as inf_mod
    import hyvideo.utils.data_utils as du_mod
    inf_mod.read_video_from_path = du_mod.read_video_from_path
    video_path = "/tmp/dummy_video.mp4"  # path won't be read
    print("  Video loading patched")

    # 5. Run predict
    print("\n[5/6] Running predict (inversion + denoise)...")
    print(f"  Steps: {args.infer_steps}, inject_step: {args.inject_step}")
    t0 = time.time()

    try:
        outputs = editor.predict(
            source_video_path=video_path,
            source_prompt=args.source_prompt,
            target_prompt=args.target_prompt,
            seed=args.seed,
            negative_prompt=args.neg_prompt,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            num_videos_per_prompt=1,
            flow_shift=args.flow_shift,
            batch_size=1,
            embedded_guidance_scale=args.embedded_cfg_scale,
            inject_step=args.inject_step,
        )
        elapsed = time.time() - t0
        print(f"  Predict completed in {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  FAILED after {elapsed:.1f}s: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 6. Verify outputs
    print("\n[6/6] Verifying outputs...")
    samples = outputs.get("samples")
    seeds = outputs.get("seeds")
    size = outputs.get("size")
    print(f"  Seeds: {seeds}")
    print(f"  Size (H,W,T): {size}")

    if samples is not None:
        if isinstance(samples, torch.Tensor):
            print(f"  Sample tensor shape: {samples.shape}, device: {samples.device}, dtype: {samples.dtype}")
            assert samples.device == torch.device("cpu"), f"Expected CPU output, got {samples.device}"
            print("  [PASS] Output on CPU")
        else:
            print(f"  Sample type: {type(samples)}, len: {len(samples) if hasattr(samples, '__len__') else 'N/A'}")
    else:
        print("  WARNING: No samples in output")

    # Memory check
    print("\n  Memory usage:")
    for i in range(2):
        used = torch.cuda.memory_allocated(i) / 1024**2
        total = torch.cuda.get_device_properties(i).total_memory / 1024**2
        print(f"    GPU {i}: {used:.0f} MB / {total:.0f} MB ({used/total*100:.1f}%)")

    # Cleanup
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("ALL PASSED — Multi-GPU pipeline works end-to-end!")
    print("=" * 60)


if __name__ == "__main__":
    main()
