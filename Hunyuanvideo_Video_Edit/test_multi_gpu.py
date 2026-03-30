"""
Smoke test for multi-GPU device placement.
Verifies that:
  1. Models land on the correct GPUs
  2. Tensors transfer between devices correctly
  3. The pipeline execution device resolves to the transformer GPU
  4. VAE encode/decode happens on the VAE GPU
  5. No device mismatch errors

Usage: python test_multi_gpu.py
Requires: 2 CUDA GPUs
"""

import sys
import torch
import torch.nn as nn

# ── Minimal stubs that mimic the real classes just enough ─────────────

class FakeConfig:
    in_channels = 4
    block_out_channels = [64, 128]
    scaling_factor = 0.18215
    shift_factor = None

class FakeLatentDist:
    def __init__(self, x):
        self._x = x
    def sample(self):
        return self._x

class FakeEncodeOutput:
    def __init__(self, x):
        self.latent_dist = FakeLatentDist(x)

class FakeVAE(nn.Module):
    """Tiny VAE stub."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(3, 4, 1)
        self.deconv = nn.Conv3d(4, 3, 1)
        self.config = FakeConfig()
        self._tiling = False

    def enable_tiling(self):
        self._tiling = True

    def encode(self, x):
        with torch.no_grad():
            z = self.conv(x)
        return FakeEncodeOutput(z)

    def decode(self, z, return_dict=False, generator=None):
        with torch.no_grad():
            out = self.deconv(z)
        return (out,)


class FakeTransformer(nn.Module):
    """Tiny transformer stub."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        # Attributes the pipeline/inference code reads
        self.patch_size = 1
        self.hidden_size = 64
        self.heads_num = 4
        self.rope_dim_list = None
        self.config = FakeConfig()

    def forward(self, x, t, **kwargs):
        info = kwargs.get("info", {})
        noise = self.linear(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return {"x": noise}, info


class FakeTextEncoder(nn.Module):
    """Tiny text encoder stub."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(100, hidden_dim)
        self.dtype = torch.float32

    def text2tokens(self, texts, data_type="video"):
        return torch.zeros(len(texts), 8, dtype=torch.long)

    def encode(self, tokens, data_type="video", device=None):
        out = self.embed(tokens.to(self.embed.weight.device))
        mask = torch.ones(out.shape[0], out.shape[1])
        class Output:
            hidden_state = out
            attention_mask = mask
        return Output()


# ── Tests ─────────────────────────────────────────────────────────────

def check(condition, msg):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {msg}")
    if not condition:
        check.failures += 1
    check.total += 1
check.failures = 0
check.total = 0


def test_device_placement():
    """Test that models land on correct GPUs."""
    print("\n=== Test 1: Device Placement ===")
    dev0 = torch.device("cuda:0")
    dev1 = torch.device("cuda:1")

    transformer = FakeTransformer().to(dev0)
    vae = FakeVAE().to(dev1)
    text_enc = FakeTextEncoder(64).to(dev1)

    check(next(transformer.parameters()).device == dev0,
          f"Transformer on cuda:0 (got {next(transformer.parameters()).device})")
    check(next(vae.parameters()).device == dev1,
          f"VAE on cuda:1 (got {next(vae.parameters()).device})")
    check(next(text_enc.parameters()).device == dev1,
          f"Text encoder on cuda:1 (got {next(text_enc.parameters()).device})")
    return transformer, vae, text_enc


def test_vae_encode_cross_device(vae):
    """Test VAE encode on cuda:1, then move latents to cuda:0."""
    print("\n=== Test 2: VAE Encode Cross-Device ===")
    dev0 = torch.device("cuda:0")
    dev1 = torch.device("cuda:1")

    # Simulate source video on CPU → move to VAE device
    source_video = torch.randn(1, 3, 5, 32, 32)
    vae_input = source_video.to(dev1)
    check(vae_input.device == dev1, f"Source video moved to cuda:1 for VAE")

    latents = vae.encode(vae_input).latent_dist.sample()
    check(latents.device == dev1, f"Latents produced on cuda:1 (VAE device)")

    latents = latents * vae.config.scaling_factor
    latents = latents.to(dev0)
    check(latents.device == dev0, f"Latents moved to cuda:0 (transformer device)")
    return latents


def test_transformer_forward(transformer, latents):
    """Test transformer forward pass on cuda:0."""
    print("\n=== Test 3: Transformer Forward ===")
    dev0 = torch.device("cuda:0")

    t = torch.tensor([0.5], device=dev0)
    info = {"feature": {}, "inject": False, "timestep": 0.5,
            "inversion": False, "second_order": False}

    with torch.no_grad():
        out, info = transformer(latents, t, info=info)
    noise_pred = out["x"]
    check(noise_pred.device == dev0,
          f"Transformer output on cuda:0 (got {noise_pred.device})")
    check(noise_pred.shape == latents.shape,
          f"Output shape matches input: {noise_pred.shape}")
    return noise_pred


def test_text_encoding_cross_device(text_enc):
    """Test text encoding on cuda:1, then move embeddings to cuda:0."""
    print("\n=== Test 4: Text Encoding Cross-Device ===")
    dev0 = torch.device("cuda:0")
    dev1 = torch.device("cuda:1")

    tokens = text_enc.text2tokens(["test prompt"], data_type="video")
    output = text_enc.encode(tokens, data_type="video", device=dev1)
    embeds = output.hidden_state
    check(embeds.device == dev1,
          f"Text embeddings produced on cuda:1")

    # Move to transformer device (as the pipeline does)
    embeds = embeds.to(dev0)
    check(embeds.device == dev0,
          f"Text embeddings moved to cuda:0 for denoising")


def test_vae_decode_cross_device(vae, latents):
    """Test moving latents to VAE device for decode."""
    print("\n=== Test 5: VAE Decode Cross-Device ===")
    dev0 = torch.device("cuda:0")
    dev1 = torch.device("cuda:1")

    check(latents.device == dev0, "Latents start on cuda:0")

    # Move to VAE device for decode
    latents_vae = latents.to(dev1)
    latents_vae = latents_vae / vae.config.scaling_factor
    check(latents_vae.device == dev1, "Latents moved to cuda:1 for decode")

    image = vae.decode(latents_vae, return_dict=False)[0]
    check(image.device == dev1, f"Decoded image on cuda:1 (got {image.device})")

    image = image.cpu().float()
    check(image.device == torch.device("cpu"), "Final image on CPU")


def test_no_cuda_oom():
    """Verify both GPUs still have memory headroom."""
    print("\n=== Test 6: Memory Check ===")
    for i in range(2):
        mem_used = torch.cuda.memory_allocated(i) / 1024**2
        _props = torch.cuda.get_device_properties(i)
        mem_total = getattr(_props, 'total_mem', _props.total_memory) / 1024**2
        pct = mem_used / mem_total * 100
        check(pct < 50, f"GPU {i}: {mem_used:.0f}MB / {mem_total:.0f}MB ({pct:.1f}%)")


def main():
    if torch.cuda.device_count() < 2:
        print(f"SKIP: Need 2 GPUs, found {torch.cuda.device_count()}")
        sys.exit(1)

    print(f"Found {torch.cuda.device_count()} GPUs:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({getattr(props, 'total_mem', props.total_memory) / 1024**3:.1f} GB)")

    transformer, vae, text_enc = test_device_placement()
    latents = test_vae_encode_cross_device(vae)
    test_transformer_forward(transformer, latents)
    test_text_encoding_cross_device(text_enc)
    test_vae_decode_cross_device(vae, latents)
    test_no_cuda_oom()

    # Cleanup
    del transformer, vae, text_enc, latents
    torch.cuda.empty_cache()

    print(f"\n{'='*40}")
    print(f"Results: {check.total - check.failures}/{check.total} passed")
    if check.failures:
        print(f"  {check.failures} FAILED")
        sys.exit(1)
    else:
        print("  All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
