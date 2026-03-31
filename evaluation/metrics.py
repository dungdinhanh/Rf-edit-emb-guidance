"""
Image-level evaluation metrics for RF-Solver-Edit.

Metrics from the paper "Taming Rectified Flow for Inversion and Editing" (arXiv:2411.04746):
  - MSE: Mean Squared Error (pixel-level reconstruction accuracy)
  - PSNR: Peak Signal-to-Noise Ratio (reconstruction quality)
  - SSIM: Structural Similarity Index (structural preservation)
  - LPIPS: Learned Perceptual Image Patch Similarity (perceptual quality)
  - CLIP Score: Text-image alignment
  - FID: Fréchet Inception Distance (distribution-level quality)
"""

import math
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tensor(img: Union[torch.Tensor, np.ndarray, Image.Image]) -> torch.Tensor:
    """Convert input to float32 tensor in [0, 1] with shape (C, H, W)."""
    if isinstance(img, Image.Image):
        img = transforms.ToTensor()(img)
    elif isinstance(img, np.ndarray):
        if img.ndim == 2:
            img = img[None]
        elif img.ndim == 3 and img.shape[2] in (1, 3):
            img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        if img.max() > 1.0:
            img = img / 255.0
    elif isinstance(img, torch.Tensor):
        img = img.float()
        if img.max() > 1.0:
            img = img / 255.0
        if img.ndim == 2:
            img = img.unsqueeze(0)
        elif img.ndim == 3 and img.shape[2] in (1, 3):
            img = img.permute(2, 0, 1)
    return img


def _load_image(path: Union[str, Path]) -> torch.Tensor:
    return _to_tensor(Image.open(path).convert("RGB"))


# ---------------------------------------------------------------------------
# MSE
# ---------------------------------------------------------------------------

def compute_mse(
    source: Union[torch.Tensor, np.ndarray, Image.Image, str],
    target: Union[torch.Tensor, np.ndarray, Image.Image, str],
) -> float:
    """Mean Squared Error between two images. Lower is better."""
    if isinstance(source, (str, Path)):
        source = _load_image(source)
    if isinstance(target, (str, Path)):
        target = _load_image(target)
    source, target = _to_tensor(source), _to_tensor(target)
    return F.mse_loss(source, target).item()


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def compute_psnr(
    source: Union[torch.Tensor, np.ndarray, Image.Image, str],
    target: Union[torch.Tensor, np.ndarray, Image.Image, str],
    max_val: float = 1.0,
) -> float:
    """Peak Signal-to-Noise Ratio. Higher is better."""
    mse = compute_mse(source, target)
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / mse)


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _gaussian_kernel_2d(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    k1d = _gaussian_kernel_1d(size, sigma)
    return k1d.unsqueeze(1) @ k1d.unsqueeze(0)


def compute_ssim(
    source: Union[torch.Tensor, np.ndarray, Image.Image, str],
    target: Union[torch.Tensor, np.ndarray, Image.Image, str],
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    channel_mean: bool = True,
) -> float:
    """Structural Similarity Index. Higher is better.

    Follows Wang et al. 2004 with Gaussian-weighted windows.
    """
    if isinstance(source, (str, Path)):
        source = _load_image(source)
    if isinstance(target, (str, Path)):
        target = _load_image(target)
    source, target = _to_tensor(source), _to_tensor(target)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    kernel = _gaussian_kernel_2d(window_size, sigma)
    # Shape: (C, 1, H, W) for depthwise conv
    n_channels = source.shape[0]
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(n_channels, -1, -1, -1)

    source = source.unsqueeze(0)  # (1, C, H, W)
    target = target.unsqueeze(0)

    pad = window_size // 2

    mu_s = F.conv2d(source, kernel, padding=pad, groups=n_channels)
    mu_t = F.conv2d(target, kernel, padding=pad, groups=n_channels)

    mu_s_sq = mu_s ** 2
    mu_t_sq = mu_t ** 2
    mu_st = mu_s * mu_t

    sigma_s_sq = F.conv2d(source ** 2, kernel, padding=pad, groups=n_channels) - mu_s_sq
    sigma_t_sq = F.conv2d(target ** 2, kernel, padding=pad, groups=n_channels) - mu_t_sq
    sigma_st = F.conv2d(source * target, kernel, padding=pad, groups=n_channels) - mu_st

    numerator = (2 * mu_st + C1) * (2 * sigma_st + C2)
    denominator = (mu_s_sq + mu_t_sq + C1) * (sigma_s_sq + sigma_t_sq + C2)

    ssim_map = numerator / denominator

    if channel_mean:
        return ssim_map.mean().item()
    return ssim_map.mean(dim=[0, 2, 3]).tolist()


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------

_lpips_model = None


def _get_lpips_model(device: torch.device = torch.device("cpu")):
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net="vgg").eval()
    _lpips_model = _lpips_model.to(device)
    return _lpips_model


def compute_lpips(
    source: Union[torch.Tensor, np.ndarray, Image.Image, str],
    target: Union[torch.Tensor, np.ndarray, Image.Image, str],
    device: Optional[torch.device] = None,
) -> float:
    """Learned Perceptual Image Patch Similarity (VGG). Lower is better.

    Requires the `lpips` package: pip install lpips
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(source, (str, Path)):
        source = _load_image(source)
    if isinstance(target, (str, Path)):
        target = _load_image(target)
    source, target = _to_tensor(source).to(device), _to_tensor(target).to(device)
    # LPIPS expects [-1, 1]
    source = source * 2.0 - 1.0
    target = target * 2.0 - 1.0

    model = _get_lpips_model(device)
    with torch.no_grad():
        score = model(source.unsqueeze(0), target.unsqueeze(0))
    return score.item()


# ---------------------------------------------------------------------------
# CLIP Score (text-image alignment)
# ---------------------------------------------------------------------------

_clip_model = None
_clip_preprocess = None


def _get_clip_model(device: torch.device = torch.device("cpu")):
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        import clip
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
    _clip_model = _clip_model.to(device)
    return _clip_model, _clip_preprocess


def compute_clip_score(
    image: Union[torch.Tensor, np.ndarray, Image.Image, str],
    text: str,
    device: Optional[torch.device] = None,
) -> float:
    """CLIP cosine similarity between image and text. Higher is better.

    Requires the `clip` package: pip install git+https://github.com/openai/CLIP.git
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import clip

    model, preprocess = _get_clip_model(device)

    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, (torch.Tensor, np.ndarray)):
        image = _to_tensor(image)
        image = transforms.ToPILImage()(image)

    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = clip.tokenize([text], truncate=True).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).item()

    return similarity


# ---------------------------------------------------------------------------
# FID (Fréchet Inception Distance)
# ---------------------------------------------------------------------------

def _get_inception_features(
    images: List[Union[str, Path, Image.Image]],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract Inception-v3 pool3 features for a list of images."""
    from torchvision.models import inception_v3, Inception_V3_Weights

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval().to(device)

    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_features = []
    batch = []
    for img in images:
        if isinstance(img, (str, Path)):
            img = Image.open(img).convert("RGB")
        batch.append(preprocess(img))
        if len(batch) == batch_size:
            batch_tensor = torch.stack(batch).to(device)
            with torch.no_grad():
                feats = model(batch_tensor).cpu().numpy()
            all_features.append(feats)
            batch = []
    if batch:
        batch_tensor = torch.stack(batch).to(device)
        with torch.no_grad():
            feats = model(batch_tensor).cpu().numpy()
        all_features.append(feats)

    return np.concatenate(all_features, axis=0)


def compute_fid(
    real_images: List[Union[str, Path, Image.Image]],
    generated_images: List[Union[str, Path, Image.Image]],
    device: Optional[torch.device] = None,
    batch_size: int = 32,
) -> float:
    """Fréchet Inception Distance between two image sets. Lower is better.

    Uses Inception-v3 pool3 features and computes:
        FID = ||mu_r - mu_g||^2 + Tr(C_r + C_g - 2*sqrt(C_r @ C_g))
    """
    from scipy import linalg

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feats_real = _get_inception_features(real_images, device, batch_size)
    feats_gen = _get_inception_features(generated_images, device, batch_size)

    mu_r, sigma_r = feats_real.mean(axis=0), np.cov(feats_real, rowvar=False)
    mu_g, sigma_g = feats_gen.mean(axis=0), np.cov(feats_gen, rowvar=False)

    diff = mu_r - mu_g
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_g - 2.0 * covmean)
    return float(fid)


# ---------------------------------------------------------------------------
# Convenience: compute all image-pair metrics at once
# ---------------------------------------------------------------------------

def compute_all_image_metrics(
    source: Union[torch.Tensor, np.ndarray, Image.Image, str],
    edited: Union[torch.Tensor, np.ndarray, Image.Image, str],
    target_prompt: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Compute all image-level metrics between source and edited image.

    Returns a dict with keys: mse, psnr, ssim, lpips, and optionally clip_score.
    """
    results = {
        "mse": compute_mse(source, edited),
        "psnr": compute_psnr(source, edited),
        "ssim": compute_ssim(source, edited),
        "lpips": compute_lpips(source, edited, device=device),
    }
    if target_prompt is not None:
        results["clip_score"] = compute_clip_score(edited, target_prompt, device=device)
    return results
