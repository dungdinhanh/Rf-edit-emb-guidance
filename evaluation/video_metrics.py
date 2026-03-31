"""
Video-level evaluation metrics for RF-Solver-Edit.

VBench metrics from the paper "Taming Rectified Flow for Inversion and Editing" (arXiv:2411.04746):
  - Scene Consistency (SC): visual consistency across frames
  - Motion Smoothness (MS): temporal motion quality
  - Aesthetic Quality (AQ): overall visual appeal
  - Imaging Quality (IQ): technical image quality

Requires the `vbench` package: pip install vbench
VBench repo: https://github.com/Vchitect/VBench
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Per-frame image metrics (applied to video)
# ---------------------------------------------------------------------------

def _load_video_frames(
    video_path: Union[str, Path],
) -> List[torch.Tensor]:
    """Load video frames as list of (C, H, W) float tensors in [0, 1]."""
    import decord
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(str(video_path))
    frames = []
    for i in range(len(vr)):
        frame = vr[i]  # (H, W, C) uint8
        frame = frame.permute(2, 0, 1).float() / 255.0
        frames.append(frame)
    return frames


def compute_video_psnr(
    source_video: Union[str, Path],
    edited_video: Union[str, Path],
) -> Dict[str, float]:
    """Frame-averaged PSNR between source and edited video."""
    from .metrics import compute_psnr

    src_frames = _load_video_frames(source_video)
    edt_frames = _load_video_frames(edited_video)
    n = min(len(src_frames), len(edt_frames))

    psnrs = [compute_psnr(src_frames[i], edt_frames[i]) for i in range(n)]
    return {"psnr_mean": float(np.mean(psnrs)), "psnr_std": float(np.std(psnrs))}


def compute_video_ssim(
    source_video: Union[str, Path],
    edited_video: Union[str, Path],
) -> Dict[str, float]:
    """Frame-averaged SSIM between source and edited video."""
    from .metrics import compute_ssim

    src_frames = _load_video_frames(source_video)
    edt_frames = _load_video_frames(edited_video)
    n = min(len(src_frames), len(edt_frames))

    ssims = [compute_ssim(src_frames[i], edt_frames[i]) for i in range(n)]
    return {"ssim_mean": float(np.mean(ssims)), "ssim_std": float(np.std(ssims))}


def compute_video_lpips(
    source_video: Union[str, Path],
    edited_video: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Frame-averaged LPIPS between source and edited video."""
    from .metrics import compute_lpips

    src_frames = _load_video_frames(source_video)
    edt_frames = _load_video_frames(edited_video)
    n = min(len(src_frames), len(edt_frames))

    scores = [compute_lpips(src_frames[i], edt_frames[i], device=device) for i in range(n)]
    return {"lpips_mean": float(np.mean(scores)), "lpips_std": float(np.std(scores))}


# ---------------------------------------------------------------------------
# CLIP Score (text-video alignment)
# ---------------------------------------------------------------------------

def compute_video_clip_score(
    edited_video: Union[str, Path],
    target_prompt: str,
    device: Optional[torch.device] = None,
    sample_frames: int = 8,
) -> Dict[str, float]:
    """Frame-sampled CLIP score between edited video and target prompt.

    Samples `sample_frames` evenly spaced frames and averages CLIP scores.
    """
    from .metrics import compute_clip_score

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames = _load_video_frames(edited_video)
    n = len(frames)

    # Sample frames evenly
    if n <= sample_frames:
        indices = list(range(n))
    else:
        indices = [int(i * n / sample_frames) for i in range(sample_frames)]

    scores = []
    for idx in indices:
        frame = frames[idx]  # (C, H, W) float [0, 1]
        score = compute_clip_score(frame, target_prompt, device=device)
        scores.append(score)

    return {"clip_score_mean": float(np.mean(scores)), "clip_score_std": float(np.std(scores))}


# ---------------------------------------------------------------------------
# VBench metrics (wrapping the official VBench library)
# ---------------------------------------------------------------------------

# The 4 VBench dimensions used in the RF-Solver-Edit paper
VBENCH_DIMENSIONS = [
    "scene",               # Scene Consistency (SC)
    "motion_smoothness",   # Motion Smoothness (MS)
    "aesthetic_quality",   # Aesthetic Quality (AQ)
    "imaging_quality",     # Imaging Quality (IQ)
]


def compute_vbench_metrics(
    video_paths: Union[str, Path, List[Union[str, Path]]],
    dimensions: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Compute VBench metrics for video editing evaluation.

    Args:
        video_paths: Path(s) to generated/edited video(s). Can be a single
            video path or a list. If a directory, all .mp4 files in it are used.
        dimensions: VBench dimensions to evaluate. Defaults to the 4 used in
            the RF-Solver-Edit paper: scene, motion_smoothness,
            aesthetic_quality, imaging_quality.
        device: Torch device. Defaults to cuda if available.

    Returns:
        Dict mapping dimension name to score.

    Requires:
        pip install vbench
        See https://github.com/Vchitect/VBench for setup details.
    """
    try:
        from vbench import VBench
    except ImportError:
        raise ImportError(
            "VBench is not installed. Install it with:\n"
            "  pip install vbench\n"
            "See https://github.com/Vchitect/VBench for details."
        )

    if dimensions is None:
        dimensions = VBENCH_DIMENSIONS

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize video_paths to a list of strings
    if isinstance(video_paths, (str, Path)):
        video_paths = Path(video_paths)
        if video_paths.is_dir():
            video_paths = sorted(video_paths.glob("*.mp4"))
        else:
            video_paths = [video_paths]
    video_paths = [str(p) for p in video_paths]

    # VBench expects a result directory for caching
    output_dir = Path(video_paths[0]).parent / "vbench_results"
    output_dir.mkdir(exist_ok=True)

    results = {}
    for dim in dimensions:
        bench = VBench(device=str(device), full_json_dir=None)
        score = bench.evaluate(
            videos_path=video_paths,
            name=dim,
            dimension_list=[dim],
            output_path=str(output_dir),
        )
        # VBench returns list of (dimension, score) tuples
        if isinstance(score, list):
            for d, s in score:
                results[d] = s
        elif isinstance(score, dict):
            results.update(score)
        else:
            results[dim] = float(score)

    return results
