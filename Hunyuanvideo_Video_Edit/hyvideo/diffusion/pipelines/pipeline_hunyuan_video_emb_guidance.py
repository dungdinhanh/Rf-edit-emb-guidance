# Embedding-level CFG pipeline for HunyuanVideo
# Instead of standard CFG on noise predictions:
#   noise = uncond + scale * (cond - uncond)
# We apply CFG on prompt embeddings:
#   guided_emb = (1 + alpha) * cond_emb - alpha * uncond_emb
# Then run a single forward pass with the guided embedding.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch
import numpy as np
from dataclasses import dataclass
from packaging import version

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput

from ...constants import PRECISION_TO_TYPE
from ...vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from ...text_encoder import TextEncoder
from ...modules import HYVideoDiffusionTransformer

# Reuse retrieve_timesteps from original pipeline
from .pipeline_hunyuan_video import retrieve_timesteps, HunyuanVideoPipelineOutput

logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """"""


class HunyuanVideoEmbGuidancePipeline(DiffusionPipeline):
    """
    HunyuanVideo pipeline with embedding-level classifier-free guidance.

    Instead of doing 2 forward passes (cond + uncond) and combining noise predictions,
    this pipeline combines the prompt embeddings directly:
        guided_emb = (1 + alpha) * cond_emb - alpha * uncond_emb
    and runs a single forward pass.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = ["text_encoder_2"]
    _exclude_from_cpu_offload = ["transformer"]
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: TextEncoder,
        transformer: HYVideoDiffusionTransformer,
        scheduler: KarrasDiffusionSchedulers,
        text_encoder_2: Optional[TextEncoder] = None,
        progress_bar_config: Dict[str, Any] = None,
        args=None,
    ):
        super().__init__()
        if progress_bar_config is None:
            progress_bar_config = {}
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(progress_bar_config)
        self.args = args

        if (
            hasattr(scheduler.config, "steps_offset")
            and scheduler.config.steps_offset != 1
        ):
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if (
            hasattr(scheduler.config, "clip_sample")
            and scheduler.config.clip_sample is True
        ):
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            text_encoder_2=text_encoder_2,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self.vae, "config") and hasattr(self.vae.config, "block_out_channels") else 8

    def encode_prompt_separate(
        self,
        prompt,
        negative_prompt,
        device,
        num_videos_per_prompt=1,
        text_encoder=None,
        data_type="video",
    ):
        """
        Encode prompt and negative prompt separately (not concatenated).
        Returns (cond_embeds, cond_mask, uncond_embeds, uncond_mask).
        """
        if text_encoder is None:
            text_encoder = self.text_encoder

        # Determine encoding device: use the text encoder's own device
        # (may be CPU in sharded multi-GPU mode), outputs are moved to target device below
        enc_device = next(text_encoder.model.parameters()).device if hasattr(text_encoder, 'model') else device

        # Encode conditional prompt
        text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)
        prompt_outputs = text_encoder.encode(text_inputs, data_type=data_type, device=enc_device)
        cond_embeds = prompt_outputs.hidden_state
        cond_mask = prompt_outputs.attention_mask
        if cond_mask is not None:
            cond_mask = cond_mask.to(device)

        # Encode unconditional prompt
        uncond_tokens = negative_prompt if negative_prompt is not None else [""]
        uncond_input = text_encoder.text2tokens(uncond_tokens, data_type=data_type)
        uncond_outputs = text_encoder.encode(uncond_input, data_type=data_type, device=enc_device)
        uncond_embeds = uncond_outputs.hidden_state
        uncond_mask = uncond_outputs.attention_mask
        if uncond_mask is not None:
            uncond_mask = uncond_mask.to(device)

        if text_encoder is not None:
            dtype = text_encoder.dtype
        else:
            dtype = cond_embeds.dtype

        cond_embeds = cond_embeds.to(dtype=dtype, device=device)
        uncond_embeds = uncond_embeds.to(dtype=dtype, device=device)

        return cond_embeds, cond_mask, uncond_embeds, uncond_mask

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_extra_func_kwargs(self, func, kwargs):
        extra_step_kwargs = {}
        for k, v in kwargs.items():
            if k in set(inspect.signature(func).parameters.keys()):
                extra_step_kwargs[k] = v
        return extra_step_kwargs

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        video_length: int,
        inject_step: int,
        emb_guidance_alpha: float = 1.0,
        data_type: str = "video",
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        vae_ver: str = "88-4c-sd",
        enable_tiling: bool = False,
        n_tokens: Optional[int] = None,
        embedded_guidance_scale: Optional[float] = None,
        inversion=False,
        info={},
        **kwargs,
    ):
        """
        Embedding-level CFG pipeline call.

        Args:
            emb_guidance_alpha: The alpha value for embedding guidance.
                guided_emb = (1 + alpha) * cond_emb - alpha * uncond_emb
                alpha=0 means no guidance (just cond_emb)
                alpha=1 means standard-strength guidance
        """
        # 1. Setup
        if isinstance(prompt, str):
            prompt = [prompt]
            batch_size = 1
        else:
            batch_size = len(prompt)

        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        elif negative_prompt is None:
            negative_prompt = [""]

        device = self._execution_device
        # In multi-GPU mode, use transformer's first device for inputs
        if hasattr(self, '_multi_gpu') and self._multi_gpu:
            if self.transformer.is_sharded:
                device = self.transformer._shard_device0
            else:
                device = next(self.transformer.parameters()).device

        # 2. Encode prompts separately
        cond_embeds, cond_mask, uncond_embeds, uncond_mask = self.encode_prompt_separate(
            prompt, negative_prompt, device, num_videos_per_prompt, data_type=data_type,
        )

        # Also encode with text_encoder_2 if available
        if self.text_encoder_2 is not None:
            cond_embeds_2, _, uncond_embeds_2, _ = self.encode_prompt_separate(
                prompt, negative_prompt, device, num_videos_per_prompt,
                text_encoder=self.text_encoder_2, data_type=data_type,
            )
        else:
            cond_embeds_2 = None
            uncond_embeds_2 = None

        # 3. Apply embedding-level guidance:
        #    guided_emb = (1 + alpha) * cond_emb - alpha * uncond_emb
        alpha = emb_guidance_alpha
        guided_embeds = (1.0 + alpha) * cond_embeds - alpha * uncond_embeds
        # For the mask, use the cond mask (it has the actual prompt tokens)
        guided_mask = cond_mask

        if cond_embeds_2 is not None and uncond_embeds_2 is not None:
            guided_embeds_2 = (1.0 + alpha) * cond_embeds_2 - alpha * uncond_embeds_2
        else:
            guided_embeds_2 = None

        logger.info(
            f"Embedding guidance: alpha={alpha}, "
            f"cond_emb norm={cond_embeds.norm().item():.4f}, "
            f"uncond_emb norm={uncond_embeds.norm().item():.4f}, "
            f"guided_emb norm={guided_embeds.norm().item():.4f}"
        )

        # 4. Prepare timesteps
        extra_set_timesteps_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.set_timesteps, {"n_tokens": n_tokens}
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, inversion, num_inference_steps, device,
            timesteps, sigmas, **extra_set_timesteps_kwargs,
        )

        if "884" in vae_ver:
            video_length = (video_length - 1) // 4 + 1
        elif "888" in vae_ver:
            video_length = (video_length - 1) // 8 + 1

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents, height, width, video_length,
            guided_embeds.dtype, device, generator, latents,
        )

        # 6. Extra step kwargs
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step, {"generator": generator, "eta": eta},
        )

        target_dtype = PRECISION_TO_TYPE[self.args.precision]
        autocast_enabled = target_dtype != torch.float32 and not self.args.disable_autocast
        vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
        vae_autocast_enabled = vae_dtype != torch.float32 and not self.args.disable_autocast

        # 7. Denoising loop — single forward pass per step (no CFG duplication)
        num_warmup_steps = len(timesteps[:-1]) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps[:-1])

        info['inversion'] = inversion
        inject_list = [True] * inject_step + [False] * (len(timesteps[:-1]) - inject_step)
        if inversion:
            inject_list = inject_list[::-1]

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps[:-1]):
                info['timestep'] = t.item() if not inversion else timesteps[i+1].item()
                info['inject'] = inject_list[i]

                # Single forward pass — no latent duplication
                latent_model_input = self.scheduler.scale_model_input(latents, t)
                t_expand = t.repeat(latent_model_input.shape[0])

                guidance_expand = (
                    torch.tensor(
                        [embedded_guidance_scale] * latent_model_input.shape[0],
                        dtype=torch.float32, device=device,
                    ).to(target_dtype) * 1000.0
                    if embedded_guidance_scale is not None
                    else None
                )

                with torch.autocast(
                    device_type="cuda", dtype=target_dtype, enabled=autocast_enabled
                ):
                    info['second_order'] = False
                    noise_pred, info = self.transformer(
                        latent_model_input,
                        t_expand,
                        info=info,
                        text_states=guided_embeds,
                        text_mask=guided_mask,
                        text_states_2=guided_embeds_2,
                        freqs_cos=freqs_cis[0],
                        freqs_sin=freqs_cis[1],
                        guidance=guidance_expand,
                        return_dict=True,
                    )
                    noise_pred = noise_pred["x"]
                    # Move output back to input device if sharded
                    if noise_pred.device != device:
                        noise_pred = noise_pred.to(device)

                # No CFG on output — guidance is already in the embeddings

                # RF solver: midpoint method
                # 1. obtain z_mid
                latents_mid = self.scheduler.step_mid(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # 2. obtain prediction at z_mid
                t_mid = (t + timesteps[i+1]) / 2
                latent_model_input_mid = self.scheduler.scale_model_input(latents_mid, t_mid)
                t_expand_mid = t_mid.repeat(latent_model_input.shape[0])

                with torch.autocast(
                    device_type="cuda", dtype=target_dtype, enabled=autocast_enabled
                ):
                    info['second_order'] = True
                    noise_pred_mid, info = self.transformer(
                        latent_model_input_mid,
                        t_expand_mid,
                        text_states=guided_embeds,
                        text_mask=guided_mask,
                        text_states_2=guided_embeds_2,
                        freqs_cos=freqs_cis[0],
                        freqs_sin=freqs_cis[1],
                        guidance=guidance_expand,
                        return_dict=True,
                        info=info,
                    )
                    noise_pred_mid = noise_pred_mid["x"]
                    # Move output back to input device if sharded
                    if noise_pred_mid.device != device:
                        noise_pred_mid = noise_pred_mid.to(device)

                # 3. step forward
                latents = self.scheduler.step_solver(
                    noise_pred_mid, noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                if i == len(timesteps[:-1]) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    if progress_bar is not None:
                        progress_bar.update()

        # 8. Decode
        if not output_type == "latent":
            _multi_gpu = hasattr(self, '_multi_gpu') and self._multi_gpu
            # Free transformer GPU memory before VAE decode
            self.transformer = self.transformer.cpu()
            torch.cuda.empty_cache()
            if _multi_gpu:
                latents = latents.to(self._vae_device)
            expand_temporal_dim = False
            if len(latents.shape) == 4:
                if isinstance(self.vae, AutoencoderKLCausal3D):
                    latents = latents.unsqueeze(2)
                    expand_temporal_dim = True
            elif len(latents.shape) == 5:
                pass
            else:
                raise ValueError(
                    f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
                )

            if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
                latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
            else:
                latents = latents / self.vae.config.scaling_factor

            with torch.autocast(
                device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled
            ):
                if enable_tiling:
                    self.vae.enable_tiling()
                image = self.vae.decode(latents, return_dict=False, generator=generator)[0]

            if expand_temporal_dim or image.shape[2] == 1:
                image = image.squeeze(2)
            image = (image / 2 + 0.5).clamp(0, 1)
        else:
            image = latents

        image = image.cpu().float()
        self.maybe_free_model_hooks()

        if not return_dict:
            return image

        return HunyuanVideoPipelineOutput(videos=image), info
