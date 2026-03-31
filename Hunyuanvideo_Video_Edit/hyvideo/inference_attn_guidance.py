"""
Inference class for embedding-level CFG experiment.
Extends HunyuanVideoEdit to use the EmbGuidance pipeline.
"""

import time
import random
from pathlib import Path
from loguru import logger

import torch
from hyvideo.constants import PROMPT_TEMPLATE, NEGATIVE_PROMPT, PRECISION_TO_TYPE
from hyvideo.vae import load_vae
from hyvideo.modules import load_model
from hyvideo.text_encoder import TextEncoder
from hyvideo.utils.data_utils import align_to, read_video_from_path
from hyvideo.modules.posemb_layers import get_nd_rotary_pos_embed
from hyvideo.diffusion.schedulers import FlowMatchDiscreteScheduler
from hyvideo.diffusion.pipelines.pipeline_hunyuan_video_attn_guidance import HunyuanVideoAttnGuidancePipeline
from hyvideo.inference import Inference
import torch.nn.functional as F


class HunyuanVideoEditAttnGuidance(Inference):
    """
    HunyuanVideoEdit variant that uses embedding-level CFG.
    """

    def __init__(self, args, vae, vae_kwargs, text_encoder, model,
                 text_encoder_2=None, pipeline=None, use_cpu_offload=False,
                 device=0, vae_device=None, logger=None):
        super().__init__(
            args, vae, vae_kwargs, text_encoder, model,
            text_encoder_2=text_encoder_2, pipeline=pipeline,
            use_cpu_offload=use_cpu_offload, device=device,
            vae_device=vae_device, logger=logger,
        )
        self.pipeline = self.load_diffusion_pipeline(
            args=args, vae=self.vae, text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2, model=self.model, device=self.device,
        )
        self.default_negative_prompt = NEGATIVE_PROMPT

    @classmethod
    def from_pretrained(cls, pretrained_model_path, args, device=None, **kwargs):
        """
        Override parent to build model on CPU when using cpu_offload.
        This avoids OOM on 24GB GPUs (the model alone is ~24GB in bf16).
        The pipeline's enable_sequential_cpu_offload() handles moving modules
        to GPU on-demand during inference.
        """
        logger.info(f"Got text-to-video model root path: {pretrained_model_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_grad_enabled(False)

        # Determine device placement
        multi_gpu = getattr(args, 'multi_gpu', False)
        if multi_gpu:
            gpu_ids = args.gpu_ids
            device = f"cuda:{gpu_ids[0]}"
            vae_device = f"cuda:{gpu_ids[1]}"
            logger.info(f"Multi-GPU: sharding transformer across {device} and {vae_device}, "
                        f"VAE+text encoders on {vae_device}")
        elif args.use_cpu_offload:
            vae_device = "cpu"
        else:
            vae_device = device

        # Build and load weights on CPU first, then shard/move to target device(s).
        logger.info(f"Building model on cpu...")
        factor_kwargs = {"device": "cpu", "dtype": PRECISION_TO_TYPE[args.precision]}
        model = load_model(
            args, in_channels=args.latent_channels,
            out_channels=args.latent_channels, factor_kwargs=factor_kwargs,
        )
        model = Inference.load_state_dict(args, model, pretrained_model_path)

        if multi_gpu:
            model.shard_to_devices(device, vae_device)
            logger.info(f"Transformer sharded: double blocks split at {model._double_split}, "
                        f"single blocks split at {model._single_split}")
        else:
            model = model.to(device)
        model.eval()

        # VAE
        vae, _, s_ratio, t_ratio = load_vae(
            args.vae, args.vae_precision, logger=logger,
            device=vae_device,
        )
        vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}

        # Text encoder setup
        if args.prompt_template_video is not None:
            crop_start = PROMPT_TEMPLATE[args.prompt_template_video].get("crop_start", 0)
        elif args.prompt_template is not None:
            crop_start = PROMPT_TEMPLATE[args.prompt_template].get("crop_start", 0)
        else:
            crop_start = 0
        max_length = args.text_len + crop_start

        prompt_template = PROMPT_TEMPLATE[args.prompt_template] if args.prompt_template else None
        prompt_template_video = PROMPT_TEMPLATE[args.prompt_template_video] if args.prompt_template_video else None

        # In multi-GPU sharded mode, load text encoders on CPU to save GPU memory.
        # They're only used once for prompt encoding at the start.
        # CPU doesn't support fp16 matmul, so force fp32 when on CPU.
        text_enc_device = "cpu" if multi_gpu else vae_device
        text_enc_precision = "fp32" if multi_gpu else args.text_encoder_precision
        text_enc_precision_2 = "fp32" if multi_gpu else args.text_encoder_precision_2

        text_encoder = TextEncoder(
            text_encoder_type=args.text_encoder, max_length=max_length,
            text_encoder_precision=text_enc_precision,
            tokenizer_type=args.tokenizer,
            prompt_template=prompt_template, prompt_template_video=prompt_template_video,
            hidden_state_skip_layer=args.hidden_state_skip_layer,
            apply_final_norm=args.apply_final_norm, reproduce=args.reproduce,
            logger=logger, device=text_enc_device,
        )
        text_encoder_2 = None
        if args.text_encoder_2 is not None:
            text_encoder_2 = TextEncoder(
                text_encoder_type=args.text_encoder_2, max_length=args.text_len_2,
                text_encoder_precision=text_enc_precision_2,
                tokenizer_type=args.tokenizer_2, reproduce=args.reproduce,
                logger=logger, device=text_enc_device,
            )

        return cls(
            args=args, vae=vae, vae_kwargs=vae_kwargs,
            text_encoder=text_encoder, text_encoder_2=text_encoder_2,
            model=model, use_cpu_offload=args.use_cpu_offload,
            device=device, vae_device=vae_device, logger=logger,
        )

    def load_diffusion_pipeline(self, args, vae, text_encoder, text_encoder_2,
                                 model, scheduler=None, device=None,
                                 progress_bar_config=None, data_type="video"):
        if scheduler is None:
            if args.denoise_type == "flow":
                scheduler = FlowMatchDiscreteScheduler(
                    shift=args.flow_shift, reverse=args.flow_reverse,
                    solver=args.flow_solver,
                )
            else:
                raise ValueError(f"Invalid denoise type {args.denoise_type}")

        pipeline = HunyuanVideoAttnGuidancePipeline(
            vae=vae, text_encoder=text_encoder, text_encoder_2=text_encoder_2,
            transformer=model, scheduler=scheduler,
            progress_bar_config=progress_bar_config, args=args,
        )
        if getattr(self.args, 'multi_gpu', False):
            pipeline._multi_gpu = True
            pipeline._vae_device = torch.device(self.vae_device)
        elif self.use_cpu_offload:
            pipeline.enable_sequential_cpu_offload()
        else:
            pipeline = pipeline.to(device)
        return pipeline

    def adjust_video_frames(self, source_video):
        dim_size = source_video.size(2)
        if (dim_size - 1) % 4 != 0:
            new_size = ((dim_size - 1) // 4) * 4 + 1
            source_video = source_video[:, :, :new_size, :, :]
        return source_video

    def get_rotary_pos_embed(self, video_length, height, width):
        target_ndim = 3
        ndim = 5 - 2
        if "884" in self.args.vae:
            latents_size = [(video_length - 1) // 4 + 1, height // 8, width // 8]
        elif "888" in self.args.vae:
            latents_size = [(video_length - 1) // 8 + 1, height // 8, width // 8]
        else:
            latents_size = [video_length, height // 8, width // 8]

        if isinstance(self.model.patch_size, int):
            rope_sizes = [s // self.model.patch_size for s in latents_size]
        elif isinstance(self.model.patch_size, list):
            rope_sizes = [s // self.model.patch_size[idx] for idx, s in enumerate(latents_size)]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes
        head_dim = self.model.hidden_size // self.model.heads_num
        rope_dim_list = self.model.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]

        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list, rope_sizes, theta=self.args.rope_theta,
            use_real=True, theta_rescale_factor=1,
        )
        return freqs_cos, freqs_sin

    @torch.no_grad()
    def predict(
        self,
        source_video_path,
        source_prompt,
        target_prompt,
        seed=None,
        negative_prompt=None,
        infer_steps=50,
        guidance_scale=6,
        flow_shift=5.0,
        embedded_guidance_scale=None,
        batch_size=1,
        num_videos_per_prompt=1,
        inject_step=0,
        attn_guidance_alpha=1.0,
        attn_guidance_k=32,
        overlap_target="uncond",
        **kwargs,
    ):
        # Load video
        source_video = read_video_from_path(source_video_path, transform_name='norm').unsqueeze(0)
        source_video = self.adjust_video_frames(source_video).to(torch.float16)
        height, width, video_length = source_video.shape[3], source_video.shape[4], source_video.shape[2]
        out_dict = dict()

        # Seeds
        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [random.randint(0, 1_000_000) for _ in range(batch_size * num_videos_per_prompt)]
        elif isinstance(seed, int):
            seeds = [seed + i for _ in range(batch_size) for i in range(num_videos_per_prompt)]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [int(seed[i]) + j for i in range(batch_size) for j in range(num_videos_per_prompt)]
            else:
                seeds = [int(s) for s in seed]
        else:
            raise ValueError(f"Invalid seed: {seed}")
        generator = [torch.Generator(self.device).manual_seed(s) for s in seeds]
        out_dict["seeds"] = seeds

        # Video dimensions
        if width <= 0 or height <= 0 or video_length <= 0:
            raise ValueError(f"Invalid dimensions: h={height}, w={width}, f={video_length}")
        if (video_length - 1) % 4 != 0:
            raise ValueError(f"`video_length-1` must be a multiple of 4, got {video_length}")

        target_height = align_to(height, 16)
        target_width = align_to(width, 16)
        target_video_length = video_length
        out_dict["size"] = (target_height, target_width, target_video_length)

        # Prompts
        if not isinstance(source_prompt, str):
            raise TypeError(f"`source_prompt` must be a string")
        source_prompt = [source_prompt.strip()]
        if not isinstance(target_prompt, str):
            raise TypeError(f"`target_prompt` must be a string")
        target_prompt = [target_prompt.strip()]

        if negative_prompt is None or negative_prompt == "":
            negative_prompt = self.default_negative_prompt
        if not isinstance(negative_prompt, str):
            raise TypeError(f"`negative_prompt` must be a string")
        negative_prompt = [negative_prompt.strip()]

        # Scheduler
        scheduler = FlowMatchDiscreteScheduler(
            shift=flow_shift, reverse=self.args.flow_reverse, solver=self.args.flow_solver,
        )
        self.pipeline.scheduler = scheduler

        # Rotary embeddings
        freqs_cos, freqs_sin = self.get_rotary_pos_embed(target_video_length, target_height, target_width)
        n_tokens = freqs_cos.shape[0]

        logger.info(f"""
        === Embedding-level CFG Experiment ===
        height: {target_height}, width: {target_width}, video_length: {target_video_length}
        source_prompt: {source_prompt}
        target_prompt: {target_prompt}
        attn_guidance_alpha: {attn_guidance_alpha}
        embedded_guidance_scale: {embedded_guidance_scale}
        inject_step: {inject_step}
        infer_steps: {infer_steps}
        seed: {seed}
        """)

        start_time = time.time()
        multi_gpu = getattr(self.args, 'multi_gpu', False)

        # VAE encode source video
        if self.args.vae_tiling:
            self.pipeline.vae.enable_tiling()
        vae_input = source_video.to(self.vae_device) if multi_gpu else source_video
        latents = self.pipeline.vae.encode(vae_input).latent_dist.sample()
        if hasattr(self.pipeline.vae.config, "shift_factor") and self.pipeline.vae.config.shift_factor:
            latents = (latents - self.pipeline.vae.config.shift_factor) * self.pipeline.vae.config.scaling_factor
        else:
            latents = latents * self.pipeline.vae.config.scaling_factor
        # Move latents to transformer device for denoising
        if multi_gpu:
            latents = latents.to(self.device)

        # Step 1: Inversion (source_prompt, no guidance)
        info = {'feature': {}}
        noise, info = self.pipeline(
            latents=latents,
            prompt=source_prompt,
            height=target_height,
            width=target_width,
            video_length=target_video_length,
            inject_step=inject_step,
            num_inference_steps=infer_steps,
            attn_guidance_k=attn_guidance_k,
            attn_guidance_alpha=0.0,  # No guidance during inversion
            overlap_target=overlap_target,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            output_type="latent",
            freqs_cis=(freqs_cos, freqs_sin),
            n_tokens=n_tokens,
            embedded_guidance_scale=1,  # No embedded guidance during inversion
            data_type="video" if target_video_length > 1 else "image",
            inversion=True,
            info=info,
        )
        noise = noise[0]

        # Step 2: Sampling with embedding-level guidance (target_prompt)
        samples, _ = self.pipeline(
            latents=noise,
            prompt=target_prompt,
            height=target_height,
            width=target_width,
            video_length=target_video_length,
            inject_step=inject_step,
            num_inference_steps=infer_steps,
            attn_guidance_k=attn_guidance_k,
            attn_guidance_alpha=attn_guidance_alpha,
            overlap_target=overlap_target,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            output_type="pil",
            freqs_cis=(freqs_cos, freqs_sin),
            n_tokens=n_tokens,
            embedded_guidance_scale=embedded_guidance_scale,
            data_type="video" if target_video_length > 1 else "image",
            inversion=False,
            info=info,
        )
        samples = samples[0]
        out_dict["samples"] = samples
        out_dict["prompts"] = target_prompt

        gen_time = time.time() - start_time
        logger.info(f"Success, time: {gen_time}")
        return out_dict
