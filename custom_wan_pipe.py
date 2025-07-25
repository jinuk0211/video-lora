import types
from diffsynth.models import ModelManager
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.models.wan_video_text_encoder import WanTextEncoder
from diffsynth.models.wan_video_vae import WanVideoVAE
from diffsynth.models.wan_video_image_encoder import WanImageEncoder
from diffsynth.schedulers.flow_match import FlowMatchScheduler
from diffsynth.pipelines.base import BasePipeline
from diffsynth.prompters import WanPrompter
import torch, os
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional

from diffsynth.vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from diffsynth.models.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from diffsynth.models.wan_video_dit import RMSNorm, sinusoidal_embedding_1d
from diffsynth.models.wan_video_vae import RMS_norm, CausalConv3d, Upsample
from diffsynth.models.wan_video_motion_controller import WanMotionControllerModel



class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.model_names = ['text_encoder', 'dit', 'vae', 'image_encoder', 'motion_controller']
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.use_unified_sequence_parallel = False


    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.motion_controller is not None:
            dtype = next(iter(self.motion_controller.parameters())).dtype
            enable_vram_management(
                self.motion_controller,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()


    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        self.motion_controller = model_manager.fetch_model("wan_video_motion_controller")


    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None, use_usp=False):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

            for block in pipe.dit.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            pipe.dit.forward = types.MethodType(usp_dit_forward, pipe.dit)
            pipe.sp_size = get_sequence_parallel_world_size()
            pipe.use_unified_sequence_parallel = True
        return pipe
    
    
    def denoising_model(self):
        return self.dit


    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive, device=self.device)
        return {"context": prompt_emb}
    
    
    def encode_image(self, image, pseudo_video_path=None, mask_video_path=None, end_image=None, num_frames=81, height=480, width=832):
        """
        Encode images and videos for I2V inference
        Args:
            image: First frame image (PIL Image)
            pseudo_video_path: Pseudo video file path (new mode)
            mask_video_path: Mask video file path (new mode)
            end_image: End frame image (original mode)
            num_frames: Number of frames
            height, width: Video dimensions
        """
        # New mode: Use pseudo video and mask video
        if pseudo_video_path is not None and mask_video_path is not None:
            import cv2
            import numpy as np
            import torch.nn.functional as F
            
            # Read pseudo video
            cap_pseudo = cv2.VideoCapture(pseudo_video_path)
            pseudo_frames = []
            while True:
                ret, frame = cap_pseudo.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pseudo_frames.append(Image.fromarray(frame))
            cap_pseudo.release()
            
            # Read mask video
            cap_mask = cv2.VideoCapture(mask_video_path)
            mask_frames = []
            while True:
                ret, frame = cap_mask.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mask_frames.append(Image.fromarray(frame))
            cap_mask.release()
            
            # Determine dimensions and frame count based on pseudo video
            num_frames = len(pseudo_frames)
            width, height = pseudo_frames[0].size
            
            # Check if first frame image and pseudo video dimensions are consistent
            input_width, input_height = image.size
            assert input_width == width and input_height == height, \
                f"Edited image size ({input_width}x{input_height}) must match source video size ({width}x{height})"
                
            # Ensure mask video frame count matches
            if len(mask_frames) != num_frames:
                raise ValueError(f"Mask video frames ({len(mask_frames)}) don't match pseudo video frames ({num_frames})")
                
            # Resize all frames to target dimensions
            pseudo_frames = [frame.resize((width, height)) for frame in pseudo_frames]
            mask_frames = [frame.resize((width, height)) for frame in mask_frames]
            
            # Replace first frame of pseudo video with input image
            first_frame = image.resize((width, height))
            pseudo_frames[0] = first_frame
            
            # Set first frame mask to all black
            black_mask = Image.new('RGB', (width, height), (0, 0, 0))
            mask_frames[0] = black_mask
            
            # Process first frame for CLIP encoding
            first_frame_tensor = self.preprocess_image(first_frame).to(self.device)
            clip_context = self.image_encoder.encode_image([first_frame_tensor])
            
            # Convert pseudo video to tensor (as condition)
            pseudo_tensors = []
            for frame in pseudo_frames:
                tensor = self.preprocess_image(frame)  # [1, 3, 480, 832]
                tensor = tensor.squeeze(0)  # Remove batch dimension -> [3, 480, 832]
                pseudo_tensors.append(tensor)
            
            # Stack along time dimension: [3, 480, 832] -> [3, T, 480, 832]
            pseudo_video_tensor = torch.stack(pseudo_tensors, dim=1)
            
            # Add batch dimension: [3, T, 480, 832] -> [1, 3, T, 480, 832]
            pseudo_video_tensor = pseudo_video_tensor.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            # Convert mask video to tensor
            mask_tensors = []
            for frame in mask_frames:
                tensor = self.preprocess_image(frame).squeeze(0)  # Remove batch dimension
                mask_tensors.append(tensor)
            # Same processing
            mask_video_tensor = torch.stack(mask_tensors, dim=1)  # [3, T, 480, 832]
            mask_video_tensor = mask_video_tensor.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)  # [1, 3, T, 480, 832]
            
            # VAE encode pseudo video (as condition y)
            # VAE expects format [C, T, H, W], so need to remove batch dimension
            pseudo_video_for_vae = pseudo_video_tensor.squeeze(0)  # [3, T, H, W]
            y = self.vae.encode([pseudo_video_for_vae], device=self.device)[0]
            
            # Process mask_frames following training code logic
            bs, c, f, h_orig, w_orig = mask_video_tensor.shape
            
            # 1. Convert to single channel (calculate mean)
            mask = mask_video_tensor.mean(dim=1, keepdim=False)  # (bs, f, h_orig, w_orig)
            
            # 2. Interpolate to latent space resolution
            h_latent, w_latent = height//8, width//8
            mask = F.interpolate(
                mask, 
                size=(h_latent, w_latent),
                mode='nearest'
            )
            
            # 3. Binarize: black (low values) corresponds to 1, white (high values) corresponds to 0
            # Assuming input is in [-1,1] range, first normalize to [0,1]
            mask = (mask + 1) / 2
            mask = (mask < 0.5).float()  # Black regions as 1, white regions as 0
            
            mask = torch.concat([torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1), mask[:, 1:]], dim=1)
            mask = mask.view(mask.shape[0], mask.shape[1] // 4, 4, mask.shape[2], mask.shape[3])
            mask = mask.transpose(1, 2)  # (bs, 4, f, h, w)
            
            # VAE output y is 4D, need to remove mask's batch dimension to match
            mask = mask.squeeze(0)  # Remove batch dimension (4, f, h, w)
            
            # Concatenate mask with y, same as original logic
            y = torch.cat([mask, y], dim=0)  # Change to concatenate along dim 0
            
            # Add batch dimension to match expected format in model_fn_wan_video
            y = y.unsqueeze(0)  # (c, f, h, w) -> (1, c, f, h, w)
            
            clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
            y = y.to(dtype=self.torch_dtype, device=self.device)
            
            return {
                "clip_feature": clip_context, 
                "y": y,
                "num_frames": num_frames,
                "height": height,
                "width": width
            }
        
        # Original mode: use end_image (backward compatibility)
        else:
            image = self.preprocess_image(image.resize((width, height))).to(self.device)
            clip_context = self.image_encoder.encode_image([image])
            msk = torch.ones(1, num_frames, height//8, width//8, device=self.device)
            msk[:, 1:] = 0
            if end_image is not None:
                end_image = self.preprocess_image(end_image.resize((width, height))).to(self.device)
                vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
                msk[:, -1:] = 1
            else:
                vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
            msk = msk.transpose(1, 2)[0]
            
            y = self.vae.encode([vae_input.to(dtype=self.torch_dtype, device=self.device)], device=self.device)[0]
            y = torch.concat([msk, y])
            y = y.unsqueeze(0)
            clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
            y = y.to(dtype=self.torch_dtype, device=self.device)
            return {"clip_feature": clip_context, "y": y}
    
    
    def encode_control_video(self, control_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        control_video = self.preprocess_images(control_video)
        # preprocess_images returns tensor list, each may be [1, 3, H, W], need to remove batch dimension
        control_video = [tensor.squeeze(0) if tensor.dim() == 4 else tensor for tensor in control_video]
        control_video = torch.stack(control_video, dim=1)  # [3, T, H, W]
        control_video = control_video.to(dtype=self.torch_dtype, device=self.device)  # [3, T, H, W] - VAE expects this format
        latents = self.encode_video(control_video, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=self.torch_dtype, device=self.device)
        return latents
    
    
    def prepare_controlnet_kwargs(self, control_video, num_frames, height, width, clip_feature=None, y=None, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        if control_video is not None:
            control_latents = self.encode_control_video(control_video, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            if clip_feature is None or y is None:
                clip_feature = torch.zeros((1, 257, 1280), dtype=self.torch_dtype, device=self.device)
                y = torch.zeros((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), dtype=self.torch_dtype, device=self.device)
            else:
                y = y[:, -16:]
            y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}


    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames
    
    
    def prepare_extra_input(self, latents=None):
        return {}
    
    
    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    
    
    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames
    
    
    def prepare_unified_sequence_parallel(self):
        return {"use_unified_sequence_parallel": self.use_unified_sequence_parallel}
    
    
    def prepare_motion_bucket_id(self, motion_bucket_id):
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(dtype=self.torch_dtype, device=self.device)
        return {"motion_bucket_id": motion_bucket_id}


    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image=None,
        pseudo_video_path=None,
        mask_video_path=None,
        end_image=None,
        input_video=None,
        control_video=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        motion_bucket_id=None,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        tea_cache_l1_thresh=None,
        tea_cache_model_id="",
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        # Parameter check
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")
        
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Initialize noise
        if pseudo_video_path is not None and mask_video_path is not None:
            # For new I2V mode, preprocess first to get actual video information
            self.load_models_to_device(["image_encoder", "vae"])
            temp_image_emb = self.encode_image(input_image, pseudo_video_path, mask_video_path)
            actual_height = temp_image_emb["height"]
            actual_width = temp_image_emb["width"]
            actual_num_frames = temp_image_emb["num_frames"]
            
            # Use actual video parameters
            noise = self.generate_noise(
                (1, 16, (actual_num_frames - 1) // 4 + 1, actual_height//8, actual_width//8), 
                seed=seed, device=rand_device, dtype=torch.float32
            )
        else:
            # Original logic
            noise = self.generate_noise(
                (1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), 
                seed=seed, device=rand_device, dtype=torch.float32
            )
            
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        if input_video is not None:
            self.load_models_to_device(['vae'])
            input_video = self.preprocess_images(input_video)
            # Remove possible batch dimension
            input_video = [tensor.squeeze(0) if tensor.dim() == 4 else tensor for tensor in input_video]
            input_video = torch.stack(input_video, dim=1)  # [3, T, H, W]
            input_video = input_video.to(dtype=self.torch_dtype, device=self.device)  # [3, T, H, W] - VAE expects this format
            latents = self.encode_video(input_video, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = noise
        
        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
            
        # Encode image
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            if pseudo_video_path is not None and mask_video_path is not None:
                # New I2V mode
                image_emb = self.encode_image(input_image, pseudo_video_path, mask_video_path)
            else:
                # Original mode (backward compatibility)
                image_emb = self.encode_image(input_image, end_image=end_image, num_frames=num_frames, height=height, width=width)
        else:
            image_emb = {}
            
        # ControlNet
        if control_video is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.prepare_controlnet_kwargs(control_video, num_frames, height, width, **image_emb, **tiler_kwargs)
            
        # Motion Controller
        if self.motion_controller is not None and motion_bucket_id is not None:
            motion_kwargs = self.prepare_motion_bucket_id(motion_bucket_id)
        else:
            motion_kwargs = {}
            
        # Extra input
        extra_input = self.prepare_extra_input(latents)
        
        # TeaCache
        tea_cache_posi = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}
        tea_cache_nega = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}
        
        # Unified Sequence Parallel
        usp_kwargs = self.prepare_unified_sequence_parallel()

        # Denoise
        self.load_models_to_device(["dit", "motion_controller"])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Inference
            noise_pred_posi = model_fn_wan_video(
                self.dit, motion_controller=self.motion_controller,
                x=latents, timestep=timestep,
                **prompt_emb_posi, **image_emb, **extra_input,
                **tea_cache_posi, **usp_kwargs, **motion_kwargs
            )
            if cfg_scale != 1.0:
                noise_pred_nega = model_fn_wan_video(
                    self.dit, motion_controller=self.motion_controller,
                    x=latents, timestep=timestep,
                    **prompt_emb_nega, **image_emb, **extra_input,
                    **tea_cache_nega, **usp_kwargs, **motion_kwargs
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)

        # Decode
        self.load_models_to_device(['vae'])
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        return frames



class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    x: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    **kwargs,
):
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)
    
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)
    
    if dit.has_image_input:
        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    
    x, (f, h, w) = dit.patchify(x)
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
    
    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        for block in dit.blocks:
            x = block(x, context, t_mod, freqs)
        if tea_cache is not None:
            tea_cache.store(x)

    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
    x = dit.unpatchify(x, (f, h, w))
    return x