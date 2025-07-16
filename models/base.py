from pathlib import Path
import re

import peft
import torch
from torch import nn
import torch.nn.functional as F
import safetensors.torch
import torchvision
from PIL import Image, ImageOps
from torchvision import transforms
import imageio

from utils.common import is_main_process, VIDEO_EXTENSIONS, round_to_nearest_multiple, round_down_to_multiple


def make_contiguous(*tensors):
    return tuple(x.contiguous() for x in tensors)


# Remove video clipping function, keep complete video
# def extract_clips(video, target_frames, video_clip_mode):


# Remove size adjustment function, no longer needed
# def convert_crop_and_resize(pil_img, width_and_height):


class PreprocessMediaFile:
    def __init__(self, config, support_video=False, framerate=None, round_height=1, round_width=1, round_frames=1):
        self.config = config
        self.pil_to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.support_video = support_video
        self.framerate = framerate
        print(f'using framerate={self.framerate}')
        if self.support_video:
            assert self.framerate

    def __call__(self, filepath, mask_filepath, size_bucket=None):
        is_video = (Path(filepath).suffix in VIDEO_EXTENSIONS)
        if is_video:
            assert self.support_video
            num_frames = 0
            for frame in imageio.v3.imiter(filepath):
                num_frames += 1
                height, width = frame.shape[:2]
            video = imageio.v3.imiter(filepath)
        else:
            num_frames = 1
            pil_img = Image.open(filepath)
            height, width = pil_img.height, pil_img.width
            video = [pil_img]

        if mask_filepath:
            mask_img = Image.open(mask_filepath).convert('RGB')
            img_hw = (height, width)
            mask_hw = (mask_img.height, mask_img.width)
            if mask_hw != img_hw:
                raise ValueError(
                    f'Mask shape {mask_hw} was not the same as image shape {img_hw}.\n'
                    f'Image path: {filepath}\n'
                    f'Mask path: {mask_filepath}'
                )
            mask = torchvision.transforms.functional.to_tensor(mask_img)[0].to(torch.float16)  # use first channel
        else:
            mask = None

        resized_video = torch.empty((num_frames, 3, height, width))
        for i, frame in enumerate(video):
            if not isinstance(frame, Image.Image):
                frame = torchvision.transforms.functional.to_pil_image(frame)
            if frame.mode not in ['RGB', 'RGBA'] and 'transparency' in frame.info:
                frame = frame.convert('RGBA')
            if frame.mode == 'RGBA':
                canvas = Image.new('RGBA', frame.size, (255, 255, 255))
                canvas.alpha_composite(frame)
                frame = canvas.convert('RGB')
            else:
                frame = frame.convert('RGB')
            resized_video[i, ...] = self.pil_to_tensor(frame)

        if not self.support_video:
            return [(resized_video.squeeze(0), mask)]

        # (num_frames, channels, height, width) -> (channels, num_frames, height, width)
        resized_video = torch.permute(resized_video, (1, 0, 2, 3))
        if not is_video:
            return [(resized_video, mask)]
        else:
            return [(resized_video, mask)]


class BasePipeline:
    framerate = None

    def load_diffusion_model(self):
        pass

    def get_vae(self):
        raise NotImplementedError()

    def get_text_encoders(self):
        raise NotImplementedError()

    def configure_adapter(self, adapter_config):
        target_linear_modules = set()
        exclude_linear_modules = adapter_config.get('exclude_linear_modules', [])
        for name, module in self.transformer.named_modules():
            if module.__class__.__name__ not in self.adapter_target_modules:
                continue
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, nn.Linear):
                    # Check if the complete path contains module names to be excluded
                    should_exclude = any(exclude_name in full_submodule_name for exclude_name in exclude_linear_modules)
                    if not should_exclude:
                        target_linear_modules.add(full_submodule_name)
        target_linear_modules = list(target_linear_modules)

        adapter_type = adapter_config['type']
        if adapter_type == 'lora':
            peft_config = peft.LoraConfig(
                r=adapter_config['rank'],
                lora_alpha=adapter_config['alpha'],
                lora_dropout=adapter_config['dropout'],
                bias='none',
                target_modules=target_linear_modules
            )
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')
        self.peft_config = peft_config
        self.lora_model = peft.get_peft_model(self.transformer, peft_config)
        if is_main_process():
            self.lora_model.print_trainable_parameters()
        for name, p in self.transformer.named_parameters():
            p.original_name = name
            if p.requires_grad:
                p.data = p.data.to(adapter_config['dtype'])

    def save_adapter(self, save_dir, peft_state_dict):
        raise NotImplementedError()

    def load_adapter_weights(self, adapter_path):
        if is_main_process():
            print(f'Loading adapter weights from path {adapter_path}')
        safetensors_files = list(Path(adapter_path).glob('*.safetensors'))
        if len(safetensors_files) == 0:
            raise RuntimeError(f'No safetensors file found in {adapter_path}')
        if len(safetensors_files) > 1:
            raise RuntimeError(f'Multiple safetensors files found in {adapter_path}')
        adapter_state_dict = safetensors.torch.load_file(safetensors_files[0])
        modified_state_dict = {}
        model_parameters = set(name for name, p in self.transformer.named_parameters())
        for k, v in adapter_state_dict.items():
            # Replace Diffusers or ComfyUI prefix
            k = re.sub(r'^(transformer|diffusion_model)\.', '', k)
            # Replace weight at end for LoRA format
            k = re.sub(r'\.weight$', '.default.weight', k)
            if k not in model_parameters:
                raise RuntimeError(f'modified_state_dict key {k} is not in the model parameters')
            modified_state_dict[k] = v
        self.transformer.load_state_dict(modified_state_dict, strict=False)

    def save_model(self, save_dir, diffusers_sd):
        raise NotImplementedError()

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(self.config, support_video=False)

    def get_call_vae_fn(self, vae):
        raise NotImplementedError()

    def get_call_text_encoder_fn(self, text_encoder):
        raise NotImplementedError()

    def prepare_inputs(self, inputs, timestep_quantile=None):
        raise NotImplementedError()

    def to_layers(self):
        raise NotImplementedError()

    def model_specific_dataset_config_validation(self, dataset_config):
        pass

    # Get param groups that will be passed into the optimizer. Models can override this, e.g. SDXL
    # supports separate learning rates for unet and text encoders.
    def get_param_groups(self, parameters):
        return parameters

    # Default loss_fn. MSE between output and target, with mask support.
    def get_loss_fn(self):
        def loss_fn(output, label):
            target, mask = label
            with torch.autocast('cuda', enabled=False):
                output = output.to(torch.float32)
                target = target.to(output.device, torch.float32)
                loss = F.mse_loss(output, target, reduction='none')
                # empty tensor means no masking
                if mask.numel() > 0:
                    mask = mask.to(output.device, torch.float32)
                    loss *= mask
                loss = loss.mean()
            return loss
        return loss_fn

    def enable_block_swap(self, blocks_to_swap):
        raise NotImplementedError('Block swapping is not implemented for this model')

    def prepare_block_swap_training(self):
        pass

    def prepare_block_swap_inference(self, disable_block_swap=False):
        pass