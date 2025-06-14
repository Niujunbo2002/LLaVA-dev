import os
import torch
import warnings
from torch import nn
from safetensors.torch import load_file
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

# ANSI color codes for CLI output
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'

class Qwen2_5_VisionTransformerPretrainedModelForLLaVA(nn.Module):
    def __init__(self, model_path, args):
        super().__init__()

        self.model_path = model_path
        self.vision_tower_name = model_path
        self.select_layer = -1
        self.select_feature = 'patch'

        self.min_token = getattr(args, "mm_min_image_token", 4)
        self.max_token = getattr(args, "mm_max_image_token", 2048)
        self.resize_image_size = getattr(args, "resize_image_size", None)

        self.is_loaded = False
        self.load_model(self.model_path)

    def load_model(self, model_path):
        """Load vision tower from pretrained model"""
        config = Qwen2_5_VLVisionConfig.from_pretrained(model_path)

        self.vision_tower = Qwen2_5_VisionTransformerPretrainedModel._from_config(
            config=config,
            use_flash_attention_2=True,
        ).half()

        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

        self.reset_image_processor(self.min_token, self.max_token)

    def reset_image_processor(self, min_tokens, max_tokens):
        """Initialize the image processor with token-based resolution bounds"""
        min_pixels = min_tokens * 28 * 28
        max_pixels = max_tokens * 28 * 28

        self.image_processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        self.image_processor.resize_image_size = self.resize_image_size

        print(f"{GREEN}Image token limits:\n  MIN_PIXELS: {min_tokens} * 28 * 28 = {min_pixels}\n"
              f"  MAX_PIXELS: {max_tokens} * 28 * 28 = {max_pixels}{RESET}")

    def forward(self, pixel_values, grid_thw):
        """
        Forward pass to extract visual features.

        Args:
            pixel_values (Tensor): Image tensor of shape [all_seq_len, C * H * W]
            grid_thw (Tensor): Tensor of shape [num_images, 3] where each row is [T, H, W] in patch units

        Returns:
            Tensor: Extracted image features of shape [total_tokens, hidden_size]
        """
        image_features = self.vision_tower(pixel_values, grid_thw=grid_thw)
        return image_features

    @property
    def dummy_feature(self):
        """Get dummy placeholder features (useful for padding or fallback)"""
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        """Return the default data type of the model"""
        return self.vision_tower.dtype

    @property
    def device(self):
        """Return the device the model is on"""
        return self.vision_tower.device

    @property
    def config(self):
        """Return the configuration object"""
        return self.vision_tower.config if self.is_loaded else self.cfg_only

    @property
    def hidden_size(self):
        """Return the hidden size of the vision model output"""
        return self.config.out_hidden_size
