import os
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
)
from transformers import AutoConfig, AutoProcessor
from safetensors.torch import load_file
from torch import nn
import torch
import warnings
# ANSI 颜色代码
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'

class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs

class Qwen2VisionTransformerPretrainedModelForLLaVA(nn.Module):
    def __init__(self, model_path,args):
        super().__init__()

        self.is_loaded = False
        self.model_path = model_path
        self.vision_tower_name = model_path
        self.select_layer = -1
        self.select_feature = 'patch'
        self.min_token=getattr(args,"mm_min_image_token",4)
        self.max_token=getattr(args,"mm_max_image_token",2048)
        self.resize_image_size=getattr(args,"resize_image_size",None)
        self.load_model(self.model_path)

    def load_model(self, model_path):
        """
        加载 Qwen2VisionTransformerPretrainedModel，但丢弃 merger 部分。
        """
        config = AutoConfig.from_pretrained(model_path)
        visual_model = Qwen2VisionTransformerPretrainedModel._from_config(
            config=config.vision_config,
            use_flash_attention_2=True, 
        )

        # 关键步骤 1: 用一个 Identity 层替换 merger
        # Identity 层是一个占位符，它不执行任何操作，也没有任何参数
        print(f"{GREEN}Replacing the 'merger' module...{RESET}")
        visual_model.merger = torch.nn.Identity()

        # import pdb; pdb.set_trace()
        checkpoint_path = os.path.join(model_path, "model-00001-of-00002.safetensors")
        
        print(f"{GREEN}Loading QwenViT weights (excluding merger)...{RESET}")
        
        checkpoint = load_file(checkpoint_path)
        visual_weights = {
            key.replace("visual.", ""): value
            for key, value in checkpoint.items()
            if key.startswith("visual.")
        }
        
        # 关键步骤 2: 使用 strict=False 加载权重
        # 这将加载所有匹配的键，并忽略 visual_weights 中存在但模型中已不存在的键（即 merger 的权重）
        missing_keys, unexpected_keys = visual_model.load_state_dict(visual_weights, strict=False)
        
        # 打印出未加载的权重，以确认它们确实是 merger 的权重
        print(f"{GREEN}Weights loaded. Unexpected keys (should be merger weights):{RESET}")
        print(unexpected_keys)


        # 确认没有丢失任何必要的权重
        # missing_keys 应该为空
        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys}")

        print(f"{GREEN}QwenViT loaded successfully without merger!{RESET}")
        self.vision_tower = visual_model
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True
        # self.image_processor = AutoProcessor.from_pretrained(model_path)
        self.reset_image_processor(self.min_token,self.max_token)
        self.image_processor.resize_image_size=self.resize_image_size
        
    def reset_image_processor(self, min_tokens, max_tokens):
        min_pixels=min_tokens*28*28
        max_pixels=max_tokens*28*28
        self.image_processor = AutoProcessor.from_pretrained(self.model_path,min_pixels=min_pixels,max_pixels=max_pixels)
        # Simplified output format
        print(f"{GREEN}MIN_PIXELS: {min_tokens} * 28 * 28 \nMAX_PIXELS: {max_tokens} * 28 * 28{RESET}")

    def forward(self, pixel_values, grid_thw):
        """
        pixel_values:[all_seq_len,patch_size*patch_size*3*2]
        image_grid_thw:[num_img,3],每个长度为3的向量为[1,h,w],1表示时间,如果为video,则会大于1.h,w为图像的高和宽(以patch为单位)
        """
        return self.vision_tower(pixel_values, grid_thw=grid_thw)#[all_seq_len//4,hidden_size(1536)]
    
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):#!应该需要改成bf16
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return 1280


if __name__=="__main__":
    config=AutoConfig.from_pretrained("/root/Downloads/zhengyuanhong/model/Qwen2-VL-2B-Instruct")
