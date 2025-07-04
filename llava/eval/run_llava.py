import sys
sys.path.append('/root/Downloads/zhengyuanhong/LLaVA')  # 替换为你的实际路径
import argparse
import torch
import warnings
warnings.filterwarnings("ignore")
from transformers import logging
logging.set_verbosity_warning()
from transformers import logging
logging.set_verbosity_error()

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
# ANSI 颜色代码
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files,packing):
    if packing:
        return image_files
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    packing=False
    if 'qwen' in args.model_path.lower():
        packing=True
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    #绿色输出loading model
    print(f"{GREEN}Loading model from {args.model_path} ...{RESET}")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        None, 
        model_name,
        min_image_tokens=args.min_image_tokens,
        max_image_tokens=args.max_image_tokens,
    )
    print(f"{GREEN}Model loaded successfully!\n{RESET}")
    
    qs = args.query
    qs=qs.replace("\\n", "\n")
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs # <image>\nWhat are the things I should be cautious about when I visit here?
    
    conv_mode="vicuna_v1"#!lmms-eval用的时候设置的事这个
    print(f"{GREEN}conv_mode: {conv_mode}\n{RESET}")

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files,packing)#如果packing就返回路径,否则返回图像
    image_sizes = [x.size for x in images] if not packing else None
    images_tensor, grid_thw = process_images(#images不可能为空,输出的images_tensor直接作为prepare_inputs...函数的输入
        images,
        image_processor,
        model.config,
        packing,
    )
    images_tensor=images_tensor.to(model.device, dtype=torch.float16)
    grid_thw=grid_thw.to(model.device) if grid_thw is not None else None
    
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        # import ipdb;ipdb.set_trace()
        output_ids = model.generate(
            input_ids,#
            images=images_tensor,#
            image_sizes=image_sizes,#
            # do_sample=True if args.temperature > 0 else False,#true
            do_sample=False,
            temperature=args.temperature,#0.2
            top_p=args.top_p,
            num_beams=args.num_beams,#1
            max_new_tokens=args.max_new_tokens,#512
            use_cache=True,
            packing=packing,
            grid_thw=grid_thw,
        )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    # 打印带颜色的输出
    print(f"{GREEN}输出的回答是:{RESET}")
    print(f"{BLUE}{outputs}{RESET}")


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 在导入深度学习框架（如 PyTorch/TensorFlow）之前设置

    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="/root/Downloads/zhengyuanhong/ckpts/source-llava-5198")
    # parser.add_argument("--model-path", type=str, default="/root/Downloads/model/liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-path", type=str, default="/root/Downloads/zhengyuanhong/ckpts/source-llava-qwenvit-5000")
    # parser.add_argument("--model-path", type=str, default="/root/Downloads/zhengyuanhong/ckpts/source-llava-qwenvit-1200_consolidate")
    parser.add_argument("--model-base", type=str, default=None)#设置成None别动
    parser.add_argument("--image-file", type=str, default="/root/Downloads/zhengyuanhong/Dataset/MME/MME_Benchmark_release_version/MME_Benchmark/artwork/images/3241.jpg")
    # parser.add_argument("--image-file", type=str, default="/root/Downloads/zhengyuanhong/LLaVA/demo.jpeg")
    # parser.add_argument("--image-file", type=str, default="/root/Downloads/zhengyuanhong/LLaVA/000000052846.jpg")
    # parser.add_argument("--image-file", type=str, default="/root/Downloads/zhengyuanhong/LLaVA/resized_view.jpg")
    # parser.add_argument("--query", type=str, default="What are the things I should be cautious about when I visit here?")
    # parser.add_argument("--query", type=str, default="Describe the image in detail." )
    parser.add_argument("--query", type=str, default="Is this artwork titled friedrich iii, the wise, and johann i, the constant, electors of saxony?\nAnswer the question using a single word or phrase." )
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--min_image_tokens", type=int, default=256)
    parser.add_argument("--max_image_tokens", type=int, default=1280)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)

#srun -J debug -p mineru2_data --gres=gpu:1 python llava/eval/run_llava.py
