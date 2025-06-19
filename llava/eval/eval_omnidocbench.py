import argparse
import torch
import os
import json
from tqdm import tqdm

import shortuuid
import sys
sys.path.append("/mnt/petrelfs/niujunbo/NativeRes-LLaVA")
import argparse
import torch
import warnings
warnings.filterwarnings("ignore")
from transformers import logging
logging.set_verbosity_warning()
from transformers import logging
logging.set_verbosity_error()

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<Mineru-Image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<Mineru_start>"
DEFAULT_IM_END_TOKEN = "<Mineru_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
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


from PIL import Image
import math
import warnings
warnings.filterwarnings("ignore")

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

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
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)



    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        None, 
        model_name,
        min_image_tokens=args.min_image_tokens,
        max_image_tokens=args.max_image_tokens,
    )

    output_path = os.path.expanduser(args.output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    data_path = os.path.expanduser(args.data_dir)
    
    image_files = os.listdir(data_path)
    image_paths = []
    has_subdirectories = False
    
    for image_file in image_files:
        full_path = os.path.join(data_path, image_file)
        if image_file.lower().endswith('.json'):
            continue
        if os.path.isdir(full_path):
            has_subdirectories = True
            sub_images = os.listdir(full_path)
            for sub_image in sub_images:
                sub_full_path = os.path.join(full_path, sub_image)
                if sub_full_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_paths.append(sub_full_path)
        else:
            if full_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(full_path)

    image_paths = get_chunk(image_paths, args.num_chunks, args.chunk_idx)
    for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths), desc="Processing Images"):
        if has_subdirectories:
            dir_name = os.path.basename(os.path.dirname(image_path))
            base_name = os.path.basename(image_path)
            file_name = f"{dir_name}-{base_name.rsplit('.', 1)[0]}.md"
        else:
            base_name = os.path.basename(image_path)
            file_name = f"{base_name.rsplit('.', 1)[0]}.md"
        save_path = os.path.join(output_path, file_name)
        if os.path.exists(save_path):
            continue
        
        # image = Image.open(image_path)
        # image_tensor = process_images([image], image_processor, model.config)[0]
        # images = image_tensor.unsqueeze(0).half().cuda()
        # image_sizes = [image.size]

        # if args.conv_mode == 'plain':
        #     prompt = DEFAULT_IMAGE_TOKEN + '\n'
        # else:
        #     question = DEFAULT_IMAGE_TOKEN + '\nOCR with format: '
        #     conv = conv_templates[args.conv_mode].copy()
        #     conv.append_message(conv.roles[0], question)
        #     conv.append_message(conv.roles[1], None)
        #     prompt = conv.get_prompt()
            
        # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        # with torch.inference_mode():
        #     output_ids = model.generate(
        #         input_ids,
        #         images=images,
        #         image_sizes=image_sizes,
        #         no_repeat_ngram_size = 20,
        #         num_beams = 1,
        #         do_sample=False,
        #         temperature=1.0,
        #         max_new_tokens=8192,
        #         use_cache=True,
        #         early_stopping=True,
        #     )
        #     outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
        #     with open(save_path, 'w', encoding='utf-8') as f:
        #         f.write(outputs)


        image_files = [image_path]
        images = load_images(image_files,packing)
        image_sizes = [x.size for x in images] if not packing else None
        images_tensor, grid_thw = process_images(
            images,
            image_processor,
            model.config,
            packing,
        )
        images_tensor=images_tensor.to(model.device, dtype=torch.float16)
        grid_thw=grid_thw.to(model.device) if grid_thw is not None else None

        qs = '\nOCR with format: '
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
                # qs = DEFAULT_IMAGE_TOKEN + "\n" + qs # <image>\nWhat are the things I should be cautious about when I visit here?
                num_images = len(image_files)
                qs = (DEFAULT_IMAGE_TOKEN + "\n") * num_images + qs

        conv_mode=args.conv_mode

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print(f"{GREEN}prompt: {prompt}\n{RESET}")

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                # do_sample=True if args.temperature > 0 else False,#true
                do_sample=False,
                # repetition_penalty=1.1,
                # temperature=args.temperature,
                # top_p=args.top_p,
                # top_k=20,
                # num_beams=args.num_beams,
                no_repeat_ngram_size = 100,
                repetition_penalty=1.0,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                packing=packing,
                grid_thw=grid_thw,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(outputs)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a multimodal model on a set of images.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--model-base", type=str, default=None, help="Base model name or path.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing images to process.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output markdown files.")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5") #qwen_1_5
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--min_image_tokens", type=int, default=4)
    parser.add_argument("--max_image_tokens", type=int, default=4000)
    parser.add_argument("--max_new_tokens", type=int, default=14000)


    args = parser.parse_args()
    
    eval_model(args)