
This is a repo enabling you train a LLaVA using images with any resolution.
## Install
Install the required environment in `requirements.txt`. The Transforms version should be able to support at least `Qwen2-VL model`.
## Run
### Pretrain

If you want to run using siglip ViT, which not support NativeRes, you can run:
```
bash scripts/train/pretrain_siglip.sh
```
Otherwise you can run in NativeRes mode which utilize Qwen2-VL ViT to support any resolution:
```
bash scripts/train/pretrain_qwenvit.sh
```

### Finetune
For finetuning using siglip, just run
```
bash scripts/train/direct_finetune_siglip_a4_v1.5.sh
```
Otherwise you can run in NativeRes mode by:(using the LLaVA1.5 Fintuning Dataset now, you can change it anyway.)
```
bash scripts/train/direct_finetune_qwen_a4_v1.5_4_2048.sh
```

### Inference
For Inference, we have a simple example, just run:
```
python llava/eval/model_vqa.py
```

## Notes
1. Still not support zero3 in NativeRes mode now.
2. Update `sys.path.append("/mnt/petrelfs/niujunbo/zhengyuanhong/NativeResLLaVA")` to your personal path.
3. Still not support `video` now.
