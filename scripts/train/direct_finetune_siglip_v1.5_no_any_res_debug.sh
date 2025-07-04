export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=1  # 添加这行
export NCCL_IB_GID_INDEX=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1

LLM_VERSION="/root/Downloads/model/Qwen/Qwen2-0.5B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="/root/Downloads/zhengyuanhong/model/google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
MID_RUN_NAME="LLaVA-Next-Qwen2-0.5B-Instruct-FT"
############### Pretrain ################

PROMPT_VERSION="qwen_1_5"

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}"  \
    -m debugpy --listen localhost:8164 --wait-for-client \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path /root/Downloads/dataset/liuhaotian/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /root/Downloads/dataset/liuhaotian/LLaVA-Pretrain/images \
    --pretrain_mm_mlp_adapter="/checkpoints/projectors/${BASE_RUN_NAME}/mm_projector.bin" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio pad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir "/checkpoints/${MID_RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    # --attn_implementation sdpa

# You can delete the sdpa attn_implementation if you want to use flash attn