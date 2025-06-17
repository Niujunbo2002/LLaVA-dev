#!/bin/bash

############################################
# ✅ Runtime Environment Variables
############################################

# Threading & NCCL settings
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_GID_INDEX=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=""
export CUDA_VISIBLE_DEVICES=2,3,4,5
export NUM_GPUS=4
export TOKENIZERS_PARALLELISM=false


# Image token length
export MIN_IMG_TOKEN=4
export MAX_IMG_TOKEN=4096

# Dataset & Model config
export DATA_PATH="/data/niujunbo/datasets/liuhaotian/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
export IMG_FOLDER="/data/niujunbo/datasets/liuhaotian/LLaVA-Pretrain/images"
export PER_DEVICE_BS=1
export ACC_BS=1
export MODEL_MAX_LEN=4096

# Model paths
export LLM_VERSION="/data/niujunbo/models/Niujunbo2002/NativeRes-LLaVA-qwen2-0.5b-qwen2vit"
export VISION_MODEL_VERSION="/home/mineru/Document/niujunbo/qwen2vl-665m-patch14-native"
export LLM_VERSION_CLEAN="$(basename "$LLM_VERSION")"
export VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# Cluster settings
export PARTITION="mineru2"
export NODES=1
export MASTER_PORT=12349
export PROMPT_VERSION="qwen_2"

# Naming & logging
export MID_RUN_NAME="NativeRes-${LLM_VERSION_CLEAN}-qwen2_5_vit-ft"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

export CKPT_PATH=${LLM_VERSION}  # Can be replaced with another checkpoint path if needed

# Logging setup
export SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export SCRIPT_PATH=$(realpath "$0")
export SCRIPT_NAME=$(basename "$SCRIPT_PATH")
export DATE=$(date +%m-%d)
export TIME=$(date +%H-%M)
export OUTPUT_LOG_DIR="${SCRIPT_DIR}/../../playground/training/${DATE}/${MID_RUN_NAME}"
export OUTPUT_CKPT_DIR="${OUTPUT_LOG_DIR}/checkpoints/${MID_RUN_NAME}"

mkdir -p "$OUTPUT_LOG_DIR"
cp "$SCRIPT_PATH" "${OUTPUT_LOG_DIR}/${SCRIPT_NAME}"


############################################
# ✅ Launch Training Job
############################################


ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}"  \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${CKPT_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_PATH} \
    --image_folder ${IMG_FOLDER} \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_vision_tower_lr=2e-6 \
    --mm_min_image_token ${MIN_IMG_TOKEN} \
    --mm_max_image_token ${MAX_IMG_TOKEN} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --run_name ${MID_RUN_NAME} \
    --output_dir ${OUTPUT_CKPT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${PER_DEVICE_BS} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${ACC_BS} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 50 \
    --tf32 True \
    --model_max_length ${MODEL_MAX_LEN} \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to none \