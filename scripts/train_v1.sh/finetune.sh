#!/bin/bash

############################################
# ✅ CUDA & Compiler Environment
############################################

# CUDA 12.1
export CUDA_HOME="/mnt/petrelfs/share/test-cuda/cuda-12.1"
export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

# GCC 10.2.0
export GCC_HOME="/mnt/petrelfs/share/gcc/gcc-10.2.0"
export PATH="${GCC_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${GCC_HOME}/lib64:$LD_LIBRARY_PATH"

# GCC dependencies
export LD_LIBRARY_PATH="/mnt/petrelfs/share/gcc/mpc-0.8.1/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/mnt/petrelfs/share/gcc/gmp-4.3.2/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/mnt/petrelfs/share/gcc/mpfr-4.1.0/lib:$LD_LIBRARY_PATH"

# OpenMPI
export PATH="/mnt/petrelfs/share/openmpi/bin:$PATH"
export LD_LIBRARY_PATH="/mnt/petrelfs/share/openmpi/lib:$LD_LIBRARY_PATH"

# glibc
export PATH="/mnt/petrelfs/share/glibc-2.27/bin:$PATH"
export LD_LIBRARY_PATH="/mnt/petrelfs/share/glibc-2.27/lib64:$LD_LIBRARY_PATH"

# Compiler
export CC="${GCC_HOME}/bin/gcc"
export CXX="${GCC_HOME}/bin/g++"


############################################
# ✅ Runtime Environment Variables
############################################

# Threading & NCCL settings
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=""
export TOKENIZERS_PARALLELISM=false

# Image token length
export MIN_IMG_TOKEN=4
export MAX_IMG_TOKEN=4096

# Dataset & Model config
export DATA_PATH="/mnt/petrelfs/niujunbo/doc_parse/lmms-lab/LLaVA-NeXT-Data/llava_next_raw_format/llava_next_raw_format_processed.json"
export IMG_FOLDER="/mnt/petrelfs/niujunbo/doc_parse/lmms-lab/LLaVA-NeXT-Data/llava_next_raw_format"
export TRAIN_PARTS="mm_mlp_adapter"
export PER_DEVICE_BS=4
export ACC_BS=1
export MODEL_MAX_LEN=4096

# Model paths
export LLM_VERSION="/mnt/hwfile/mllm/niujunbo/model-image/Qwen/Qwen2-1.5B-Instruct"
export VISION_MODEL_VERSION="/mnt/petrelfs/niujunbo/niujunbo_dev/ocr_ckpts/qwen2_5_vl-668m-patch14-native"
export LLM_VERSION_CLEAN="$(basename "$LLM_VERSION")"
export VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# Cluster settings
export PARTITION="mineru2"
export NODES=1
export MASTER_PORT=12349
export PROMPT_VERSION="qwen_2"

# Naming & logging
export BASE_RUN_NAME="nativeres-llava-${LLM_VERSION_CLEAN}-qwenvit_2_5-pt"
export MID_RUN_NAME="NativeRes-${LLM_VERSION_CLEAN}-qwen2_5_vit-ft"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
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

srun -J debug \
    -p ${PARTITION} \
    --nodes=${NODES} \
    --ntasks-per-node=1 \
    --gres=gpu:8 \
    bash -c '
    ACCELERATE_CPU_AFFINITY=1 \
    torchrun --nproc_per_node=8 \
             --nnodes=${NODES} \
             --node_rank=${SLURM_NODEID} \
             --master_addr=$(scontrol show hostname $SLURM_NODELIST | head -n1) \
             --master_port=${MASTER_PORT} \
             llava/train/train_mem.py \
        --deepspeed scripts/zero2.json \
        --model_name_or_path ${CKPT_PATH} \
        --version ${PROMPT_VERSION} \
        --data_path ${DATA_PATH} \
        --image_folder ${IMG_FOLDER} \
        --pretrain_mm_mlp_adapter="/mnt/petrelfs/niujunbo/zhengyuanhong/NativeResLLaVA/checkpoints/projectors/${BASE_RUN_NAME}/mm_projector.bin" \
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
        2>&1 | tee ${OUTPUT_LOG_DIR}/train.log'
