#!/bin/bash

############################################
# ✅ Runtime Environment Variables
############################################

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=""
export TOKENIZERS_PARALLELISM=false

export MIN_IMG_TOKEN=4
export MAX_IMG_TOKEN=4096

############################################
# ✅ Model Version Configuration
############################################

# LLM model (choose one)
# export LLM_VERSION="/mnt/hwfile/mllm/niujunbo/model-image/Qwen/Qwen2.5-0.5B-Instruct"
export LLM_VERSION="/mnt/hwfile/mllm/niujunbo/model-image/Qwen/Qwen2.5-1.5B-Instruct"
# export LLM_VERSION="/mnt/hwfile/mllm/niujunbo/model-image/Qwen/Qwen2-7B-Instruct"

export LLM_VERSION_CLEAN="$(basename "$LLM_VERSION")"

# Vision model
export VISION_MODEL_VERSION="/mnt/petrelfs/niujunbo/niujunbo_dev/ocr_ckpts/qwen2_5_vl-668m-patch14-native"

# Data & Training Parameters
export DATA_PATH="/mnt/petrelfs/niujunbo/zhengyuanhong/LLaVA/playground/data/blip_laion_cc_sbu_558k.json"
export IMG_FOLDER="/mnt/hwfile/opendatalab/bigdata_mineru/niujunbo/dataset/llava/LLaVA-Pretrain"
export TRAIN_PARTS="mm_mlp_adapter"
export PER_DEVICE_BS=8
export ACC_BS=2
export MODEL_MAX_LEN=4096

############################################
# ✅ Job Metadata and Logging
############################################

export SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export SCRIPT_PATH=$(realpath "$0")
export SCRIPT_NAME=$(basename "$SCRIPT_PATH")

export DATE=$(date +%m-%d)

export BASE_RUN_NAME="NativeRes-${LLM_VERSION_CLEAN}-qwen2_5_vit-pt"

echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

mkdir -p "$SCRIPT_DIR/../../playground/training/$DATE/$BASE_RUN_NAME"
cp "$SCRIPT_PATH" "$SCRIPT_DIR/../../playground/training/$DATE/$BASE_RUN_NAME/$SCRIPT_NAME"

############################################
# ✅ SLURM and Training Config
############################################

export PROMPT_VERSION="plain"
export PARTITION="mineru2"
export NODES=1
export CPUS=128
export MASTER_PORT=12351

############################################
# ✅ Launch Training
############################################

srun -J debug \
    -p "$PARTITION" \
    --nodes="$NODES" \
    --ntasks-per-node=1 \
    --gres=gpu:8 \
    bash -c 'ACCELERATE_CPU_AFFINITY=1 torchrun \
        --nproc_per_node=8 \
        --nnodes=${NODES} \
        --node_rank=${SLURM_NODEID} \
        --master_addr=$(scontrol show hostname $SLURM_NODELIST | head -n1) \
        --master_port=${MASTER_PORT} \
        llava/train/train_mem.py \
        --deepspeed scripts/zero2.json \
        --model_name_or_path ${LLM_VERSION} \
        --version ${PROMPT_VERSION} \
        --data_path ${DATA_PATH} \
        --image_folder ${IMG_FOLDER} \
        --vision_tower ${VISION_MODEL_VERSION} \
        --mm_tunable_parts ${TRAIN_PARTS} \
        --mm_vision_select_layer -1 \
        --mm_projector_type mlp2x_gelu \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --bf16 True \
        --output_dir ${SCRIPT_DIR}/../../playground/training/${DATE}/${BASE_RUN_NAME}/checkpoints/projectors/${BASE_RUN_NAME} \
        --num_train_epochs 1 \
        --mm_min_image_token ${MIN_IMG_TOKEN} \
        --mm_max_image_token ${MAX_IMG_TOKEN} \
        --per_device_train_batch_size ${PER_DEVICE_BS} \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps ${ACC_BS} \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 500 \
        --learning_rate 1e-3 \
        --weight_decay 0.0 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 50 \
        --tf32 True \
        --model_max_length ${MODEL_MAX_LEN} \
        --gradient_checkpointing True \
        --dataloader_num_workers 16 \
        --lazy_preprocess True \
        --report_to none \
        --run_name ${BASE_RUN_NAME} \
        2>&1 | tee ${SCRIPT_DIR}/../../playground/training/${DATE}/${BASE_RUN_NAME}/train.log'
