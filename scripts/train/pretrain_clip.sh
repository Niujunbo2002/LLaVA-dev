#!/bin/bash
export PATH="/mnt/petrelfs/share/test-cuda/cuda-12.1/bin:$PATH"
export LD_LIBRARY_PATH="/mnt/petrelfs/share/test-cuda/cuda-12.1/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME="/mnt/petrelfs/share/test-cuda/cuda-12.1"
export GCC_HOME="/mnt/petrelfs/share/gcc/gcc-10.2.0"
export LD_LIBRARY_PATH=$GCC_HOME/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PATH=$GCC_HOME/bin:$PATH
export LD_LIBRARY_PATH="/mnt/petrelfs/share/gcc/mpc-0.8.1/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/mnt/petrelfs/share/gcc/gmp-4.3.2/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/mnt/petrelfs/share/gcc/mpfr-4.1.0/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/mnt/petrelfs/share/openmpi/lib:$LD_LIBRARY_PATH"
export CXX=$GCC_HOME/bin/g++
export CC=$GCC_HOME/bin/gcc
export PATH=/mnt/petrelfs/share/glibc-2.27/bin:$PATH
export PATH=/mnt/petrelfs/share/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/glibc-2.27/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=""

LLM_VERSION="/mnt/hwfile/mllm/niujunbo/model-image/Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="/mnt/hwfile/mllm/niujunbo/model-image/openai/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

PROMPT_VERSION=plain

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path /blip_558k/blip_558k_plain.json \
    --image_folder /blip_558k/images \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /checkpoints/projectors/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 50000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $BASE_RUN_NAME \
    --attn_implementation sdpa

# You can delete the sdpa attn_implementation if you want to use flash attn