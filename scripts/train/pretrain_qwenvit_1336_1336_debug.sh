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
# export NCCL_DEBUG=INFO
export NCCL_DEBUG=""

export LLM_VERSION="/mnt/hwfile/mllm/niujunbo/model-image/Qwen/Qwen2-7B-Instruct"
export LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
export VISION_MODEL_VERSION="/mnt/hwfile/mllm/niujunbo/model-image/Qwen/Qwen2-VL-2B-Instruct"
export VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

export PROMPT_VERSION=plain

export BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

export PARTITION="mineru2"
export NODES=1
export CPUS=128
export MASTER_PORT=12349

srun -J debug \
    -p $PARTITION \
    --nodes=$NODES \
    --ntasks-per-node=1 \
    --gres=gpu:1 \
    --cpus-per-task=$CPUS \
    bash -c  'ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node 1 --nnodes ${NODES} --node_rank ${SLURM_NODEID} --master_addr $(scontrol show hostname $SLURM_NODELIST | head -n1) --master_port ${MASTER_PORT} llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path /mnt/petrelfs/niujunbo/zhengyuanhong/LLaVA/playground/data/blip_laion_cc_sbu_558k.json \
    --image_folder /mnt/hwfile/opendatalab/bigdata_mineru/niujunbo/dataset/llava/LLaVA-Pretrain \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -1 \
    --mm_projector_type mlp2x_gelu \
    --mm_min_image_token 1336 \
    --mm_max_image_token 1336 \
    --resize_image_size 1024 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /mnt/hwfile/doc_parse/niujunbo/llava/checkpoints/projectors/${BASE_RUN_NAME}_1336 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 5192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $BASE_RUN_NAME \
    2>&1 | tee /mnt/petrelfs/niujunbo/zhengyuanhong/NativeResLLaVA/logs/NativeRes-LLaVA-Next-pt-1336.log'
    # --attn_implementation sdpa

# You can delete the sdpa attn_implementation if you want to use flash attn
