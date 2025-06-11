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
export NCCL_DEBUG=''

export LLM_VERSION="/mnt/hwfile/mllm/niujunbo/model-image/lmsys/vicuna-7b-v1.5"
export LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
export VISION_MODEL_VERSION="/mnt/hwfile/mllm/niujunbo/model-image/Qwen/Qwen2-VL-2B-Instruct"
export VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

export PARTITION="mineru2"
export NODES=1
export CPUS=128
export MASTER_PORT=12349
export MID_RUN_NAME="llava-next-vicuna-7b-v1.5-qwenvit-4_4096_visual_token-ft-v1"
export PROMPT_VERSION="v1"

export BASE_RUN_NAME="nativeresllava-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

export CKPT_PATH=$LLM_VERSION # this could also be the previous stage checkpoint

srun -J debug \
    -p $PARTITION \
    --nodes=$NODES \
    --ntasks-per-node=1 \
    --gres=gpu:8 \
    --cpus-per-task=$CPUS \
    bash -c 'ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node 8 --nnodes ${NODES} --node_rank ${SLURM_NODEID} --master_addr $(scontrol show hostname $SLURM_NODELIST | head -n1) --master_port ${MASTER_PORT} llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${CKPT_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path=/mnt/hwfile/doc_parse/niujunbo/llava/LLaVA-Instruct-150K/llava_v1_5_mix665k.json \
    --image_folder /mnt/hwfile/mllm/qianrui/llava \
    --pretrain_mm_mlp_adapter="/mnt/hwfile/doc_parse/niujunbo/llava/checkpoints/projectors/${BASE_RUN_NAME}/mm_projector.bin" \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_min_image_token 4\
    --mm_max_image_token 4096\
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir "/mnt/hwfile/doc_parse/niujunbo/llava/checkpoints/${MID_RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 5120 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    2>&1 | tee /mnt/petrelfs/niujunbo/zhengyuanhong/NativeResLLaVA/logs/${MID_RUN_NAME}.log'

# You can delete the sdpa attn_implementation if you want to use flash attn
