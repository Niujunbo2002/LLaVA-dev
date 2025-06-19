gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}
echo ${gpu_list}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llava/eval/eval_omnidocbench.py \
    --model-path /mnt/petrelfs/niujunbo/zhengyuanhong/NativeResLLaVA_mineru/nativeres-llava-_mnt_hwfile_mllm_niujunbo_model-image_Qwen_Qwen2-VL-2B-Instruct-_mnt_hwfile_mllm_niujunbo_model-image_Qwen_Qwen2-0.5B-Instruct-stage2-v3_power_data-4096 \
    --data_dir /mnt/hwfile/opendatalab/bigdata_mineru/ouyanglinke/PDF_Formula/Docparse/news_notes_200dpi_masked \
    --output_dir /mnt/petrelfs/niujunbo/doc_parse/nativeres_powerdata/4000-4096 \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX  \
    --conv-mode qwen_1_5 &
done

wait