cd src/open-r1-multimodal

export DEBUG_MODE="true"
# export CUDA_VISIBLE_DEVICES=4,5,6,7

RUN_NAME="Qwen2.5-VL-7B-GRPO-SPA-Sub10k-Counting2k"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo_video.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path /mnt/moonfs/ouyangkun-m2/code/spatial_reasoning/VLM-R1/src/open-r1-multimodal/output/Qwen2.5-VL-7B-GRPO-SPA-Sub10k/checkpoint-9900 \
    --dataset_name data_config/spa.yaml \
    --image_root /mnt/moonfs/ouyangkun-m2/dataset/VSI_bench \
    --max_prompt_length 1024 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true