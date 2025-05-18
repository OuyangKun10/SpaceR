cd src/r1-v

export DEBUG_MODE="false" 
export LOG_PATH="./debug_log_SpaceR_sft.txt"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
    src/open_r1/sft.py \
    --output_dir "/Output/SpaceR-SFT" \
    --model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --dataset_name "SpaceR-151k.jsonl" \
    --deepspeed local_scripts/zero2.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name SpaceR-SFT \
    --save_steps 1000 \
    --max_grad_norm 5 \
    --save_only_model true \