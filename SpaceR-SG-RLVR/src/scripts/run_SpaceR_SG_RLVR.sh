cd src/r1-v

export DEBUG_MODE="false" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_SpaceR.txt"

# For resume training:  --resume_from_checkpoint Model_Path \
# Set temporal to flase to speed training 

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12365" \
    src/open_r1/SG-RLVR.py  \
    --output_dir "/Output/SpaceR-SG-RLVR" \
    --model_name_or_path 'Qwen/Qwen2.5-VL-7B-Instruct' \
    --dataset_name "SpaceR-151k.jsonl" \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 16384 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --temporal true \
    --len_control true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name SpaceR-SG-RLVR \
    --save_steps 1000 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model true \
    --num_generations 8  # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
