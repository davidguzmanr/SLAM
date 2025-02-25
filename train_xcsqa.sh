export HF_ENDPOINT=https://hf-mirror.com
# export CUDA_LAUNCH_BLOCKING=1
##################################################################
# 训练MetaMath的时候记得改tokenizer！！！！
##################################################################
OUTPUT_DIR=/Path/to/save/models
mkdir -p $OUTPUT_DIR

deepspeed --include localhost:0,1,2,3 --master_port=9905 train_bash.py \
    --deepspeed ./ds_config.json \
    --stage sft \
    --do_train \
    --cutoff_len 512 \
    --model_name_or_path ./xcsqa-en-train-llama2-7B \
    --dataset flores200_xcsqa_combined \
    --template default \
    --finetuning_type full \
    --special_train_layers 0 1 2 3 \
    --only_mlp \
    --output_dir $OUTPUT_DIR \
    --overwrite_cache \
    --gradient_checkpointing 1 \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 50 \
    --learning_rate 2e-5 \
    --save_only_model \
    --save_total_limit 6 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --overwrite_output_dir \
    --bf16