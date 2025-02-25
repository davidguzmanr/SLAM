export HF_ENDPOINT=https://hf-mirror.com

OUTPUT_DIR=/Path/to/save/models

deepspeed --include localhost:4,5,6,7 --master_port=9905 train_bash.py \
    --deepspeed ./ds_config.json \
    --stage sft \
    --do_train \
    --cutoff_len 1024 \
    --model_name_or_path /Path/to/MetaMath-7B-V1.0 \
    --dataset mgsm_trans_question_answer2 \
    --template default \
    --finetuning_type full \
    --special_train_layers 0 1 2 3 4\
    --only_mlp \
    --output_dir $OUTPUT_DIR \
    --overwrite_cache \
    --gradient_checkpointing 1 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --preprocessing_num_workers 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 25 \
    --learning_rate 2e-5 \
    --save_only_model \
    --save_total_limit 16 \
    --num_train_epochs 4.0 \
    --plot_loss \
    --overwrite_output_dir \
    --bf16
