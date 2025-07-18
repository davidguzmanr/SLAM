#!/bin/bash
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --account=def-annielee
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@alumni.utoronto.ca
#SBATCH --output=slam-afrimgsm-metamath-mistral-7b_%j.out
#SBATCH --error=slam-afrimgsm-metamath-mistral-7b_%j.out

#############################################################
# install the environment by loading in python and required packages
module load python/3.10.13
module load gcc/12.3
module load cuda/12.2
module load arrow/17.0.0 

source ~/SLAM-env/bin/activate
export HF_HOME=~/scratch/huggingface
#############################################################

# Redirect all output and errors from this point forward
exec > slam-afrimgsm-metamath-mistral-7b_${SLURM_JOB_ID}.out 2>&1

echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo $HF_HOME

cd /home/davidguz/scratch/AfricanLLM/SLAM/src

echo "Training"
deepspeed train_bash.py \
    --deepspeed ds_config.json \
    --stage sft \
    --do_train \
    --cutoff_len 1024 \
    --model_name_or_path meta-math/MetaMath-Mistral-7B \
    --dataset slam_afrimgsm_query_translation \
    --template default \
    --finetuning_type full \
    --special_train_layers 0 1 2 3 4\
    --only_mlp \
    --output_dir output-afrimgsm-MetaMath-Mistral-7B \
    --overwrite_cache \
    --gradient_checkpointing 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --preprocessing_num_workers 8 \
    --lr_scheduler_type cosine \
    --logging_steps 100 \
    --save_steps 20000 \
    --learning_rate 2e-5 \
    --save_only_model \
    --save_total_limit 2 \
    --num_train_epochs 4.0 \
    --plot_loss \
    --overwrite_output_dir \
    --bf16 \
    --report_to none

echo "Evaluating"
lm_eval \
--model hf \
--model_args pretrained="checkpoint-76500" \
--tasks afrimgsm_amh_prompt_1,afrimgsm_eng_prompt_1,afrimgsm_ewe_prompt_1,afrimgsm_fra_prompt_1,afrimgsm_hau_prompt_1,afrimgsm_ibo_prompt_1,afrimgsm_kin_prompt_1,afrimgsm_lin_prompt_1,afrimgsm_lug_prompt_1,afrimgsm_orm_prompt_1,afrimgsm_sna_prompt_1,afrimgsm_sot_prompt_1,afrimgsm_swa_prompt_1,afrimgsm_twi_prompt_1,afrimgsm_vai_prompt_1,afrimgsm_wol_prompt_1,afrimgsm_xho_prompt_1,afrimgsm_yor_prompt_1,afrimgsm_zul_prompt_1 \
--device cuda:0 \
--batch_size 8 \
--verbosity DEBUG \
--output_path AfriMGSM-MetaMath-Mistral-7B.json

# In case something goes wrong, we can sleep for a long time to keep the job alive and still use the GPU via srun
echo "Sleeping"
sleep 24h