export CUDA_VISIBLE_DEVICES=0
python eval2statistic.py \
        --model_path /localnvme/application/sc_new/fmm/LLaMA-Factory-backup/8.7/xcsqa-mtrain-with-flores-llama2-7B-layers0-3-3epoch-step50 \
        --streategy Parallel \
        --batch_size 64