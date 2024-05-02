#!/bin/bash
export WANDB_PROJECT=llama-factory


CUDA_VISIBLE_DEVICES=2 python ../../src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset alpaca_gpt4_en \
    --dataset_dir ../../data \
    --template default \
    --optim adamw_hf\
    --finetuning_type full \
    --use_lisa \
    --lisa_activated_layers 2 \
    --lisa_interval_steps 3 \
    --output_dir ../../saves/LLaMA2-7B/badam/sft \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 15 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --warmup_steps 0 \
    --save_steps 100000000 \
    --learning_rate 1e-5 \
    --report_to wandb \
    --run_name lisa-llama-2-7b-2-3-batch8-acc15-1e-5 \
    --num_train_epochs 1 \
    --plot_loss \
    --bf16
