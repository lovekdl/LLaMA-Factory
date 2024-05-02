#!/bin/bash
export WANDB_PROJECT=llama-factory
CUDA_VISIBLE_DEVICES=3 python ../../src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset alpaca_gpt4_en \
    --dataset_dir ../../data \
    --template default \
    --finetuning_type full \
    --use_badam \
    --badam_switch_mode ascending \
    --badam_switch_block_every 1 \
    --badam_verbose 2 \
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
    --save_steps 10000000000 \
    --learning_rate 1e-5 \
    --report_to wandb \
    --run_name badam-llama-2-7b-switch10-asc-batch8-acc15-epoch7-1e-5 \
    --num_train_epochs 7 \
    --plot_loss \
    --bf16
