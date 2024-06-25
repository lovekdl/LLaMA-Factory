#!/bin/bash
export WANDB_PROJECT=llama-factory

CUDA_VISIBLE_DEVICES=1 python ../../src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.3 \
    --dataset alpaca_gpt4_en \
    --dataset_dir ../../data \
    --template default \
    --finetuning_type full \
    --use_lisa \
    --lisa_activated_layers 3 \
    --lisa_interval_steps 30 \
    --including_embed_and_lm_head true\
    --lisa_order ascending \
    --output_dir ../../saves/Mistral-7B-Instruct-v0.3/lisa/lisa-mistral-7b-3-50-random-middle-batch16-acc1-epoch2-1e-5 \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --warmup_steps 0 \
    --save_steps 100000000 \
    --learning_rate 1e-5 \
    --report_to wandb \
    --run_name lisa-llama-2-7b-3-50-random-middle-batch16-acc1-epoch2-1e-5 \
    --num_train_epochs 3 \
    --plot_loss \
    --bf16