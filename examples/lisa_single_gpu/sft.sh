#!/bin/bash
export WANDB_PROJECT=lisa-project

# hyperparameters
num_train_epochs=4
lr=1e-5
gradient_accumulation_steps=1
batch_size=16
# Lisa hyperparameters
lisa_activated_layers=1
lisa_interval_steps=30
lisa_order=min_grad

exp_name=lisa-llama-2-7b-${lisa_activated_layers}layers-${lisa_interval_steps}interval-${lisa_order}-middle-batch${batch_size}-acc${gradient_accumulation_steps}-epoch${num_train_epochs}-${lr}

CUDA_VISIBLE_DEVICES=3 python ../../src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset alpaca_gpt4_en \
    --dataset_dir ../../data \
    --template default \
    --finetuning_type full \
    --use_lisa \
    --lisa_activated_layers ${lisa_activated_layers} \
    --lisa_interval_steps ${lisa_interval_steps} \
    --including_embed_and_lm_head false \
    --lisa_order ${lisa_order} \
    --output_dir ../../saves/LLaMA2-7B/lisa/${exp_name} \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --logging_steps 1 \
    --warmup_steps 0 \
    --save_steps 100000000 \
    --learning_rate ${lr} \
    --report_to wandb \
    --run_name ${exp_name} \
    --num_train_epochs ${num_train_epochs} \
    --plot_loss \
    --bf16
