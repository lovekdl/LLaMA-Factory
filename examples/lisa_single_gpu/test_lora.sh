#!/bin/bash
export WANDB_PROJECT=llama-factory


CUDA_VISIBLE_DEVICES=0 python ../../src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset alpaca_gpt4_en \
    --dataset_dir ../../data \
    --template default \
    --finetuning_type lora \
    --lora_rank 100 \
    --output_dir ../../saves/LLaMA2-7B/lora/sft \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 15 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --warmup_steps 0 \
    --save_steps 1000000000 \
    --report_to wandb \
    --run_name lora-llama-2-7b-r100-batch8-acc15-epoch7-1e-5 \
    --learning_rate 1e-5 \
    --num_train_epochs 7 \
    --plot_loss \
    --fp16
