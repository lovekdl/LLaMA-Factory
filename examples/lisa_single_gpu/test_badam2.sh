export WANDB_PROJECT=llama-factory

CUDA_VISIBLE_DEVICES=0 python ../../src/train_bash.py \
    --stage sft \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --do_train \
    --dataset alpaca_gpt4_en \
    --dataset_dir ../../data \
    --template default \
    --finetuning_type full \
    --output_dir ../../saves/LLaMA2-7B/badam/badam-llama-2-7b-asc-switch50-batch8-acc15-epoch7-LR1e-5 \
    --run_name badam-llama-2-7b-asc-switch50-batch8-acc15-epoch7-LR1e-5\
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 15 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 10000 \
    --learning_rate 1e-5 \
    --num_train_epochs 7 \
    --overwrite_output_dir \
    --plot_loss \
    --use_badam \
    --badam_switch_mode ascending \
    --badam_switch_block_every 50 \
    --bf16