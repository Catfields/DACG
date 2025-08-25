#!/bin/bash

# 解决tmux下无法正常显示loss的脚本
# 通过设置环境变量来确保输出及时刷新

# 设置Python无缓冲输出
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# 设置tqdm在tmux下正常工作
export TQDM_DISABLE=0

# 运行训练脚本
accelerate launch --config_file accelerate_config_single_gpu.yaml train_with_accelerate.py \
    --image_dir data/mimic_cxr/images/ \
    --ann_path data/mimic_cxr/annotation_100k.json \
    --dataset_name mimic_cxr \
    --max_seq_length 60 \
    --batch_size 1 \
    --epochs 2 \
    --gradient_accumulation_steps 8 \
    --save_dir results/mimic_cxr_100k \
    --log_period 10 \
    --lr_ed 1e-5 \
    --lr_ve 5e-5 \
    --optim AdamW \
    --lr_scheduler StepLR \
    --step_size 50 \
    --gamma 0.1 \
    --record_dir records/ \
    --use_amp \
    --monitor_metric BLEU_4 \
    --monitor_mode max \
    --save_period 1