#!/bin/bash
# 使用Hugging Face Accelerate框架启动训练
# 使用方法：
#   1. 安装accelerate: pip install accelerate
#   2. 配置accelerate: accelerate config
#   3. 运行脚本: bash run_mimic_cxr.sh

# 通用参数
COMMON_ARGS="--image_dir data/mimic_cxr/images/ \
--ann_path data/mimic_cxr/annotation.json \
--dataset_name mimic_cxr \
--max_seq_length 100 \
--threshold 10 \
--gradient_accumulation_steps 8 \
--save_dir results/mimic_cxr \
--step_size 1 \
--gamma 0.8 \
--seed 456789 \
--use_amp"

# 改进: 使用更大的batch_size和更多的训练轮数
echo "启动训练 - 使用更大的batch_size和更多的训练轮数"
accelerate launch --config_file accelerate_config_single_gpu.yaml train_with_accelerate.py \
$COMMON_ARGS \
--batch_size 2 \
--epochs 10 \
--early_stop 20