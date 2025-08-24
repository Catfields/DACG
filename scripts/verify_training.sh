#!/bin/bash

# 验证小样本量训练流程的脚本
# 设置Python无缓冲输出
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# 设置tqdm在tmux下正常工作
export TQDM_DISABLE=0

# 检查是否启用小样本模式
SAMPLE_MODE=false
SAMPLE_COUNT=100
while [[ "$1" != "" ]]; do
    case $1 in
        --sample )
            shift
            SAMPLE_MODE=true
            SAMPLE_COUNT=$1
            ;;
    esac
    shift
done

# 如果启用小样本模式，生成小样本数据集
if [ "$SAMPLE_MODE" = true ]; then
    echo "启用小样本模式，抽取 $SAMPLE_COUNT 样本..."
    python scripts/extract_small_dataset.py --count $SAMPLE_COUNT
    ANN_PATH="data/mimic_cxr/small_annotation.json"
    SAVE_DIR="results/test_${SAMPLE_COUNT}_samples"
else
    ANN_PATH="data/mimic_cxr/annotation.json"
    SAVE_DIR="results/full_training"
fi

# 运行训练脚本
accelerate launch --config_file accelerate_config_single_gpu.yaml train_with_accelerate.py \
    --image_dir data/mimic_cxr/images/ \
    --ann_path $ANN_PATH \
    --dataset_name mimic_cxr \
    --max_seq_length 60 \
    --batch_size 1 \
    --epochs 1 \
    --gradient_accumulation_steps 1 \
    --save_dir $SAVE_DIR \
    --log_period 1 \
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