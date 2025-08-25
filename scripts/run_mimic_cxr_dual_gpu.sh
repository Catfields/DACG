#!/bin/bash
# 使用Hugging Face Accelerate框架启动双GPU训练
# 使用方法：
#   1. 确保已安装accelerate: pip install accelerate
#   2. 运行脚本: bash run_mimic_cxr_dual_gpu.sh

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}启动DACG双GPU训练...${NC}"

# 检查GPU数量
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo -e "${YELLOW}检测到GPU数量: $GPU_COUNT${NC}"

if [ $GPU_COUNT -lt 2 ]; then
    echo -e "${RED}警告: 检测到少于2个GPU，可能无法正常运行双GPU训练${NC}"
    read -p "是否继续? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查CUDA可用性
if ! python -c "import torch; print('CUDA可用:', torch.cuda.is_available())" 2>/dev/null; then
    echo -e "${RED}错误: CUDA不可用，请检查CUDA安装${NC}"
    exit 1
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1

# 训练参数配置
IMAGE_DIR="data/mimic_cxr/images/"
ANN_PATH="data/mimic_cxr/annotation.json"
DATASET_NAME="mimic_cxr"
BATCH_SIZE_PER_GPU=4
GRADIENT_ACCUMULATION_STEPS=4
TOTAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * 2 * GRADIENT_ACCUMULATION_STEPS))

# 显示配置信息
echo -e "${GREEN}训练配置:${NC}"
echo "数据集: $DATASET_NAME"
echo "单GPU batch_size: $BATCH_SIZE_PER_GPU"
echo "GPU数量: 2"
echo "梯度累积步数: $GRADIENT_ACCUMULATION_STEPS"
echo "实际全局batch_size: $TOTAL_BATCH_SIZE"
echo "保存路径: results/mimic_cxr_dual"

# 创建保存目录
mkdir -p results/mimic_cxr_dual

# 启动训练
echo -e "${GREEN}开始训练...${NC}"
accelerate launch --num_processes 2 train_with_accelerate.py \
--image_dir "$IMAGE_DIR" \
--ann_path "$ANN_PATH" \
--dataset_name "$DATASET_NAME" \
--max_seq_length 100 \
--threshold 10 \
--batch_size $BATCH_SIZE_PER_GPU \
--gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
--epochs 30 \
--save_dir results/mimic_cxr_dual \
--step_size 1 \
--gamma 0.8 \
--early_stop 5 \
--seed 456789 \
--use_amp

# 检查训练结果
if [ $? -eq 0 ]; then
    echo -e "${GREEN}训练成功完成！${NC}"
    echo "模型保存在: results/mimic_cxr_dual/"
else
    echo -e "${RED}训练过程中出现错误，请检查日志${NC}"
    exit 1
fi