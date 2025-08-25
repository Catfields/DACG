#!/bin/bash
# Hugging Face Accelerate 一键启动脚本
# 支持多种训练配置，包括单GPU、多GPU、混合精度等

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印帮助信息
print_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --dataset DATASET    数据集名称 (iu_xray|mimic_cxr, default: iu_xray)"
    echo "  -g, --gpus NUM          GPU数量 (1|2|4, default: 1)"
    echo "  -b, --batch_size SIZE    批次大小 (default: 16 for iu_xray, 2 for mimic_cxr)"
    echo "  -e, --epochs NUM        训练轮数 (default: 100)"
    echo "  -a, --amp               启用混合精度训练"
    echo "  -c, --config FILE       使用自定义配置文件"
    echo "  -h, --help              显示帮助信息"
    echo ""
    echo "Examples:"
    echo "  $0 -d iu_xray -g 2 -a          # iu_xray数据集，2GPU，混合精度"
    echo "  $0 -d mimic_cxr -g 4 -e 50    # mimic_cxr数据集，4GPU，50轮"
    echo "  $0 -c accelerate_config.yaml  # 使用配置文件"
}

# 默认参数
DATASET="iu_xray"
GPUS=1
BATCH_SIZE=""
EPOCHS=100
USE_AMP=""
CONFIG=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -g|--gpus)
            GPUS="$2"
            shift 2
            ;;
        -b|--batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -a|--amp)
            USE_AMP="--use_amp"
            shift
            ;;
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            print_help
            exit 1
            ;;
    esac
done

# 设置数据集特定参数
case $DATASET in
    iu_xray)
        IMAGE_DIR="data/iu_xray/images/"
        ANN_PATH="data/iu_xray/annotation.json"
        MAX_SEQ_LENGTH=60
        THRESHOLD=3
        DEFAULT_BATCH_SIZE=16
        SAVE_DIR="results/iu_xray"
        STEP_SIZE=50
        GAMMA=0.1
        ;;
    mimic_cxr)
        IMAGE_DIR="data/mimic_cxr/images/"
        ANN_PATH="data/mimic_cxr/annotation.json"
        MAX_SEQ_LENGTH=100
        THRESHOLD=10
        DEFAULT_BATCH_SIZE=2
        SAVE_DIR="results/mimic_cxr"
        STEP_SIZE=1
        GAMMA=0.8
        ;;
    *)
        echo -e "${RED}错误: 不支持的数据集 '$DATASET'${NC}"
        echo "支持的数据集: iu_xray, mimic_cxr"
        exit 1
        ;;
esac

# 设置批次大小
if [[ -z "$BATCH_SIZE" ]]; then
    BATCH_SIZE=$DEFAULT_BATCH_SIZE
fi

# 检查accelerate是否安装
if ! command -v accelerate &> /dev/null; then
    echo -e "${RED}错误: accelerate 未安装${NC}"
    echo "请运行: pip install accelerate"
    exit 1
fi

# 检查GPU可用性
if [[ $GPUS -gt 1 ]]; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [[ $GPUS -gt $GPU_COUNT ]]; then
        echo -e "${YELLOW}警告: 请求的GPU数量 ($GPUS) 超过可用数量 ($GPU_COUNT)${NC}"
        GPUS=$GPU_COUNT
    fi
fi

# 构建命令
if [[ -n "$CONFIG" ]]; then
    CMD="accelerate launch --config_file $CONFIG"
else
    CMD="accelerate launch --num_processes=$GPUS"
    if [[ -n "$USE_AMP" ]]; then
        CMD="$CMD --mixed_precision fp16"
    fi
fi

# 添加训练参数
CMD="$CMD train_with_accelerate.py \
--image_dir $IMAGE_DIR \
--ann_path $ANN_PATH \
--dataset_name $DATASET \
--max_seq_length $MAX_SEQ_LENGTH \
--threshold $THRESHOLD \
--batch_size $BATCH_SIZE \
--epochs $EPOCHS \
--save_dir $SAVE_DIR \
--step_size $STEP_SIZE \
--gamma $GAMMA \
--early_stop 50 \
--seed 9233"

# 添加混合精度参数
if [[ -n "$USE_AMP" ]]; then
    CMD="$CMD $USE_AMP"
fi

# 打印配置信息
echo -e "${GREEN}=== 训练配置 ===${NC}"
echo "数据集: $DATASET"
echo "GPU数量: $GPUS"
echo "批次大小: $BATCH_SIZE"
echo "训练轮数: $EPOCHS"
echo "混合精度: $([[ -n "$USE_AMP" ]] && echo "启用" || echo "禁用")"
echo "命令: $CMD"
echo -e "${GREEN}================${NC}"

# 执行训练
echo -e "${GREEN}开始训练...${NC}"
eval $CMD

echo -e "${GREEN}训练完成！${NC}"