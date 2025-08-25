#!/usr/bin/env python3
"""
计算DACG模型的参数量
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.dacg import DACGModel
from modules.tokenizers import Tokenizer
import argparse

def count_parameters(model):
    """计算模型的总参数量和可训练参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def main():
    # 设置参数（从train_with_accelerate.py复制）
    parser = argparse.ArgumentParser()
    
    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/mimic_cxr/images/')
    parser.add_argument('--ann_path', type=str, default='data/mimic_cxr/annotation.json')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr', choices=['iu_xray', 'mimic_cxr'])
    parser.add_argument('--max_seq_length', type=int, default=100)
    parser.add_argument('--threshold', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=16)

    # Model settings
    parser.add_argument('--visual_extractor', type=str, default='resnet101')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--d_vf', type=int, default=2048)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--rm_num_slots', type=int, default=3)
    parser.add_argument('--rm_num_heads', type=int, default=8)
    parser.add_argument('--rm_d_model', type=int, default=512)
    
    args = parser.parse_args()
    
    # 初始化tokenizer
    tokenizer = Tokenizer(args)
    
    # 初始化模型
    model = DACGModel(args, tokenizer)
    
    # 计算参数量
    total_params, trainable_params = count_parameters(model)
    
    # 打印结果
    print("=" * 50)
    print("DACG模型参数量统计")
    print("=" * 50)
    print(f"数据集: {args.dataset_name}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"冻结参数量: {total_params - trainable_params:,}")
    print(f"可训练参数比例: {(trainable_params/total_params)*100:.2f}%")
    
    # 详细的参数统计
    print("\n" + "=" * 50)
    print("各组件参数量:")
    print("=" * 50)
    
    # Visual Extractor参数
    visual_params = sum(p.numel() for p in model.visual_extractor.parameters())
    visual_trainable = sum(p.numel() for p in model.visual_extractor.parameters() if p.requires_grad)
    print(f"视觉提取器 ({args.visual_extractor}):")
    print(f"  总参数: {visual_params:,}")
    print(f"  可训练: {visual_trainable:,}")
    
    # Encoder-Decoder参数
    encoder_decoder_params = sum(p.numel() for p in model.encoder_decoder.parameters())
    encoder_decoder_trainable = sum(p.numel() for p in model.encoder_decoder.parameters() if p.requires_grad)
    print(f"编码器-解码器:")
    print(f"  总参数: {encoder_decoder_params:,}")
    print(f"  可训练: {encoder_decoder_trainable:,}")
    
    # 计算内存占用估计
    param_size = total_params * 4  # 假设float32，每个参数4字节
    param_memory = param_size / (1024 ** 2)  # 转换为MB
    
    # 考虑优化器状态（通常参数量*2）
    optimizer_memory = param_size * 2 / (1024 ** 2)
    
    print("\n" + "=" * 50)
    print("内存占用估计:")
    print("=" * 50)
    print(f"模型参数内存: {param_memory:.2f} MB")
    print(f"优化器状态内存: {optimizer_memory:.2f} MB")
    print(f"总内存需求: {param_memory + optimizer_memory:.2f} MB")
    
    print("\n" + "=" * 50)
    print("模型结构:")
    print("=" * 50)
    print(model)

if __name__ == '__main__':
    main()