#!/usr/bin/env python3
"""
分析EncoderDecoder中各组件的参数量分布
"""

import torch
import torch.nn as nn
from modules.dacg import EncoderDecoder
from modules.tokenizers import Tokenizer
import argparse

def count_parameters(model, name=""):
    """计算模型参数"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if name:
        print(f"{name}:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    print(f"  冻结参数量: {total_params - trainable_params:,}")
    return total_params, trainable_params

def analyze_encoder_decoder_components(args):
    """分析EncoderDecoder各组件的参数量"""
    
    # 创建tokenizer
    tokenizer = Tokenizer(args)
    
    # 创建完整模型
    model = EncoderDecoder(args, tokenizer)
    
    print("=" * 60)
    print("EncoderDecoder 组件参数量分析")
    print("=" * 60)
    
    # 分析完整模型
    total_params, trainable_params = count_parameters(model, "完整 EncoderDecoder 模型")
    
    print("\n" + "=" * 60)
    print("各组件详细分析")
    print("=" * 60)
    
    # 分析视觉提取器
    if hasattr(model, 'visual_extractor'):
        visual_params, visual_trainable = count_parameters(model.visual_extractor, "视觉提取器 (Visual Extractor)")
    else:
        print("视觉提取器: 未找到独立模块")
        visual_params = 0
        visual_trainable = 0
    
    # 分析编码器
    encoder_params, encoder_trainable = count_parameters(model.encoder, "编码器 (Encoder)")
    
    # 分析解码器
    decoder_params, decoder_trainable = count_parameters(model.decoder, "解码器 (Decoder)")
    
    # 分析GMG (引导记忆生成器)
    if hasattr(model, 'gmg'):
        gmg_params, gmg_trainable = count_parameters(model.gmg, "引导记忆生成器 (GMG)")
    else:
        print("引导记忆生成器: 未找到独立模块")
        gmg_params = 0
        gmg_trainable = 0
    
    # 详细分析编码器内部
    print("\n" + "=" * 60)
    print("编码器内部组件分析")
    print("=" * 60)
    
    # 编码器的主要组件
    encoder_components = {
        'dual_attention': model.encoder.dual_attention,
        'input_projection': model.encoder.input_projection,
        'encoder_layers': model.encoder.layers
    }
    
    for name, component in encoder_components.items():
        if hasattr(component, '__len__'):  # 是ModuleList
            layer_params = sum(p.numel() for layer in component for p in layer.parameters())
            layer_trainable = sum(p.numel() for layer in component for p in layer.parameters() if p.requires_grad)
            print(f"{name}: {layer_params:,} 参数")
        else:
            params, trainable = count_parameters(component, f"  {name}")
    
    # 详细分析解码器内部
    print("\n" + "=" * 60)
    print("解码器内部组件分析")
    print("=" * 60)
    
    decoder_components = {
        'embedding': model.decoder.embedding,
        'pos_encoding': model.decoder.pos_encoding,
        'decoder_layers': model.decoder.layers
    }
    
    for name, component in decoder_components.items():
        if name == 'decoder_layers':
            # 分析每一层
            for i, layer in enumerate(component):
                layer_params, layer_trainable = count_parameters(layer, f"  解码器层 {i+1}")
                
                # 分析解码器层的子组件
                if hasattr(layer, 'masked_mha'):
                    count_parameters(layer.masked_mha, f"    掩码多头注意力")
                if hasattr(layer, 'cross_mha'):
                    count_parameters(layer.cross_mha, f"    交叉多头注意力")
                if hasattr(layer, 'feed_forward'):
                    count_parameters(layer.feed_forward, f"    前馈网络")
                if hasattr(layer, 'cnl_1'):
                    count_parameters(layer.cnl_1, f"    上下文归一化1")
                if hasattr(layer, 'cnl_2'):
                    count_parameters(layer.cnl_2, f"    上下文归一化2")
                if hasattr(layer, 'cnl_3'):
                    count_parameters(layer.cnl_3, f"    上下文归一化3")
                if hasattr(layer, 'fc_out') and layer.fc_out is not None:
                    count_parameters(layer.fc_out, f"    输出全连接层")
        else:
            params, trainable = count_parameters(component, f"  {name}")
    
    # 计算各组件占比
    print("\n" + "=" * 60)
    print("参数量占比分析")
    print("=" * 60)
    
    components = [
        ("视觉提取器", visual_params),
        ("编码器", encoder_params),
        ("解码器", decoder_params),
        ("引导记忆生成器", gmg_params)
    ]
    
    for name, params in components:
        if params > 0:
            percentage = (params / total_params) * 100
            print(f"{name}: {params:,} 参数 ({percentage:.1f}%)")
    
    # 找出最大组件
    max_component = max([(name, params) for name, params in components if params > 0], 
                       key=lambda x: x[1])
    
    print(f"\n📊 最大可训练参数组件: {max_component[0]}")
    print(f"📈 参数量: {max_component[1]:,} ({(max_component[1]/total_params)*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze EncoderDecoder parameters')
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr', 
                       help='dataset name')
    parser.add_argument('--image_dir', type=str, default='data/mimic_cxr/images', 
                       help='directory of images')
    parser.add_argument('--ann_path', type=str, default='data/mimic_cxr/annotation.json', 
                       help='path to annotation file')
    parser.add_argument('--max_seq_length', type=int, default=100, 
                       help='maximum sequence length')
    parser.add_argument('--threshold', type=int, default=10, 
                       help='threshold for words')
    parser.add_argument('--num_workers', type=int, default=2, 
                       help='number of workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=16, 
                       help='batch size')
    parser.add_argument('--d_vf', type=int, default=2048, 
                       help='dimension of visual features')
    parser.add_argument('--num_layers', type=int, default=1, 
                       help='number of layers')
    parser.add_argument('--rm_num_slots', type=int, default=3, 
                       help='read memory num slots')
    parser.add_argument('--rm_num_heads', type=int, default=8, 
                       help='read memory num heads')
    parser.add_argument('--rm_d_model', type=int, default=512, 
                       help='read memory d model')
    
    args = parser.parse_args()
    analyze_encoder_decoder_components(args)