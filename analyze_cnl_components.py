#!/usr/bin/env python3
"""
详细分析CNL(上下文引导归一化层)的具体构成和参数分布
"""

import torch
import torch.nn as nn
from modules.Context_guidance_normalization import CNL

def analyze_cnl_components(hidden_dim=512, H=49):
    """分析CNL层的具体构成"""
    
    print("=" * 80)
    print("CNL(上下文引导归一化层)详细构成分析")
    print("=" * 80)
    
    # 创建CNL实例
    cnl = CNL(hidden_dim=hidden_dim, H=H)
    
    print(f"配置参数:")
    print(f"  隐藏维度 (hidden_dim): {hidden_dim}")
    print(f"  GM行数 (H): {H}")
    print(f"  MLP扩展比例: 4")
    print(f"  Dropout: 0.1")
    
    print("\n" + "=" * 80)
    print("CNL架构流程")
    print("=" * 80)
    
    print("""
    输入:
    ├── GM_t: (H, D) - 引导记忆矩阵
    └── prev_output: (B, S, D) - 前一模块输出
    
    处理流程:
    1. GM_t 展开 → gm_t: (H×D,)
    2. MLP处理 → Δγ, Δβ: (2×D,)
    3. 参数更新 → γ_t*, β_t*: (D,)
    4. LayerNorm → normalized: (B, S, D)
    5. 仿射变换 → 输出: (B, S, D)
    """)
    
    print("\n" + "=" * 80)
    print("参数分布分析")
    print("=" * 80)
    
    total_params = 0
    
    # 1. 上下文投影层
    if hasattr(cnl, 'context_proj') and cnl.context_proj is not None:
        context_proj_params = sum(p.numel() for p in cnl.context_proj.parameters())
        print(f"1. 上下文投影层:")
        print(f"   Linear({cnl.context_dim} → {hidden_dim})")
        print(f"   参数量: {context_proj_params:,}")
        total_params += context_proj_params
    
    # 2. 可学习参数 γ 和 β
    gamma_beta_params = hidden_dim * 2  # γ 和 β 各 hidden_dim 个参数
    print(f"2. 基础可学习参数:")
    print(f"   γ: {hidden_dim} 参数")
    print(f"   β: {hidden_dim} 参数")
    print(f"   小计: {gamma_beta_params:,}")
    total_params += gamma_beta_params
    
    # 3. MLP网络
    mlp_params = 0
    print(f"3. MLP网络:")
    
    # 第一层: Linear(H×D → 4×D)
    layer1_params = (hidden_dim * H) * (hidden_dim * 4) + (hidden_dim * 4)
    print(f"   第一层 Linear({hidden_dim*H} → {hidden_dim*4})")
    print(f"   参数量: {layer1_params:,}")
    mlp_params += layer1_params
    
    # 第二层: Linear(4×D → 2×D)
    layer2_params = (hidden_dim * 4) * (hidden_dim * 2) + (hidden_dim * 2)
    print(f"   第二层 Linear({hidden_dim*4} → {hidden_dim*2})")
    print(f"   参数量: {layer2_params:,}")
    mlp_params += layer2_params
    
    print(f"   MLP总计: {mlp_params:,}")
    total_params += mlp_params
    
    # 4. LayerNorm (无参数，因为elementwise_affine=False)
    layer_norm_params = 0
    print(f"4. LayerNorm: {layer_norm_params} 参数 (无仿射变换)")
    
    print("\n" + "=" * 80)
    print("总参数量计算")
    print("=" * 80)
    
    print(f"总参数量: {total_params:,}")
    
    # 计算各组件占比
    components = [
        ("上下文投影", context_proj_params if hasattr(cnl, 'context_proj') and cnl.context_proj else 0),
        ("基础参数(γ,β)", gamma_beta_params),
        ("MLP网络", mlp_params)
    ]
    
    print("\n组件占比:")
    for name, params in components:
        if params > 0:
            percentage = (params / total_params) * 100
            print(f"  {name}: {params:,} 参数 ({percentage:.1f}%)")
    
    print("\n" + "=" * 80)
    print("计算复杂度分析")
    print("=" * 80)
    
    # 计算不同维度的参数
    print("不同维度的参数分布:")
    
    # MLP输入维度影响
    mlp_input_dim = hidden_dim * H
    mlp_hidden_dim = hidden_dim * 4
    mlp_output_dim = hidden_dim * 2
    
    print(f"  MLP输入维度: {mlp_input_dim:,}")
    print(f"  MLP隐藏维度: {mlp_hidden_dim:,}")
    print(f"  MLP输出维度: {mlp_output_dim:,}")
    
    # 关键乘法运算
    key_multiplications = [
        ("MLP第一层权重", mlp_input_dim * mlp_hidden_dim),
        ("MLP第二层权重", mlp_hidden_dim * mlp_output_dim),
        ("总权重参数", mlp_input_dim * mlp_hidden_dim + mlp_hidden_dim * mlp_output_dim)
    ]
    
    print(f"\n  关键乘法运算:")
    for name, value in key_multiplications:
        print(f"    {name}: {value:,}")
    
    print("\n" + "=" * 80)
    print("实际参数验证")
    print("=" * 80)
    
    # 实际计算模型参数
    actual_params = sum(p.numel() for p in cnl.parameters())
    print(f"实际模型参数: {actual_params:,}")
    print(f"计算参数: {total_params:,}")
    print(f"匹配: {'✓' if actual_params == total_params else '✗'}")
    
    return cnl, total_params

def analyze_memory_usage(hidden_dim=512, H=49, batch_size=32, seq_len=100):
    """分析内存使用情况"""
    print("\n" + "=" * 80)
    print("内存使用分析")
    print("=" * 80)
    
    # 输入张量大小计算
    gm_t_size = hidden_dim * H  # GM_t: (H, D)
    prev_output_size = batch_size * seq_len * hidden_dim  # prev_output: (B, S, D)
    
    # 中间激活值大小
    gm_t_flat_size = hidden_dim * H  # gm_t: (H×D,)
    mlp_output_size = hidden_dim * 2  # mlp_output: (2×D,)
    normalized_size = batch_size * seq_len * hidden_dim  # normalized: (B, S, D)
    
    # 以float32计算内存 (4 bytes per float)
    memory_mb = {
        "GM_t": gm_t_size * 4 / (1024**2),
        "prev_output": prev_output_size * 4 / (1024**2),
        "gm_t_flat": gm_t_flat_size * 4 / (1024**2),
        "mlp_output": mlp_output_size * 4 / (1024**2),
        "normalized": normalized_size * 4 / (1024**2),
        "total_activation": (gm_t_flat_size + mlp_output_size + normalized_size) * 4 / (1024**2)
    }
    
    print(f"配置: batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}, H={H}")
    print("\n内存使用 (MB):")
    for name, size_mb in memory_mb.items():
        print(f"  {name}: {size_mb:.2f} MB")
    
    return memory_mb

if __name__ == "__main__":
    # 分析不同配置
    configs = [
        {"hidden_dim": 512, "H": 49, "name": "标准配置"},
        {"hidden_dim": 768, "H": 64, "name": "大模型配置"},
        {"hidden_dim": 256, "H": 36, "name": "小模型配置"}
    ]
    
    for config in configs:
        print(f"\n{'='*100}")
        print(f"分析配置: {config['name']}")
        print('='*100)
        
        cnl, total_params = analyze_cnl_components(
            hidden_dim=config['hidden_dim'], 
            H=config['H']
        )
        
        analyze_memory_usage(
            hidden_dim=config['hidden_dim'], 
            H=config['H']
        )

    # 分析当前项目实际配置
    print(f"\n{'='*100}")
    print("当前项目实际配置分析")
    print('='*100)
    analyze_cnl_components(hidden_dim=512, H=49)