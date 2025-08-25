#!/usr/bin/env python3
"""
快速优化训练集规模 - 带进度条版本
"""

import json
import random
import time
from collections import defaultdict
from tqdm import tqdm

def create_optimized_dataset():
    """创建优化的10万样本数据集，带进度条显示"""
    
    print("🚀 开始优化训练集...")
    
    # 加载数据
    print("📊 加载原始数据集...")
    with open('data/mimic_cxr/annotation.json', 'r') as f:
        data = json.load(f)
    
    examples = data['train']
    total_samples = len(examples)
    print(f"📈 总样本数: {total_samples:,}")
    
    # 按受试者分组 - 带进度条
    print("🔍 按受试者分组...")
    subject_groups = defaultdict(list)
    for example in tqdm(examples, desc="处理样本", unit="样本"):
        subject_groups[example['subject_id']].append(example)
    
    print(f"👥 唯一受试者: {len(subject_groups):,}")
    
    # 分层采样 - 更高效的实现
    target_size = 100000
    random.seed(42)
    
    print("🎯 开始分层采样...")
    
    # 计算每个受试者应保留的样本数
    sampling_ratio = target_size / total_samples
    
    # 预分配结果列表大小
    selected_examples = []
    
    # 使用更高效的采样策略
    subject_list = list(subject_groups.items())
    random.shuffle(subject_list)  # 随机打乱受试者顺序
    
    with tqdm(total=target_size, desc="采样进度", unit="样本") as pbar:
        for subject_id, subject_examples in subject_list:
            if len(selected_examples) >= target_size:
                break
                
            keep_count = max(1, int(len(subject_examples) * sampling_ratio))
            
            if len(subject_examples) <= keep_count:
                selected_examples.extend(subject_examples)
                pbar.update(len(subject_examples))
            else:
                selected = random.sample(subject_examples, keep_count)
                selected_examples.extend(selected)
                pbar.update(keep_count)
    
    # 精确调整到10万样本
    final_selected = selected_examples[:target_size]
    print(f"✅ 最终选择样本数: {len(final_selected):,}")
    
    # 验证统计信息
    unique_subjects = len(set(ex['subject_id'] for ex in final_selected))
    unique_studies = len(set(ex['study_id'] for ex in final_selected))
    avg_report_length = sum(len(ex['report'].split()) for ex in final_selected) / len(final_selected)
    
    print(f"\n📋 优化数据集统计:")
    print(f"   样本数: {len(final_selected):,}")
    print(f"   受试者: {unique_subjects:,}")
    print(f"   研究数: {unique_studies:,}")
    print(f"   平均报告长度: {avg_report_length:.1f} 词")
    
    # 创建优化数据集
    print("💾 保存优化数据集...")
    optimized_data = {
        'train': final_selected,
        'val': data.get('val', []),
        'test': data.get('test', [])
    }
    
    with open('data/mimic_cxr/annotation_100k.json', 'w') as f:
        json.dump(optimized_data, f, indent=2)
    
    print("✨ 优化完成！")
    
    return len(final_selected)

if __name__ == "__main__":
    start_time = time.time()
    
    try:
        final_count = create_optimized_dataset()
        elapsed_time = time.time() - start_time
        print(f"\n🎉 处理完成！耗时: {elapsed_time:.2f} 秒")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")