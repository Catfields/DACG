#!/usr/bin/env python3
"""
训练集优化脚本
用于分析MIMIC-CXR训练集并提供减少样本数量的优化方案
"""

import json
import os
import random
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime


def load_annotation_data(ann_path):
    """加载注释数据"""
    with open(ann_path, 'r') as f:
        data = json.load(f)
    return data


def analyze_dataset(data, split='train'):
    """分析数据集统计信息"""
    examples = data[split]
    
    # 基础统计
    total_samples = len(examples)
    
    # 按subject_id分组
    subject_groups = defaultdict(list)
    for example in examples:
        subject_id = example['subject_id']
        subject_groups[subject_id].append(example)
    
    # 报告长度统计
    report_lengths = [len(example['report'].split()) for example in examples]
    
    # 唯一报告统计
    unique_reports = set(example['report'] for example in examples)
    
    # 按研究ID分组
    study_groups = defaultdict(list)
    for example in examples:
        study_id = example['study_id']
        study_groups[study_id].append(example)
    
    stats = {
        'total_samples': total_samples,
        'unique_subjects': len(subject_groups),
        'unique_studies': len(study_groups),
        'unique_reports': len(unique_reports),
        'avg_report_length': np.mean(report_lengths),
        'median_report_length': np.median(report_lengths),
        'min_report_length': min(report_lengths),
        'max_report_length': max(report_lengths),
        'samples_per_subject': [len(examples) for examples in subject_groups.values()],
        'samples_per_study': [len(examples) for examples in study_groups.values()]
    }
    
    return stats, examples, subject_groups, study_groups


def random_sampling(examples, target_size=100000, seed=42):
    """随机采样"""
    random.seed(seed)
    if len(examples) <= target_size:
        return examples
    return random.sample(examples, target_size)


def stratified_by_subject_sampling(examples, subject_groups, target_size=100000, seed=42):
    """按受试者分层的采样"""
    random.seed(seed)
    
    # 计算每个受试者应保留的样本比例
    total_subjects = len(subject_groups)
    avg_samples_per_subject = target_size / total_subjects
    
    selected_examples = []
    
    for subject_id, subject_examples in subject_groups.items():
        # 每个受试者至少保留1个样本
        keep_count = max(1, int(len(subject_examples) * (target_size / len(examples))))
        if len(subject_examples) <= keep_count:
            selected_examples.extend(subject_examples)
        else:
            selected_examples.extend(random.sample(subject_examples, keep_count))
    
    # 如果样本过多，随机减少到目标数量
    if len(selected_examples) > target_size:
        selected_examples = random.sample(selected_examples, target_size)
    
    return selected_examples


def quality_based_sampling(examples, target_size=100000):
    """基于报告质量的采样"""
    # 定义质量指标：报告长度、包含关键医学术语等
    quality_scores = []
    
    for example in examples:
        report = example['report'].lower()
        score = 0
        
        # 报告长度得分（适中长度更好）
        word_count = len(report.split())
        if 20 <= word_count <= 100:
            score += 2
        elif 10 <= word_count < 20 or 100 < word_count <= 150:
            score += 1
        
        # 包含关键医学术语
        key_terms = ['consolidation', 'effusion', 'pneumothorax', 'cardiomegaly', 
                    'opacity', 'nodule', 'mass', 'edema', 'fracture']
        for term in key_terms:
            if term in report:
                score += 1
        
        # 包含否定词（可能表示正常情况，也很重要）
        negation_terms = ['no', 'normal', 'clear', 'without']
        for term in negation_terms:
            if term in report:
                score += 0.5
        
        quality_scores.append((example, score))
    
    # 按得分排序，选择高分样本
    quality_scores.sort(key=lambda x: x[1], reverse=True)
    selected_examples = [example for example, score in quality_scores[:target_size]]
    
    return selected_examples


def diversity_based_sampling(examples, target_size=100000):
    """基于多样性的采样"""
    # 使用报告内容的多样性
    report_features = []
    
    for example in examples:
        report = example['report'].lower()
        words = set(report.split())
        report_features.append((example, words))
    
    selected_examples = []
    selected_words = set()
    
    # 贪心算法选择多样性最高的样本
    while len(selected_examples) < target_size and report_features:
        best_example = None
        best_new_words = set()
        best_score = -1
        
        for example, words in report_features:
            new_words = words - selected_words
            score = len(new_words)
            
            if score > best_score:
                best_score = score
                best_example = example
                best_new_words = new_words
        
        if best_example is None:
            break
            
        selected_examples.append(best_example)
        selected_words.update(best_new_words)
        report_features = [(e, w) for e, w in report_features if e != best_example]
    
    # 如果贪心算法选择不足，随机补充
    if len(selected_examples) < target_size:
        remaining = [e for e, w in report_features if e not in selected_examples]
        needed = target_size - len(selected_examples)
        if remaining and needed > 0:
            selected_examples.extend(random.sample(remaining, min(needed, len(remaining))))
    
    return selected_examples


def create_optimized_dataset(original_examples, selected_examples, output_path):
    """创建优化后的数据集文件"""
    optimized_data = {
        'train': selected_examples,
        'info': {
            'original_size': len(original_examples),
            'optimized_size': len(selected_examples),
            'reduction_ratio': len(selected_examples) / len(original_examples),
            'creation_date': datetime.now().isoformat(),
            'selection_method': 'optimized'
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(optimized_data, f, indent=2)
    
    print(f"优化后的数据集已保存到: {output_path}")
    print(f"原始样本数: {len(original_examples)}")
    print(f"优化后样本数: {len(selected_examples)}")
    print(f"减少比例: {(1 - len(selected_examples)/len(original_examples)) * 100:.2f}%")


def main():
    # 配置路径
    ann_path = '/home/y530/handsome/DACG/data/mimic_cxr/annotation.json'
    
    # 加载数据
    print("正在加载数据...")
    data = load_annotation_data(ann_path)
    
    # 分析数据集
    print("正在分析数据集...")
    stats, examples, subject_groups, study_groups = analyze_dataset(data)
    
    print("\n=== 数据集统计信息 ===")
    print(f"总样本数: {stats['total_samples']}")
    print(f"唯一受试者: {stats['unique_subjects']}")
    print(f"唯一研究: {stats['unique_studies']}")
    print(f"唯一报告: {stats['unique_reports']}")
    print(f"平均报告长度: {stats['avg_report_length']:.2f} 词")
    print(f"中位数报告长度: {stats['median_report_length']:.2f} 词")
    print(f"每受试者平均样本数: {np.mean(stats['samples_per_subject']):.2f}")
    print(f"每研究平均样本数: {np.mean(stats['samples_per_study']):.2f}")
    
    # 目标规模
    target_size = 100000
    
    # 应用不同的采样策略
    print(f"\n=== 应用采样策略 (目标规模: {target_size}) ===")
    
    strategies = {
        'random': random_sampling,
        'stratified_subject': lambda ex, size: stratified_by_subject_sampling(ex, subject_groups, size),
        'quality_based': quality_based_sampling,
        'diversity_based': diversity_based_sampling
    }
    
    results = {}
    for strategy_name, strategy_func in strategies.items():
        print(f"\n应用 {strategy_name} 采样...")
        selected = strategy_func(examples, target_size)
        results[strategy_name] = selected
        
        # 计算统计信息
        unique_subjects = len(set(ex['subject_id'] for ex in selected))
        unique_studies = len(set(ex['study_id'] for ex in selected))
        avg_report_len = np.mean([len(ex['report'].split()) for ex in selected])
        
        print(f"  选择样本数: {len(selected)}")
        print(f"  唯一受试者: {unique_subjects}")
        print(f"  唯一研究: {unique_studies}")
        print(f"  平均报告长度: {avg_report_len:.2f} 词")
    
    # 保存推荐方案
    recommended_strategy = 'stratified_subject'  # 推荐使用分层采样
    selected_examples = results[recommended_strategy]
    
    output_path = '/home/y530/handsome/DACG/data/mimic_cxr/annotation_optimized_100k.json'
    create_optimized_dataset(examples, selected_examples, output_path)
    
    # 创建训练脚本
    create_training_script(recommended_strategy)


def create_training_script(strategy_name):
    """创建使用优化数据集的训练脚本"""
    script_content = f'''#!/bin/bash
# 使用优化数据集的训练脚本
# 基于{strategy_name}策略选择100k样本

accelerate launch train_with_accelerate.py \
    --image_dir data/mimic_cxr/images \
    --ann_path data/mimic_cxr/annotation_optimized_100k.json \
    --dataset_name mimic_cxr \
    --max_seq_length 100 \
    --threshold 3 \
    --batch_size 16 \
    --gradient_accumulation_steps 1 \
    --epochs 100 \
    --save_dir results/mimic_cxr_optimized \
    --step_size 50 \
    --gamma 0.1 \
    --early_stop 50 \
    --seed 9233 \
    --use_amp

'''
    
    script_path = '/home/y530/handsome/DACG/run_mimic_cxr_optimized.sh'
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"优化训练脚本已创建: {script_path}")


if __name__ == '__main__':
    main()