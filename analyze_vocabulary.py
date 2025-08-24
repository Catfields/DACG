#!/usr/bin/env python3
"""
分析DACG模型的词表大小及其决定因素
"""

import json
import argparse
from collections import Counter
from modules.tokenizers import Tokenizer

def analyze_vocabulary_size():
    """分析词表大小及其决定因素"""
    
    print("=" * 80)
    print("DACG模型词表大小分析")
    print("=" * 80)
    
    # 模拟args配置
    class Args:
        def __init__(self):
            self.ann_path = 'data/mimic_cxr/annotation.json'
            self.threshold = 3  # 默认阈值
            self.dataset_name = 'mimic_cxr'
            self.max_seq_length = 60
    
    args = Args()
    
    # 分析原始数据
    print("1. 原始数据分析")
    print("-" * 40)
    
    with open(args.ann_path, 'r') as f:
        ann = json.load(f)
    
    # 收集所有tokens
    total_tokens = []
    tokenizer = Tokenizer(args)
    
    for example in ann['train']:
        tokens = tokenizer.clean_report(example['report']).split()
        total_tokens.extend(tokens)
    
    counter = Counter(total_tokens)
    
    print(f"原始token总数: {len(total_tokens):,}")
    print(f"唯一token数量: {len(counter):,}")
    print(f"频率阈值: {args.threshold}")
    
    # 分析不同阈值的影响
    print("\n2. 不同阈值对词表大小的影响")
    print("-" * 40)
    
    thresholds = [1, 2, 3, 5, 10]
    
    for threshold in thresholds:
        vocab_words = [k for k, v in counter.items() if v >= threshold]
        vocab_size = len(vocab_words) + 4  # +4 for special tokens
        
        print(f"阈值={threshold}: 词表大小={vocab_size:,} (词汇: {len(vocab_words)}, 特殊: 4)")
    
    # 当前实际词表大小
    print("\n3. 当前实际词表大小")
    print("-" * 40)
    
    current_vocab_size = tokenizer.get_vocab_size()
    print(f"当前词表大小: {current_vocab_size:,}")
    
    # 特殊token
    special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
    print(f"特殊tokens: {special_tokens}")
    
    # 高频词分析
    print("\n4. 高频词分析")
    print("-" * 40)
    
    most_common = counter.most_common(20)
    print("Top 20高频词:")
    for word, count in most_common:
        print(f"  {word}: {count:,}")
    
    # 词表大小对模型的影响
    print("\n5. 词表大小的影响")
    print("-" * 40)
    
    embedding_params = current_vocab_size * 512  # d_model=512
    print(f"嵌入层参数量: {embedding_params:,}")
    print(f"内存占用: {embedding_params * 4 / (1024**2):.2f} MB")
    
    return current_vocab_size

if __name__ == "__main__":
    analyze_vocabulary_size()