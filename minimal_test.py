#!/usr/bin/env python3
"""
最小化的测试脚本，只测试前向传播
"""

import os
import sys
import torch
import json
from torch.utils.data import DataLoader, Subset
import argparse

# 添加项目根目录到路径
sys.path.append('/home/y530/handsome/DACG')

from modules.tokenizers import Tokenizer
from modules.dataloaders import DACGdataLoader
from models.dacg import DACGModel
from modules.loss import compute_loss

def main():
    # 设置参数
    class Args:
        def __init__(self):
            self.image_dir = "data/mimic_cxr/images/"
            self.ann_path = "data/mimic_cxr/annotation_100k.json"
            self.dataset_name = "mimic_cxr"
            self.max_seq_length = 60
            self.threshold = 3
            self.batch_size = 2
            self.num_workers = 0
            self.d_model = 512
            self.d_ff = 512
            self.d_vf = 2048
            self.num_heads = 8
            self.num_layers = 3
            self.dropout = 0.1
            self.bos_idx = 1
            self.eos_idx = 2
            self.pad_idx = 0
            self.drop_prob_lm = 0.5
            self.visual_extractor = "resnet101"
            self.visual_extractor_pretrained = True
            self.use_bn = 0
            self.logit_layers = 1
            self.rm_num_slots = 3
            self.rm_num_heads = 8
            self.rm_d_model = 512

    args = Args()
    
    # 创建tokenizer
    print("创建tokenizer...")
    tokenizer = Tokenizer(args)
    args.vocab_size = tokenizer.get_vocab_size()
    print(f"词汇表大小: {args.vocab_size}")
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_dataloader = DACGdataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = DACGdataLoader(args, tokenizer, split='val', shuffle=False)
    
    # 限制数据量
    print(f"原始训练数据: {len(train_dataloader.dataset)}")
    print(f"原始验证数据: {len(val_dataloader.dataset)}")
    
    # 创建子集
    train_subset = Subset(train_dataloader.dataset, list(range(min(10, len(train_dataloader.dataset)))))
    val_subset = Subset(val_dataloader.dataset, list(range(min(5, len(val_dataloader.dataset)))))
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=train_dataloader.collate_fn
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=val_dataloader.collate_fn
    )
    
    print(f"训练样本: {len(train_subset)}")
    print(f"验证样本: {len(val_subset)}")
    
    # 创建模型
    print("创建模型...")
    model = DACGModel(args, tokenizer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"使用设备: {device}")
    
    # 测试一个批次
    print("\n测试训练批次...")
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        print(f"批次 {batch_idx+1}:")
        print(f"批次长度: {len(batch)}")
        
        if len(batch) == 5:
            image_ids, images, reports_ids, reports_masks, seq_lengths = batch
        else:
            image_ids, images, reports_ids, reports_masks = batch
            seq_lengths = None
            
        print(f"图像形状: {images.shape}")
        print(f"报告ID形状: {reports_ids.shape}")
        print(f"报告掩码形状: {reports_masks.shape}")
        
        # 移动到设备
        images = images.to(device)
        reports_ids = reports_ids.to(device)
        reports_masks = reports_masks.to(device)
        
        # 前向传播
        try:
            output = model(images, reports_ids[:, :-1], mode='train')
            print(f"输出形状: {output.shape}")
            
            # 计算损失
            loss = compute_loss(output, reports_ids[:, 1:], reports_masks[:, 1:])
            print(f"损失值: {loss.item():.4f}")
            
            # 反向传播
            loss.backward()
            print("反向传播成功")
            
        except Exception as e:
            print(f"前向传播错误: {e}")
            raise e
            
        break
    
    # 测试验证批次
    print("\n测试验证批次...")
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if len(batch) == 5:
                image_ids, images, reports_ids, reports_masks, seq_lengths = batch
            else:
                image_ids, images, reports_ids, reports_masks = batch
                
            images = images.to(device)
            reports_ids = reports_ids.to(device)
            reports_masks = reports_masks.to(device)
            
            try:
                output = model(images, reports_ids[:, :-1], mode='train')
                loss = compute_loss(output, reports_ids[:, 1:], reports_masks[:, 1:])
                print(f"验证损失: {loss.item():.4f}")
                
                # 测试采样
                output_sample, _ = model(images, mode='sample')
                print(f"采样输出形状: {output_sample.shape}")
                
            except Exception as e:
                print(f"验证错误: {e}")
                raise e
                
            break
    
    print("\n测试成功完成！")
    print("模型可以正常运行，数据加载和前向传播都工作正常。")


if __name__ == '__main__':
    main()