#!/bin/bash

# 仅使用已训练好的模型进行验证和测试
# 这个脚本用于加载已训练好的最佳模型，并只执行验证和测试阶段

# 设置Python无缓冲输出，确保在tmux下能看到所有日志
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# 设置tqdm在tmux下正常工作
export TQDM_DISABLE=0

# 指定加载的模型路径
MODEL_DIR="results/mimic_cxr_100k"

# 创建一个新的评估脚本
cat > eval_only.py << 'EOL'
#!/usr/bin/env python3
"""
模型评估脚本 - 只运行验证和测试阶段，不进行训练
"""
import argparse
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from modules.tokenizers import Tokenizer
from modules.dataloaders import DACGdataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from models.dacg import DACGModel
import os
import logging
import sys

def parse_args():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/mimic_cxr/images/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/mimic_cxr/annotation.json', help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr', choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=4, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor) - 添加缺少的参数
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
    
    # for Relational Memory
    parser.add_argument('--rm_num_slots', type=int, default=3, help='the number of memory slots.')
    parser.add_argument('--rm_num_heads', type=int, default=8, help='the numebr of heads in rm.')
    parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Other settings
    parser.add_argument('--seed', type=int, default=9530, help='random seed')
    parser.add_argument('--model_path', type=str, required=True, help='path to the trained model directory')
    parser.add_argument('--detailed_output', action='store_true', help='whether to print detailed output')
    parser.add_argument('--output_dir', type=str, default='eval_results', help='directory to save evaluation results')
    
    # 添加其他必要参数，以便和train_with_accelerate.py兼容
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr_ve', type=float, default=5e-5)
    parser.add_argument('--lr_ed', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--amsgrad', type=bool, default=True)
    parser.add_argument('--lr_scheduler', type=str, default='StepLR')
    parser.add_argument('--step_size', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--record_dir', type=str, default='records')
    parser.add_argument('--monitor_mode', type=str, default='max')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4')
    parser.add_argument('--early_stop', type=int, default=50)
    parser.add_argument('--save_period', type=int, default=1)
    parser.add_argument('--log_period', type=int, default=1000)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    
    args = parser.parse_args()
    return args

def evaluate_model():
    # 解析参数
    args = parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.output_dir, 'eval.log'))
        ]
    )
    logger = logging.getLogger("model_evaluator")
    
    # 使用Accelerate初始化
    accelerator = Accelerator()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 只在主进程打印参数
    if accelerator.is_main_process:
        logger.info("Evaluation Arguments:")
        for arg in vars(args):
            logger.info(f"  {arg}: {getattr(args, arg)}")
    
    # 创建tokenizer
    tokenizer = Tokenizer(args)
    args.vocab_size = tokenizer.get_vocab_size()
    
    # 创建数据加载器（只需要验证和测试集）
    val_dataloader = DACGdataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = DACGdataLoader(args, tokenizer, split='test', shuffle=False)
    
    # 构建模型
    model = DACGModel(args, tokenizer)
    
    # 获取损失和评估指标函数
    criterion = compute_loss
    metrics = compute_scores
    
    # 构建优化器和学习率调度器（虽然不会用于训练，但Trainer需要它们）
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)
    
    # 创建Trainer
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, None, val_dataloader, test_dataloader)
    
    # 修改Trainer.train方法，使其只执行验证和测试
    def evaluate_only(self):
        # 加载模型
        logger.info(f"Loading model from {args.model_path}")
        
        # 使用accelerator加载状态
        # 检查是否是模型目录还是特定的checkpoint
        if os.path.isdir(args.model_path):
            # 首先尝试加载best模型
            best_path = os.path.join(args.model_path, 'model_best')
            if os.path.exists(best_path):
                logger.info(f"Loading best model from {best_path}")
                self.accelerator.load_state(best_path)
            else:
                # 如果没有best模型，则加载最新的checkpoint
                logger.info(f"Best model not found, loading latest checkpoint from {args.model_path}")
                self.accelerator.load_state(args.model_path)
        else:
            # 直接加载指定路径
            logger.info(f"Loading checkpoint from {args.model_path}")
            self.accelerator.load_state(args.model_path)
        
        # 创建一个空的log字典
        log = {}
        
        # 在验证集上评估
        logger.info('Evaluating on validation set...')
        self.model.eval()
        val_gts, val_res = [], []
        
        for batch_idx, batch_data in enumerate(self.val_dataloader):
            images_id, images, reports_ids, reports_masks = batch_data
            
            with torch.no_grad():
                output = self.model(images, mode='sample')
                
            # 使用accelerator.gather收集所有进程的结果
            output_gathered = self.accelerator.gather(output)
            reports_ids_gathered = self.accelerator.gather(reports_ids)
            
            reports = self.model.tokenizer.decode_batch(output_gathered.detach().cpu().numpy())
            ground_truths = self.model.tokenizer.decode_batch(reports_ids_gathered[:, 1:].detach().cpu().numpy())
            
            val_res.extend(reports)
            val_gts.extend(ground_truths)
        
        # 在主进程上计算和打印指标
        if self.accelerator.is_main_process:
            # 详细输出
            logger.info(f"Validation set predictions and ground truths: {len(val_res)}, {len(val_gts)}")
            
            if args.detailed_output and len(val_res) > 0 and len(val_gts) > 0:
                logger.info("Validation samples (first 10):")
                for i in range(min(10, len(val_res))):
                    logger.info(f"Pred[{i}]: '{val_res[i]}'")
                    logger.info(f"GT[{i}]: '{val_gts[i]}'")
                
                # 统计空预测数量
                empty_cnt = sum(1 for x in val_res if not x.strip())
                logger.info(f"Empty predictions: {empty_cnt}/{len(val_res)}")
            
            # 计算评估指标
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                     {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})
            
            logger.info("Validation Results:")
            for key, value in {k: v for k, v in log.items() if k.startswith('val_')}.items():
                logger.info(f"  {key}: {value}")
        
        # 在测试集上评估
        logger.info('Evaluating on test set...')
        self.model.eval()
        test_gts, test_res = [], []
        
        for batch_idx, batch_data in enumerate(self.test_dataloader):
            images_id, images, reports_ids, reports_masks = batch_data
            
            with torch.no_grad():
                output = self.model(images, mode='sample')
                
            # 使用accelerator.gather收集所有进程的结果
            output_gathered = self.accelerator.gather(output)
            reports_ids_gathered = self.accelerator.gather(reports_ids)
            
            reports = self.model.tokenizer.decode_batch(output_gathered.detach().cpu().numpy())
            ground_truths = self.model.tokenizer.decode_batch(reports_ids_gathered[:, 1:].detach().cpu().numpy())
            
            test_res.extend(reports)
            test_gts.extend(ground_truths)
        
        # 在主进程上计算和打印指标
        if self.accelerator.is_main_process:
            # 详细输出
            logger.info(f"Test set predictions and ground truths: {len(test_res)}, {len(test_gts)}")
            
            if args.detailed_output and len(test_res) > 0 and len(test_gts) > 0:
                logger.info("Test samples (first 10):")
                for i in range(min(10, len(test_res))):
                    logger.info(f"Pred[{i}]: '{test_res[i]}'")
                    logger.info(f"GT[{i}]: '{test_gts[i]}'")
                
                # 统计空预测数量
                empty_cnt = sum(1 for x in test_res if not x.strip())
                logger.info(f"Empty predictions: {empty_cnt}/{len(test_res)}")
            
            # 计算评估指标
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                      {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            
            logger.info("Test Results:")
            for key, value in {k: v for k, v in log.items() if k.startswith('test_')}.items():
                logger.info(f"  {key}: {value}")
            
            # 保存评估结果
            import json
            with open(os.path.join(args.output_dir, 'eval_results.json'), 'w') as f:
                json.dump(log, f, indent=2)
            
            # 保存一些预测样本
            with open(os.path.join(args.output_dir, 'sample_predictions.txt'), 'w') as f:
                f.write("=== Validation Set Samples ===\n")
                for i in range(min(20, len(val_res))):
                    f.write(f"Sample {i}:\n")
                    f.write(f"Prediction: {val_res[i]}\n")
                    f.write(f"Ground Truth: {val_gts[i]}\n\n")
                
                f.write("\n=== Test Set Samples ===\n")
                for i in range(min(20, len(test_res))):
                    f.write(f"Sample {i}:\n")
                    f.write(f"Prediction: {test_res[i]}\n")
                    f.write(f"Ground Truth: {test_gts[i]}\n\n")
        
        return log
    
    # 替换Trainer的train方法
    Trainer.train = evaluate_only
    
    # 运行评估
    trainer.train()

if __name__ == '__main__':
    evaluate_model()
EOL

# 使模型评估脚本可执行
chmod +x eval_only.py

# 运行模型评估脚本
# 使用同样的数据参数，但指定加载模型的路径
echo "开始评估模型: $MODEL_DIR"
accelerate launch --config_file accelerate_config_single_gpu.yaml eval_only.py \
    --image_dir data/mimic_cxr/images/ \
    --ann_path data/mimic_cxr/annotation_100k.json \
    --dataset_name mimic_cxr \
    --max_seq_length 60 \
    --batch_size 2 \
    --model_path $MODEL_DIR \
    --detailed_output \
    --output_dir eval_results_mimic_cxr_100k \
    --seed 456789 \
    --use_amp \
    --visual_extractor resnet101 \
    --visual_extractor_pretrained True \
    --d_model 512 \
    --d_ff 512 \
    --num_heads 8 \
    --num_layers 3 \
    --dropout 0.1 \
    --optim AdamW \
    --lr_ed 1e-5 \
    --lr_ve 5e-5 \
    --step_size 50 \
    --gamma 0.1 \
    --record_dir records/

echo "评估完成! 结果保存在 eval_results_mimic_cxr_100k 目录"