import os
from abc import abstractmethod
import logging
import sys

import time
import torch
import pandas as pd
from numpy import inf
from accelerate import Accelerator


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args

        # 配置日志，确保在tmux下也能正常输出
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S', 
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(os.path.join(args.save_dir, 'training.log'))
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 强制刷新stdout，解决tmux下的缓冲问题
        if os.environ.get('TMUX'):
            self.logger.info("检测到tmux环境，启用实时输出模式")
            sys.stdout.flush()
        
        # 使用Accelerate初始化
        self.accelerator = Accelerator(
            gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 1),
            mixed_precision="fp16" if getattr(args, 'use_amp', False) else "no",
            log_with="tensorboard" if getattr(args, 'use_tensorboard', False) else None
        )
        
        self.device = self.accelerator.device
        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            for key, value in log.items():
                self.logger.info('\t{:15s}: {}'.format(str(key), value))

            best = False
            if self.mnt_mode != 'off':
                try:
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        new_records = pd.DataFrame([self.best_recorder['val'], self.best_recorder['test']])
        record_table = pd.concat([record_table, new_records], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    # Accelerate自动处理设备，不再需要_prepare_device方法
    pass

    def _save_checkpoint(self, epoch, save_best=False):
        # 使用Accelerate的保存方法
        self.accelerator.save_state(self.checkpoint_dir)
        
        # 保存额外的元信息
        metadata = {
            'epoch': epoch,
            'monitor_best': self.mnt_best
        }
        metadata_path = os.path.join(self.checkpoint_dir, 'metadata.pth')
        torch.save(metadata, metadata_path)
        
        self.logger.info("Saving checkpoint: {} ...".format(self.checkpoint_dir))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best')
            self.accelerator.save_state(best_path)
            torch.save(metadata, os.path.join(best_path, 'metadata.pth'))
            self.logger.info("Saving current best: model_best ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        
        # 使用Accelerate的加载方法
        self.accelerator.load_state(resume_path)
        
        # 加载额外的元信息
        metadata_path = os.path.join(resume_path, 'metadata.pth')
        if os.path.exists(metadata_path):
            metadata = torch.load(metadata_path)
            self.start_epoch = metadata['epoch'] + 1
            self.mnt_best = metadata['monitor_best']
            self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
        else:
            # 兼容旧的checkpoint格式
            checkpoint = torch.load(os.path.join(resume_path, 'current_checkpoint.pth'))
            self.start_epoch = checkpoint['epoch'] + 1
            self.mnt_best = checkpoint['monitor_best']
            self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        self.logger.info('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

        self.logger.info('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        
        # 使用Accelerate准备所有组件
        self.model, self.optimizer, self.lr_scheduler, self.train_dataloader, self.val_dataloader, self.test_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, lr_scheduler, train_dataloader, val_dataloader, test_dataloader
        )

    def _train_epoch(self, epoch):
        import time
        from tqdm import tqdm
        import os
        
        # 检测是否在tmux环境下运行
        is_tmux = os.environ.get('TMUX') is not None
        
        self.logger.info('[{}/{}] Start to train in the training set.'.format(epoch, self.epochs))
        train_loss = 0
        self.model.train()
        
        # 计算总batch数
        total_batches = len(self.train_dataloader)
        
        # 配置tqdm参数，优化tmux环境下的显示
        tqdm_kwargs = {
            'total': total_batches,
            'desc': f"Epoch {epoch}/{self.epochs}",
            'disable': not self.accelerator.is_main_process,
            'ncols': 100 if is_tmux else 120,  # tmux下使用较窄的列宽
            'leave': True,
            'file': sys.stdout
        }
        
        # 在tmux环境下禁用动态更新，使用更简单的进度条
        if is_tmux:
            tqdm_kwargs.update({
                'dynamic_ncols': False,
                'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            })
        
        progress_bar = tqdm(enumerate(self.train_dataloader), **tqdm_kwargs)
        
        start_time = time.time()
        
        for batch_idx, batch_data in progress_bar:
            images_id, images, reports_ids, reports_masks = batch_data
            
            with self.accelerator.accumulate(self.model):
                output = self.model(images, reports_ids, mode='train')
                loss = self.criterion(output, reports_ids, reports_masks)
                
                train_loss += loss.item()
                self.accelerator.backward(loss)
                
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.optimizer.zero_grad()

            # 计算当前平均损失
            current_loss = train_loss / (batch_idx + 1)
            
            # 计算已用时间和预估剩余时间
            elapsed_time = time.time() - start_time
            batches_per_second = (batch_idx + 1) / elapsed_time if elapsed_time > 0 else 0
            remaining_batches = total_batches - (batch_idx + 1)
            estimated_remaining_time = remaining_batches / batches_per_second if batches_per_second > 0 else 0
            
            # 格式化剩余时间
            if estimated_remaining_time < 60:
                time_str = f"{estimated_remaining_time:.1f}s"
            elif estimated_remaining_time < 3600:
                time_str = f"{estimated_remaining_time/60:.1f}m"
            else:
                time_str = f"{estimated_remaining_time/3600:.1f}h"
            
            # 更新进度条信息
            progress_bar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Time': time_str
            })
            
            # 在tmux环境下，增加日志输出的频率
            log_period = 10 if is_tmux else self.args.log_period
            if batch_idx % log_period == 0 and self.accelerator.is_main_process:
                self.logger.info('[{}/{}] Step: {}/{}, Training Loss: {:.5f}, ETA: {}'
                                 .format(epoch, self.epochs, batch_idx, total_batches,
                                         current_loss, time_str))
                # 强制刷新输出
                sys.stdout.flush()

        log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.logger.info('[{}/{}] Start to evaluate in the validation set.'.format(epoch, self.epochs))
        self.model.eval()
        val_gts, val_res = [], []
        
        # 为验证集添加进度条
        val_tqdm_kwargs = {
            'total': len(self.val_dataloader),
            'desc': f"Validation Epoch {epoch}",
            'disable': not self.accelerator.is_main_process,
            'ncols': 100 if is_tmux else 120,
            'leave': True,
            'file': sys.stdout
        }
        
        if is_tmux:
            val_tqdm_kwargs.update({
                'dynamic_ncols': False,
                'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt}'
            })
        
        val_progress_bar = tqdm(enumerate(self.val_dataloader), **val_tqdm_kwargs)
        
        for batch_idx, batch_data in val_progress_bar:
            images_id, images, reports_ids, reports_masks = batch_data
            seq_ids = self.model(images, mode='sample')                    # (B, T) LongTensor
            seq_ids = seq_ids.long()                                       # 保险起见
            seq_ids_gathered = self.accelerator.gather(seq_ids)            # DDP 收集
            reports = self.model.tokenizer.decode_batch(
                seq_ids_gathered.detach().cpu().numpy()
            )
            reports_ids_gathered = self.accelerator.gather(reports_ids)
            ground_truths = self.model.tokenizer.decode_batch(reports_ids_gathered[:, 1:].detach().cpu().numpy())
            val_res.extend(reports)
            val_gts.extend(ground_truths)
            
            # 更新验证进度条
            val_progress_bar.set_postfix({'Progress': f'{batch_idx+1}/{len(self.val_dataloader)}'})

        # 只在主进程计算指标
        # 只在主进程打印调试信息
        if self.accelerator.is_main_process:
            # 检查val_res和val_gts是否有内容
            self.logger.info(f"验证集预测和真实标签样本量: {len(val_res)}, {len(val_gts)}")
            # 打印几个样本进行检查
            if len(val_res) > 0 and len(val_gts) > 0:
                self.logger.info(f"验证集样本示例(前3个):")
                for i in range(min(3, len(val_res))):
                    self.logger.info(f"预测[{i}]: '{val_res[i]}'")
                    self.logger.info(f"真实[{i}]: '{val_gts[i]}'")
            else:
                self.logger.warning("验证集预测或真实标签为空!")
        if self.accelerator.is_main_process:
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.logger.info('[{}/{}] Start to evaluate in the test set.'.format(epoch, self.epochs))
        self.model.eval()
        test_gts, test_res = [], []
        
        # 为测试集添加进度条
        test_tqdm_kwargs = {
            'total': len(self.test_dataloader),
            'desc': f"Test Epoch {epoch}",
            'disable': not self.accelerator.is_main_process,
            'ncols': 100 if is_tmux else 120,
            'leave': True,
            'file': sys.stdout
        }
        
        if is_tmux:
            test_tqdm_kwargs.update({
                'dynamic_ncols': False,
                'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt}'
            })
        
        test_progress_bar = tqdm(enumerate(self.test_dataloader), **test_tqdm_kwargs)
        
        for batch_idx, batch_data in test_progress_bar:
            images_id, images, reports_ids, reports_masks = batch_data
            
            output = self.model(images, mode='sample')
            output_gathered = self.accelerator.gather(output)
            reports_ids_gathered = self.accelerator.gather(reports_ids)
            
            reports = self.model.tokenizer.decode_batch(output_gathered.detach().cpu().numpy())
            ground_truths = self.model.tokenizer.decode_batch(reports_ids_gathered[:, 1:].detach().cpu().numpy())
            test_res.extend(reports)
            test_gts.extend(ground_truths)
            
            # 更新测试进度条
            test_progress_bar.set_postfix({'Progress': f'{batch_idx+1}/{len(self.test_dataloader)}'})

        if self.accelerator.is_main_process:
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        self.lr_scheduler.step()

        # 在每个epoch结束时强制刷新输出
        if is_tmux and self.accelerator.is_main_process:
            sys.stdout.flush()

        return log
