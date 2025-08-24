import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        vocab_size = len(tokenizer.idx2token)
        for i in range(len(self.examples)):
            report_ids = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            # --- 添加这段代码进行检查 ---
            max_idx = max(report_ids) if report_ids else -1
            if max_idx >= vocab_size:
                print(f"Error found in example {i} (report: {self.examples[i]['report']})")
                print(f"Max index is {max_idx}, but vocab size is {vocab_size}.")
                # 抛出错误以中断执行，便于调试
                raise ValueError("Index out of bounds detected during data loading.")
            # ---------------------------

            self.examples[i]['ids'] = report_ids
            self.examples[i]['mask'] = [1] * len(report_ids)

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        
        # 原始代码中的 image_path 可能是一个列表
        # image_path = example['image_path'] 
        
        # 修改如下：从列表中获取第一个元素，确保 image_path 是一个字符串
        image_path_list = example['image_path']
        if len(image_path_list) > 0:
            image_path = image_path_list[0]
        else:
            # 处理列表为空的情况，根据你的数据处理逻辑决定是跳过还是抛出错误
            raise ValueError(f"Image path list is empty for example at index {idx}")

        # MIMIC-CXR 数据集只有一张图像
        image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
            
        # 添加一个维度以保持与 IuxrayMultiImageDataset 返回格式一致
        # 但这里只有一张图像，所以维度为 [1, C, H, W]
        image = image.unsqueeze(0)
        
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample



