'''
Author: Catfield 2022211136@stu.hit.edu.cn
Date: 2025-08-22 11:25:52
LastEditors: Catfield 2022211136@stu.hit.edu.cn
LastEditTime: 2025-08-22 11:46:14
FilePath: /DACG/scripts/extract_small_dataset.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import json
import random
import argparse
import os
import shutil

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 解析命令行参数
parser = argparse.ArgumentParser(description="从原始数据集中随机抽取小样本")
parser.add_argument("--count", type=int, default=100, help="抽取样本数量")
parser.add_argument("--seed", type=int, default=42, help="随机种子")
args = parser.parse_args()

# 设置路径
original_annotation_path = "data/mimic_cxr/annotation.json"
original_image_dir = "data/mimic_cxr/images"

# 设置新的小样本数据集路径
sample_dir = "data/mimic_cxr/image100"
sample_image_dir = os.path.join(sample_dir, "images")
sample_annotation_path = os.path.join(sample_dir, "annotation.json")

# 创建必要的目录
ensure_dir(sample_dir)
ensure_dir(sample_image_dir)

# 设置随机种子
random.seed(args.seed)

# 读取原始数据集
with open(original_annotation_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 检查数据类型并抽取样本
if isinstance(data, dict):
    keys = list(data.keys())
    sampled_keys = random.sample(keys, min(args.count, len(keys)))
    small_data = {k: data[k] for k in sampled_keys}

    # 复制对应的图片文件
    for key in sampled_keys:
        image_path = data[key]["image_path"]
        src_path = os.path.join(original_image_dir, image_path)
        dst_path = os.path.join(sample_image_dir, image_path)
        
        # 确保目标目录存在
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        # 复制图片文件
        shutil.copy2(src_path, dst_path)

elif isinstance(data, list):
    small_data = random.sample(data, min(args.count, len(data)))
    
    # 复制对应的图片文件
    for item in small_data:
        image_path = item["image_path"]
        src_path = os.path.join(original_image_dir, image_path)
        dst_path = os.path.join(sample_image_dir, image_path)
        
        # 确保目标目录存在
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        # 复制图片文件
        shutil.copy2(src_path, dst_path)
else:
    raise TypeError("不支持的数据类型")

# 保存小样本数据集的标注文件
with open(sample_annotation_path, "w", encoding="utf-8") as f:
    json.dump(small_data, f, ensure_ascii=False, indent=4)

# 创建样本记录文件，记录抽样信息
record = {
    "sample_count": len(small_data),
    "seed": args.seed,
    "timestamp": "2025-08-22",
    "sampled_files": list(small_data.keys()) if isinstance(small_data, dict) else [item["image_path"] for item in small_data]
}

with open(os.path.join(sample_dir, "sample_record.json"), "w", encoding="utf-8") as f:
    json.dump(record, f, ensure_ascii=False, indent=4)

print(f"小样本数据集已创建：")
print(f"- 图片目录：{sample_image_dir}")
print(f"- 标注文件：{sample_annotation_path}")
print(f"- 样本记录：{os.path.join(sample_dir, 'sample_record.json')}")
print(f"共包含 {len(small_data)} 个样本")