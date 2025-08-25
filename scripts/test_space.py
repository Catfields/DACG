import os
import torch
import torchvision.transforms as transforms
from PIL import Image

def preprocess(img):
    """
    将图像转换为张量的预处理函数
    这是一个通用的预处理流程，你可以根据实际需求调整
    """
    # 定义预处理转换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),      # 调整图像大小
        transforms.ToTensor(),              # 转换为张量 (0-1范围)
        transforms.Normalize(               # 标准化 (常见于深度学习)
            mean=[0.485, 0.456, 0.406],     # ImageNet均值
            std=[0.229, 0.224, 0.225]       # ImageNet标准差
        )
    ])
    
    # 如果是灰度图像，转换为RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # 应用转换
    return transform(img)

# 检查原始文件大小
image_path = 'data/mimic_cxr/images/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg'
original_size = os.path.getsize(image_path)

# 检查处理后的张量大小
img = Image.open(image_path)
img_tensor = preprocess(img)  # 使用预处理函数
processed_size = img_tensor.element_size() * img_tensor.nelement()

print(f"原始大小: {original_size/1024:.2f} KB")
print(f"处理后大小: {processed_size/1024/1024:.2f} MB")
print(f"膨胀倍数: {processed_size/original_size:.1f}x")
print(f"张量形状: {img_tensor.shape}")
print(f"数据类型: {img_tensor.dtype}")