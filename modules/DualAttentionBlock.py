import torch
import torch.nn as nn
import torch.nn.functional as F


class PAB(nn.Module):
    """
    Patch Attention Branch (PAB) for computing Spatial Attention Graph (SAG)
    
    医学图像空间注意力图计算模块，基于补丁特征计算位置间的注意力权重
    """
    
    def __init__(self, temperature=1.0):
        super(PAB, self).__init__()
        self.temperature = temperature
    
    def forward(self, A, alpha=None):
        """
        完整的PAB前向传播，包含SAG计算和特征增强
        
        Args:
            A (torch.Tensor): 医学图像补丁特征，维度为 (M, C)
                             M = H * W 是补丁总数，C 是特征通道数
            alpha (torch.Tensor, optional): 可学习参数，默认为None
        
        Returns:
            E (torch.Tensor): 增强后的特征，形状为 (M, C)
            SAG (torch.Tensor): 空间注意力图，形状为 (M, M)
        """
        if alpha is None:
            alpha = torch.tensor(0.0, device=A.device, dtype=A.dtype)
        # 获取维度信息
        M, C = A.shape
        # 计算空间注意力图 (SAG)
        # SAG = softmax(A * A.T / temperature)
        SAG = torch.einsum('mc,nc->mn', A, A) / self.temperature
        SAG = F.softmax(SAG, dim=-1)
        # 根据公式 (3) 及文字描述，计算最终输出特征 E
        # D = A.t() 对应 'cm'
        # temp1 = D @ SAG 对应 'cm,mn->cn'
        # temp2 = temp1.t() 对应 'nc'
        # E = temp2 * alpha + A
        # (M, C), (M, M) -> (M, C)
        E = torch.einsum('mc,mn->nc', A, SAG) * alpha + A
        return E, SAG
class CAB(nn.Module):
    """
    通道注意力模块 (Channel Attention Block)
    实现通道间的注意力机制，增强重要通道的特征表示
    """
    def __init__(self):
        super(CAB, self).__init__()
        # 可学习参数beta，初始化为0，训练中自动更新
        self.beta = nn.Parameter(torch.tensor(0.0))
    def forward(self, A):
        """
        通道注意力前向传播
        Args:
            A (torch.Tensor): 原始补丁特征，形状为 (M, C)
                             M = H * W 是补丁总数，C 是通道数
        
        Returns:
            E (torch.Tensor): 融合通道注意力后的特征，形状为 (M, C)
            CAG (torch.Tensor): 通道注意力图，形状为 (C, C)
        """
        # 获取维度信息
        M, C = A.shape
        # 步骤1: 计算通道注意力图 CAG
        # 公式4: CAG = softmax(A^T @ A)
        # 转置A得到pmt_A: (M, C) -> (C, M)
        pmt_A = A.t()  # pmt_A: (C, M)
        # 计算交互: pmt_A @ A = (C, M) @ (M, C) = (C, C)
        interaction = torch.matmul(pmt_A, A)  # interaction: (C, C)
        # 通过softmax归一化得到CAG
        CAG = F.softmax(interaction, dim=-1)  # CAG: (C, C)
        # 步骤2: 计算融合特征E
        # 公式5: E = A + beta * (CAG @ A^T)^T
        # 转置A得到D: (M, C) -> (C, M)
        D = A.t()  # D: (C, M)
        # CAG @ D = (C, C) @ (C, M) = (C, M)
        temp = torch.matmul(CAG, D)  # temp: (C, M)
        # 转置结果: (C, M) -> (M, C)
        temp_t = temp.t()  # temp_t: (M, C)
        # 与beta相乘后与原始A相加
        E = A + self.beta * temp_t  # E: (M, C)
        return E, CAG

    def extra_repr(self):
        return f'temperature={self.temperature}'


class CAB(nn.Module):
    """
    通道注意力模块 (Channel Attention Block)
    实现通道间的注意力机制，增强重要通道的特征表示
    """
    
    def __init__(self):
        super(CAB, self).__init__()
        # 可学习参数beta，初始化为0，训练中自动更新
        self.beta = nn.Parameter(torch.tensor(0.0))
    def forward(self, A):
        """
        通道注意力前向传播
        
        Args:
            A (torch.Tensor): 原始补丁特征，形状为 (M, C)
                             M = H * W 是补丁总数，C 是通道数
        
        Returns:
            E (torch.Tensor): 融合通道注意力后的特征，形状为 (M, C)
            CAG (torch.Tensor): 通道注意力图，形状为 (C, C)
        """
        # 获取维度信息
        M, C = A.shape
        # 步骤1: 计算通道注意力图 CAG
        # 公式4: CAG = softmax(A^T @ A)
        # 转置A得到pmt_A: (M, C) -> (C, M)
        pmt_A = A.t()  # pmt_A: (C, M)
        # 计算交互: pmt_A @ A = (C, M) @ (M, C) = (C, C)
        interaction = torch.matmul(pmt_A, A)  # interaction: (C, C)
        # 通过softmax归一化得到CAG
        CAG = F.softmax(interaction, dim=-1)  # CAG: (C, C)
        # 步骤2: 计算融合特征E
        # 公式5: E = A + beta * (CAG @ A^T)^T
        # 转置A得到D: (M, C) -> (C, M)
        D = A.t()  # D: (C, M)
        # CAG @ D = (C, C) @ (C, M) = (C, M)
        temp = torch.matmul(CAG, D)  # temp: (C, M)
        # 转置结果: (C, M) -> (M, C)
        temp_t = temp.t()  # temp_t: (M, C)
        # 与beta相乘后与原始A相加
        E = A + self.beta * temp_t  # E: (M, C)
        return E, CAG

    def extra_repr(self):
        return f'beta={self.beta.item():.4f}'


class DualAttention(nn.Module):
    """
    双注意力模块 (Dual Attention Block)
    结合位置注意力模块(PAB)和通道注意力模块(CAB)，实现空间和通道维度的双重注意力机制
    """
    
    def __init__(self, temperature=1.0):
        super(DualAttention, self).__init__()
        # 实例化位置注意力模块
        self.pab = PAB(temperature=temperature)
        # 实例化通道注意力模块
        self.cab = CAB()
    
    def forward(self, A):
        """
        双注意力前向传播
        
        Args:
            A (torch.Tensor): 输入特征，形状为 (M, C)
                             M = H * W 是补丁总数，C 是通道数
        
        Returns:
            output (torch.Tensor): 双注意力融合后的特征，形状为 (M, C)
                                   output = E_p + E_c，其中E_p是PAB输出，E_c是CAB输出
            E_p (torch.Tensor): 位置注意力模块输出，形状为 (M, C)
            E_c (torch.Tensor): 通道注意力模块输出，形状为 (M, C)
            SAG (torch.Tensor): 空间注意力图，形状为 (M, M)
            CAG (torch.Tensor): 通道注意力图，形状为 (C, C)
        """
        # 位置注意力模块前向传播
        E_p, SAG = self.pab(A)
        
        # 通道注意力模块前向传播
        E_c, CAG = self.cab(A)
        
        # 双注意力融合：output = E_p + E_c
        output = E_p + E_c
        
        return output, E_p, E_c, SAG, CAG
    
    def extra_repr(self):
        return f'PAB: {self.pab.extra_repr()}, CAB: {self.cab.extra_repr()}'


# 测试代码
if __name__ == "__main__":
    # 测试PAB模块
    batch_size = 2
    M = 49  # 7x7 patches
    C = 2048  # feature channels
    
    # 创建测试数据
    A = torch.randn(M, C)
    alpha = torch.tensor(0.5, device=A.device, dtype=A.dtype)
    
    # 创建PAB实例
    pab = PAB(temperature=1.0)
    
    # 计算增强特征和SAG
    E, sag = pab(A, alpha=alpha)
    
    print(f"输入特征A形状: {A.shape}")
    print(f"增强特征E形状: {E.shape}")
    print(f"输出SAG形状: {sag.shape}")
    print(f"SAG行和验证: {torch.sum(sag, dim=1)}")  # 应该接近1
    
    # 验证数值稳定性
    print(f"SAG最小值: {torch.min(sag)}")
    print(f"SAG最大值: {torch.max(sag)}")
    
    # 测试批次处理
    A_batch = torch.randn(batch_size, M, C)
    alpha_batch = torch.tensor(0.5, device=A_batch.device, dtype=A_batch.dtype)
    # 对批次中的每个样本单独计算
    E_batch, sag_batch = zip(*[pab(A_batch[i], alpha=alpha_batch) for i in range(batch_size)])
    E_batch = torch.stack(E_batch)
    sag_batch = torch.stack(sag_batch)
    print(f"批次增强特征E形状: {E_batch.shape}")
    print(f"批次SAG形状: {sag_batch.shape}")

    # Test CAB class
    print("\n" + "="*50)
    print("Testing CAB (Channel Attention Block)")
    print("="*50)
    
    cab_model = CAB()
    print(f"CAB initial beta: {cab_model.beta.item():.4f}")
    
    # 测试CAB
    E_cab, CAG = cab_model(A)
    print("CAB enhanced features shape:", E_cab.shape)
    print("CAB channel attention graph shape:", CAG.shape)
    print("CAG row sum verification:", torch.sum(CAG, dim=1))  # 应该接近1
    
    # 测试批次处理
    print("\nTesting CAB with batch processing...")
    E_cab_batch, CAG_batch = zip(*[cab_model(A_batch[i]) for i in range(batch_size)])
    E_cab_batch = torch.stack(E_cab_batch)
    CAG_batch = torch.stack(CAG_batch)
    print(f"批次CAB增强特征形状: {E_cab_batch.shape}")
    print(f"批次CAG形状: {CAG_batch.shape}")
    
    # 测试DualAttention类
    print("\n" + "="*50)
    print("Testing DualAttention (Dual Attention Block)")
    print("="*50)
    
    dual_attention = DualAttention(temperature=1.0)
    print(f"DualAttention: {dual_attention}")
    
    # 测试双注意力模块
    output, E_p, E_c, SAG, CAG = dual_attention(A)
    print("DualAttention output shape:", output.shape)
    print("PAB output shape:", E_p.shape)
    print("CAB output shape:", E_c.shape)
    print("SAG shape:", SAG.shape)
    print("CAG shape:", CAG.shape)
    
    # 验证双注意力融合公式
    expected_output = E_p + E_c
    print("DualAttention fusion verification:", torch.allclose(output, expected_output))
    
    # 测试批次处理
    print("\nTesting DualAttention with batch processing...")
    output_batch, E_p_batch, E_c_batch, SAG_batch, CAG_batch = zip(*[dual_attention(A_batch[i]) for i in range(batch_size)])
    output_batch = torch.stack(output_batch)
    E_p_batch = torch.stack(E_p_batch)
    E_c_batch = torch.stack(E_c_batch)
    SAG_batch = torch.stack(SAG_batch)
    CAG_batch = torch.stack(CAG_batch)
    print(f"批次DualAttention输出形状: {output_batch.shape}")
    print(f"批次PAB输出形状: {E_p_batch.shape}")
    print(f"批次CAB输出形状: {E_c_batch.shape}")
    print(f"批次SAG形状: {SAG_batch.shape}")
    print(f"批次CAG形状: {CAG_batch.shape}")