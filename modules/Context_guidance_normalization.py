import torch
import torch.nn as nn
import torch.nn.functional as F


class CNL(nn.Module):
    """
    上下文引导归一化层 (Context-guided Normalization Layer)
    基于引导记忆(GM)动态调整LayerNorm的参数，实现上下文自适应归一化
    """
    
    def __init__(self, hidden_dim=512, H=49, mlp_ratio=4, dropout=0.1, context_dim=None):
        super(CNL, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim or hidden_dim
        self.H = H
        
        # 如果上下文维度与隐藏维度不同，先投影到隐藏维度
        if self.context_dim != self.hidden_dim:
            self.context_proj = nn.Linear(self.context_dim, self.hidden_dim)
        else:
            self.context_proj = None
        
        # 原始可学习参数 γ 和 β，形状 (D,)
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))
        
        # MLP用于处理展开的GM_t，输出 Δγ 和 Δβ
        # 输入维度: H*hidden_dim (展开的GM_t)
        # 输出维度: 2*hidden_dim (Δγ 和 Δβ 拼接)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * self.H, self.hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * mlp_ratio, self.hidden_dim * 2),  # 输出 Δγ 和 Δβ
            nn.Dropout(dropout)
        )
        
        # 层归一化（用于内部计算）
        self.layer_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        
    def forward(self, GM_t, prev_output):
        """
        上下文引导归一化前向传播
        
        Args:
            GM_t (torch.Tensor): 当前时间步的引导记忆，形状 (H, D)
                                H 是 GM 行数，D 是隐藏维度
            prev_output (torch.Tensor): 前一模块的输出，形状 (B, S, D)
                                       B 是 batch 大小，S 是序列长度，D 是隐藏维度
        
        Returns:
            cnl_output (torch.Tensor): 归一化后的结果，形状 (B, S, D)
        """
        B, S, D = prev_output.shape
        H, D_GM = GM_t.shape
        
        # 确保维度匹配
        assert H == self.H, f"GM_t的行数{H}必须与初始化时指定的H={self.H}匹配"
        
        # 如果上下文维度与隐藏维度不同，先投影
        if self.context_proj is not None:
            GM_t = self.context_proj(GM_t)  # (H, hidden_dim)
        
        # 步骤1: GM展开与MLP更新 γ*/β*（公式14-15）
        # 展开 GM_t 为 gm_t：拼接所有行，得到形状 (H*hidden_dim,) 的向量
        gm_t = GM_t.reshape(-1)  # 形状: (H*hidden_dim,)
        
        # 用MLP处理gm_t，得到增量 Δγ 和 Δβ
        mlp_output = self.mlp(gm_t)  # 形状: (2*D,)
        
        # 分离 Δγ 和 Δβ
        delta_gamma = mlp_output[:D]  # 形状: (D,)
        delta_beta = mlp_output[D:]   # 形状: (D,)
        
        # 更新后参数：γ_t* = γ + Δγ，β_t* = β + Δβ
        gamma_t_star = self.gamma + delta_gamma  # 形状: (D,)
        beta_t_star = self.beta + delta_beta    # 形状: (D,)
        
        # 步骤2: CNL归一化计算（公式16）
        # 对 prev_output 计算均值 μ 和标准差 s（按最后一维 D）
        # 使用layer_norm计算归一化，但不使用其仿射变换
        normalized = self.layer_norm(prev_output)  # 形状: (B, S, D)
        
        # 应用公式: f_CNL(l) = γ_t* ⊙ (l - μ)/s + β_t*
        # 其中 γ_t* 和 β_t* 需要广播到 (B, S, D)
        gamma_t_star = gamma_t_star.view(1, 1, -1)  # 形状: (1, 1, D)
        beta_t_star = beta_t_star.view(1, 1, -1)    # 形状: (1, 1, D)
        
        cnl_output = gamma_t_star * normalized + beta_t_star  # 形状: (B, S, D)
        
        return cnl_output


if __name__ == "__main__":
    # 测试CNL类
    hidden_dim = 512
    H = 49  # 7x7的patch特征
    B = 4   # batch大小
    S = 10  # 序列长度
    
    # 创建测试数据
    GM_t = torch.randn(H, hidden_dim)
    prev_output = torch.randn(B, S, hidden_dim)
    
    # 创建CNL实例
    cnl = CNL(hidden_dim=hidden_dim, H=H)
    
    # 前向传播
    cnl_output = cnl(GM_t, prev_output)
    
    print("CNL测试:")
    print(f"输入GM_t形状: {GM_t.shape}")
    print(f"输入prev_output形状: {prev_output.shape}")
    print(f"输出cnl_output形状: {cnl_output.shape}")
    
    # 验证形状
    assert cnl_output.shape == prev_output.shape, "输出形状不匹配"
    
    # 验证数值稳定性
    print(f"cnl_output均值: {cnl_output.mean().item():.4f}")
    print(f"cnl_output标准差: {cnl_output.std().item():.4f}")
    
    # 测试梯度
    cnl_output.sum().backward()
    for name, param in cnl.named_parameters():
        if param.grad is not None:
            print(f"{name} 梯度均值: {param.grad.mean().item():.6f}")
    
    # 测试不同batch大小
    B_test = [1, 2, 8]
    for b in B_test:
        prev_output_test = torch.randn(b, S, hidden_dim)
        cnl_output_test = cnl(GM_t, prev_output_test)
        print(f"Batch {b}: 输入形状 {prev_output_test.shape} -> 输出形状 {cnl_output_test.shape}")
    
    print("所有测试通过！")