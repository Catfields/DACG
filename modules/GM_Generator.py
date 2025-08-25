import torch
import torch.nn as nn
import torch.nn.functional as F

class GMG(nn.Module):
    """
    引导记忆生成器 (Guided Memory Generator)
    实现基于多头注意力的引导记忆更新机制，用于序列到序列生成任务
    """
    
    def __init__(self, hidden_dim=512, num_heads=8, mlp_ratio=4, dropout=0.1, alpha=0.02):
        super(GMG, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.alpha = alpha # EMA更新速率
        # PyTorch原生多头注意力
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=False)
        # 用于将拼接后的K/V输入投影回hidden_dim
        self.kv_proj_k = nn.Linear(hidden_dim * 2, hidden_dim)
        self.kv_proj_v = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # MLP参数
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 门控机制参数
        self.V_f = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)
        self.V_i = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)
        self.J_f = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)
        self.J_i = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, GM_prev, y_prev):
        """
        引导记忆生成器前向传播
        
        Args:
            GM_prev (torch.Tensor): 前一时间步的引导记忆，形状 (H, D)
                                   H 是 GM 行数，D 是隐藏维度
            y_prev (torch.Tensor): 前一时间步的文本输出，形状 (1, D)
        
        Returns:
            GM_current (torch.Tensor): 当前时间步的引导记忆，形状 (H, D)
            attention_weights (torch.Tensor): 注意力权重，形状 (H, H)
        """
        H, D = GM_prev.shape
        
        # 扩展 y_prev 为 Y_{t-1}
        Y_prev = y_prev.repeat(H, 1)  # 形状: (H, D)
        
        # 步骤1: 准备多头注意力的 Q/K/V
        # Q: GM_prev (H, D)
        # K/V: 由 concat(GM_prev, Y_prev) 经过线性投影得到 (H, D)
        
        # 拼接 GM_prev 和 Y_prev 作为 K 和 V 的输入
        concat_input_kv = torch.cat([GM_prev, Y_prev], dim=1) # (H, 2D)
        
        # 线性投影 K 和 V
        K_mha = self.kv_proj_k(concat_input_kv) # (H, D)
        V_mha = self.kv_proj_v(concat_input_kv) # (H, D)
        
        # 调整形状以适应 nn.MultiheadAttention (L, N, E)
        # L=H (sequence length), N=1 (batch size), E=D (embedding dim)
        query = GM_prev.unsqueeze(1) # (H, 1, D)
        key = K_mha.unsqueeze(1) # (H, 1, D)
        value = V_mha.unsqueeze(1) # (H, 1, D)
        
        # 执行多头注意力
        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        
        # 恢复输出形状 (H, D)
        O = attn_output.squeeze(1) # (H, D)
        
        # 获取平均注意力权重 (H, H)
        avg_attention_weights = attn_output_weights.squeeze(0) # (H, H)
        
        # 步骤2: 残差连接与MLP（公式10）
        # GM_t^* = MLP(O + GM_{t-1}) + O + GM_{t-1}
        residual = O + GM_prev  # (H, D)
        normalized = self.norm1(residual)
        mlp_output = self.mlp(normalized)  # (H, D)
        GM_star = mlp_output + residual  # (H, D)
        GM_star = self.norm2(GM_star)
        
        # 步骤3: 门控机制（公式11-13）
        # 计算tanh(GM_{t-1})
        tanh_GM_prev = torch.tanh(GM_prev)  # (H, D)
        
        # 公式11: GATE_t^f = Y_{t-1} @ V^f + tanh(GM_{t-1}) @ J^f
        gate_f = torch.matmul(Y_prev, self.V_f) + torch.matmul(tanh_GM_prev, self.J_f)  # (H, D)
        
        # 公式12: GATE_t^i = Y_{t-1} @ V^i + tanh(GM_{t-1}) @ J^i
        gate_i = torch.matmul(Y_prev, self.V_i) + torch.matmul(tanh_GM_prev, self.J_i)  # (H, D)
        
        # 公式13: GM_t = sigmoid(GATE_t^f) * GM_{t-1} + sigmoid(GATE_t^i) * tanh(GM_t^*)
        gate_f_sigmoid = torch.sigmoid(gate_f)  # (H, D)
        gate_i_sigmoid = torch.sigmoid(gate_i)  # (H, D)
        tanh_GM_star = torch.tanh(GM_star)  # (H, D)
        
        # 原始的GM更新公式
        GM_target = gate_f_sigmoid * GM_prev + gate_i_sigmoid * tanh_GM_star  # (H, D)
        
        # 应用EMA更新
        GM_current = (1 - self.alpha) * GM_prev + self.alpha * GM_target # (H, D)
        
        return GM_current, avg_attention_weights


if __name__ == "__main__":
    # 测试GMG类
    hidden_dim = 512
    num_heads = 8
    H = 49  # 7x7的patch特征
    D = hidden_dim
    
    # 创建测试数据
    GM_prev = torch.randn(H, D)
    y_prev = torch.randn(1, D)
    
    # 创建GMG实例
    gmg = GMG(hidden_dim=hidden_dim, num_heads=num_heads)
    
    # 前向传播
    GM_current, attention_weights = gmg(GM_prev, y_prev)
    
    print("GMG测试:")
    print(f"输入GM_prev形状: {GM_prev.shape}")
    print(f"输入y_prev形状: {y_prev.shape}")
    print(f"输出GM_current形状: {GM_current.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 验证形状
    assert GM_current.shape == GM_prev.shape, "输出形状不匹配"
    assert attention_weights.shape == (H, H), "注意力权重形状不匹配"
    
    # 验证数值稳定性
    print(f"GM_current均值: {GM_current.mean().item():.4f}")
    print(f"GM_current标准差: {GM_current.std().item():.4f}")
    print(f"注意力权重和: {attention_weights.sum(dim=-1).mean().item():.4f}")  # 应该接近1
    
    # 测试批次处理
    batch_size = 4
    GM_prev_batch = torch.randn(batch_size, H, D)
    y_prev_batch = torch.randn(batch_size, 1, D)
    
    GM_current_batch = []
    attention_weights_batch = []
    
    for i in range(batch_size):
        gm, att = gmg(GM_prev_batch[i], y_prev_batch[i])
        GM_current_batch.append(gm)
        attention_weights_batch.append(att)
    
    GM_current_batch = torch.stack(GM_current_batch)
    attention_weights_batch = torch.stack(attention_weights_batch)
    
    print(f"\n批次处理测试:")
    print(f"批次GM_prev形状: {GM_prev_batch.shape}")
    print(f"批次y_prev形状: {y_prev_batch.shape}")
    print(f"批次GM_current形状: {GM_current_batch.shape}")
    print(f"批次注意力权重形状: {attention_weights_batch.shape}")
    
    # 验证梯度
    GM_current.sum().backward()
    for name, param in gmg.named_parameters():
        if param.grad is not None:
            print(f"{name} 梯度均值: {param.grad.mean().item():.6f}")