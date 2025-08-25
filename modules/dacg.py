import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from modules.Context_guidance_normalization import CNL
from modules.DualAttentionBlock import DualAttention
from modules.GM_Generator import GMG
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class FeedForward(nn.Module):
    """前馈网络"""
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    """编码器层 - 实现公式(17)"""
    def __init__(self, d_model, num_heads=8, d_ff=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        # 使用PyTorch原生的MultiheadAttention
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接 + 层归一化
        # PyTorch MultiheadAttention的mask参数是attn_mask，需要是(L, S)或(N*num_heads, L, S)
        # 这里的mask是(B, 1, T, T)或(B, T, T)，需要调整
        # 如果mask是布尔类型，需要转换为float类型，True表示masked
        if mask is not None:
            # 将(B, 1, T, T)或(B, T, T)转换为(T, T)并重复num_heads次，或者直接传递(T, T)
            # 或者更简单地，如果mask是布尔类型，直接传递给attn_mask
            # PyTorch 2.0+ 支持直接传递 (B, T, T) 形状的布尔掩码
            # 对于旧版本，需要转换为(T, T)并扩展
            if mask.dim() == 4: # (B, 1, T, T)
                mask = mask.squeeze(1) # (B, T, T)
            # attn_mask 期望 (L, S) 或 (N*num_heads, L, S)
            # 如果是因果掩码，通常是 (T, T)
            # PyTorch MultiheadAttention的attn_mask是加性掩码，True表示不关注
            # 我们的mask是0表示masked，所以需要反转
            attn_mask = (mask == 0) if mask.dtype == torch.bool else mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            # 对于batch_first=True, attn_mask 应该是 (N, L, S) 或 (L, S)
            # 如果是因果掩码，通常是 (L, S)
            # 这里假设mask是因果掩码，且形状为 (T, T)
            # 如果是batch_first=True，query, key, value 形状是 (N, L, E)
            # attn_mask 形状是 (L, S) 或 (N*num_heads, L, S)
            # 对于自注意力，L=S=seq_len
            # 如果是布尔掩码，True表示忽略
            # 我们的mask是0表示忽略，所以需要反转
            if mask.dtype == torch.bool:
                attn_mask = ~mask # True表示忽略
            else: # float mask
                attn_mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        else:
            attn_mask = None
        
        attn_output, _ = self.self_attention(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class Encoder(nn.Module):
    """编码器 - 集成双重注意力模块处理视觉特征"""
    def __init__(self, d_model=512, num_layers=1, num_heads=8, d_ff=2048, dropout=0.1, input_dim=2048):
        super(Encoder, self).__init__()
        self.d_model = d_model
        
        # 双重注意力模块 - 处理视觉特征
        self.dual_attention = DualAttention(temperature=1.0)
        # 输入投影层，将双重注意力输出映射到d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        # 编码器层堆叠
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        
    def forward(self, src):
        """
        前向传播 - 集成双重注意力处理
        
        Args:
            src: 输入特征，形状 (B, M, C)
                B: 批次大小，M: 补丁总数，C: 特征通道数
        
        Returns:
            隐藏状态，形状 (B, M, d_model)
                M: 隐藏状态序列长度，d_model: 隐藏维度
        """
        batch_size, M, C = src.shape
        # 双重注意力处理 - 对批次中的每个样本分别处理
        enhanced_features = []
        for i in range(batch_size):
            sample_features = src[i]  # (M, C)
            enhanced_sample = self.dual_attention(sample_features)[0]  # (M, C)
            enhanced_features.append(enhanced_sample)
        # 合并批次
        enhanced_features = torch.stack(enhanced_features)  # (B, M, C)
        # 输入投影
        x = self.input_projection(enhanced_features)  # [batch_size, M, d_model]
        # 通过所有编码器层
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderLayer(nn.Module):
    """解码器层 - 按照架构图实现"""
    def __init__(self, d_model, num_heads=8, d_ff=2048, dropout=0.1, H=49, context_dim=2048, vocab_size=10000):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        
        # Masked Multi-Head Attention 作为独立模块
        self.masked_mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        # Multi-Head Attention (交叉注意力)
        self.cross_mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        # Feed Forward Network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        # 上下文引导归一化层 - 按照架构图位置配置
        # 使用d_model作为context_dim，因为GM_t是基于编码后的memory（d_model维）初始化的
        self.cnl_1 = CNL(d_model, H=H, context_dim=d_model)  # Masked MHA 输入前
        self.cnl_2 = CNL(d_model, H=H, context_dim=d_model)  # Multi-Head Attention 输入后的CNL
        self.cnl_3 = CNL(d_model, H=H, context_dim=d_model)  # 最终输出前
        
        # 全连接层和Softmax层（Output Embedding）
        if vocab_size is not None:
            self.fc_out = nn.Linear(d_model, vocab_size)
            self.vocab_size = vocab_size
        else:
            self.fc_out = None
            self.vocab_size = None
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, GM_t, tgt_mask=True, memory_mask=None, key_padding_mask=None, memory_key_padding_mask=None, return_logits=False):
        """
        前向传播 - 每一层都包含完整的输出层

        Args:
            tgt: 目标序列嵌入，形状 (B, T, d_model)
            memory: 编码器隐藏状态，形状 (B, S, d_model)
            GM_t: 当前时间步的引导记忆，形状 (H, D)
            tgt_mask: 目标序列掩码
            memory_mask: 记忆掩码
            return_logits: 是否返回logits而非概率分布（仅当vocab_size设置时生效）

        Returns:
            如果vocab_size已设置，返回词预测概率分布，形状 (B, T, Vocab_Size)
            如果return_logits=True，返回logits，形状 (B, T, Vocab_Size)
            否则返回隐藏状态，形状 (B, T, d_model)
        """
        
        # 处理输入 - 确保tgt是嵌入后的输入
        if tgt.dim() == 2:  # 如果是token索引 (B, T)
            # 注意：DecoderLayer不应该直接处理token索引
            # 这应该由Decoder类处理
            raise ValueError("DecoderLayer期望接收嵌入后的输入，而不是token索引")
        
        # 维度检查
        batch_size, seq_len, d_model = tgt.shape
        assert d_model == self.d_model, f"输入维度{d_model}与模型维度{self.d_model}不匹配"
        assert memory.size(0) == batch_size, f"批次大小不匹配: {memory.size(0)} vs {batch_size}"
        
        # Masked Multi-Head Attention
        # PyTorch MultiheadAttention的attn_mask是加性掩码，True表示忽略
        # 我们的tgt_mask是0表示masked，所以需要反转
        if tgt_mask is not None:
            if tgt_mask.dtype == torch.bool:
                attn_mask_tgt = ~tgt_mask # True表示忽略
            else: # float mask
                attn_mask_tgt = tgt_mask.masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        else:
            attn_mask_tgt = None
        
        masked_attn, _ = self.masked_mha(tgt, tgt, tgt, attn_mask=attn_mask_tgt, key_padding_mask=key_padding_mask)
        tgt = tgt + self.dropout(masked_attn)
        # Multi-Head Attention 输入前的 CNL
        tgt_norm = self.cnl_1(GM_t, tgt)
        
        # Multi-Head Attention (交叉注意力)
        # 我们的memory_mask是0表示masked，所以需要反转
        if memory_mask is not None:
            if memory_mask.dtype == torch.bool:
                attn_mask_mem = ~memory_mask # True表示忽略
            else: # float mask
                attn_mask_mem = memory_mask.masked_fill(memory_mask == 0, float('-inf')).masked_fill(memory_mask == 1, float(0.0))
        else:
            attn_mask_mem = None

        cross_attn, _ = self.cross_mha(tgt_norm, memory, memory, attn_mask=attn_mask_mem, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(cross_attn)
        tgt_norm = self.cnl_2(GM_t, tgt)
        # Feed Forward Network
        ff_output = self.feed_forward(tgt)
        tgt = tgt + self.dropout(ff_output)
        # 最终输出前的 CNL
        tgt = self.cnl_3(GM_t, tgt)
        
        # 全连接层和Softmax层（Output Embedding）
        if self.fc_out is not None:
            logits = self.fc_out(tgt)  # (B, T, Vocab_Size)
            # 改进A: DecoderLayer 一律返回 logits，避免在层级混用logits和概率
            return logits
        return tgt  # 无输出层时返回隐藏状态


class Decoder(nn.Module):
    """解码器 - 按照架构图实现"""
    def __init__(self, vocab_size, d_model=512, num_layers=1, num_heads=8, 
                 d_ff=2048, dropout=0.1, max_seq_length=100, H=49, context_dim=2048):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # 解码器层堆叠（只有最后一层包含输出层）
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, H, context_dim, 
                        vocab_size=vocab_size if i == num_layers - 1 else None)
            for i in range(num_layers)
        ])
        
        # 存储特殊token的ID，用于采样
        self.bos_id = None
        self.eos_id = None
        self.pad_id = None
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, GM_t, tgt_mask=None):
        """
        前向传播 - 每一层都包含完整的输出层（全连接层+Softmax）
        
        Args:
            tgt: 文本嵌入序列，形状 (B, t-1) 或 (B, t-1, D)
            memory: 编码器隐藏状态，形状 (B, S, D)
            GM_t: 当前时间步的引导记忆，形状 (H, D)
            tgt_mask: 目标序列掩码
        
        Returns:
            词预测概率分布（Output Embedding），形状 (B, T, Vocab_Size)
            由最后一层DecoderLayer产生的经过softmax归一化的概率分布
        """
        # 处理输入
        tokens = None
        if tgt.dim() == 2:  # 如果是token索引
            tokens = tgt  # (B, T)
            tgt = self.embedding(tgt)
        
        # 添加位置编码
        tgt = self.pos_encoding(tgt.transpose(0, 1)).transpose(0, 1)
        tgt = self.dropout(tgt)
        
        # 生成 key_padding_mask（B, T），True表示需要屏蔽
        key_padding_mask_local = None
        if tokens is not None and hasattr(self, "pad_id") and self.pad_id is not None:
            key_padding_mask_local = (tokens == self.pad_id)
        
        # 生成因果掩码 attn_mask（T, T），True表示需要屏蔽
        B, T, _ = tgt.shape
        device = tgt.device
        if tgt_mask is None:
            attn_mask_local = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        else:
            if tgt_mask.dtype == torch.bool:
                # 若传入的是下三角True表示可见，需要取反为屏蔽位(True)
                attn_mask_local = ~tgt_mask
            else:
                attn_mask_local = (tgt_mask == 0)
        
        # 通过所有解码器层（只有最后一层包含输出层）
        x = tgt
        layer_outputs = []
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:  # 最后一层
                # 最后一层返回logits
                output = layer(x, memory, GM_t, tgt_mask=attn_mask_local, key_padding_mask=key_padding_mask_local)
            else:  # 中间层
                # 中间层返回隐藏状态（嵌入维度）
                output = layer(x, memory, GM_t, tgt_mask=attn_mask_local, key_padding_mask=key_padding_mask_local, return_logits=False)
            layer_outputs.append(output)
            # 对于中间层，保持嵌入维度；最后一层返回概率分布
            if i < len(self.layers) - 1:
                x = output  # 保持 [B, T, d_model]
            else:
                x = output  # 最后一层返回 [B, T, Vocab_Size]
        
        return x  # 最后一层的概率分布


class EncoderDecoder(nn.Module):
    """完整的编码器-解码器架构"""
    def __init__(self, args, tokenizer):
        super(EncoderDecoder, self).__init__()
        
        # 从args中提取参数
        vocab_size = tokenizer.get_vocab_size() if hasattr(tokenizer, 'get_vocab_size') else len(tokenizer)
        d_model = args.d_model if hasattr(args, 'd_model') else 512
        num_layers = args.num_layers if hasattr(args, 'num_layers') else 6
        num_heads = args.num_heads if hasattr(args, 'num_heads') else 8
        d_ff = args.d_ff if hasattr(args, 'd_ff') else 2048
        dropout = args.dropout if hasattr(args, 'dropout') else 0.1
        max_seq_length = args.max_seq_length if hasattr(args, 'max_seq_length') else 100
        H = args.H if hasattr(args, 'H') else 49
        
        # 保存特殊token的ID，避免硬编码
        self.bos_id = tokenizer.token2idx[tokenizer.BOS_TOKEN]
        self.eos_id = tokenizer.token2idx[tokenizer.EOS_TOKEN]
        self.pad_id = tokenizer.token2idx[tokenizer.PAD_TOKEN]
        
        self.encoder = Encoder(d_model, num_layers, num_heads, d_ff, dropout, 
                             input_dim=getattr(args, 'visual_feat_dim', 2048))
        self.decoder = Decoder(vocab_size, d_model, num_layers, num_heads, 
                             d_ff, dropout, max_seq_length, H, 
                             context_dim=getattr(args, 'visual_feat_dim', 2048))
        
        #将特殊token的ID也传递给decoder
        self.decoder.bos_id = self.bos_id
        self.decoder.eos_id = self.eos_id
        self.decoder.pad_id = self.pad_id
        
        # 初始化GM生成器
        self.gmg = GMG(hidden_dim=d_model, num_heads=8, dropout=dropout)
        
    def forward(self, fc_feats, att_feats, targets=None, mode='forward', GM_t=None):
        """
        完整的前向传播 - 兼容现有接口
        
        Args:
            fc_feats: 全局特征，形状 (B, C)
            att_feats: 注意力特征，形状 (B, M, C)
            targets: 目标序列，形状 (B, t-1)
            mode: 'forward' 或 'sample'
            GM_t: 初始引导记忆，形状 (H, D)。如果为None，将使用平均特征初始化
        
        Returns:
            预测结果，形状 (B, 1, Vocab_Size)
        """
        batch_size = att_feats.size(0)
        device = att_feats.device
        
        # 编码阶段 - 公式(17)
        memory = self.encoder(att_feats)  # (B, M, D)
        H, D = memory.size(1), memory.size(2)
        
        # 获取维度信息
        H, D = memory.size(1), memory.size(2)
        
        if mode == 'forward':
            # 训练模式
            assert targets is not None, "训练模式下需要提供targets"
            
            # 初始化引导记忆
            if GM_t is None:
                # 使用编码后的维度，先对批次和序列维度取平均
                GM_t = memory.mean(dim=(0, 1))  # (D,)
                GM_t = GM_t.unsqueeze(0).expand(H, -1)  # (H, D)
            else:
                GM_t = GM_t.to(device)
            
            # 获取序列长度
            seq_len = targets.size(1)
            
            # 处理每个时间步
            outputs = []
            for t in range(seq_len):
                # 获取当前时间步的输入
                current_input = targets[:, :t+1]  # (B, t+1)
                
                # 创建因果掩码
                current_seq_len = current_input.size(1)
                tgt_mask = torch.tril(torch.ones(current_seq_len, current_seq_len, device=device))
                
                # 使用当前GM_t进行解码（传入token索引，由decoder处理嵌入）
                current_output = self.decoder(current_input, memory, GM_t, tgt_mask)
                
                # 获取最后一个时间步的输出
                current_pred = current_output[:, -1:, :]  # (B, 1, Vocab_Size)
                outputs.append(current_pred)
                
                # 更新引导记忆（使用当前预测的嵌入表示作为y_prev）
                if t < seq_len - 1:  # 不需要为最后一个时间步更新
                    # 获取预测的token嵌入
                    predicted_tokens = current_pred.argmax(dim=-1)  # (B, 1)
                    predicted_embed = self.decoder.embedding(predicted_tokens)  # (B, 1, D)
                    y_prev = predicted_embed.mean(dim=0).mean(dim=0, keepdim=True)  # (1, D)
                    
                    # 更新GM_t
                    GM_t, _ = self.gmg(GM_t, y_prev)
            
            # 合并所有时间步的输出
            final_output = torch.cat(outputs, dim=1)  # (B, seq_len, Vocab_Size)
            
            # 改进C: 返回完整序列的预测结果，而不仅是最后一步
            # 这样可以对整个序列计算损失，更好地训练模型
            return final_output  # (B, seq_len, Vocab_Size)
            
        elif mode == 'sample':
            # 推理模式 - 逐步生成
            
            # 初始化引导记忆
            if GM_t is None:
                # 使用编码后的维度，先对批次和序列维度取平均
                GM_t = memory.mean(dim=(0, 1))  # (D,)
                GM_t = GM_t.unsqueeze(0).expand(H, -1)  # (H, D)
            else:
                GM_t = GM_t.to(device)
            
            # 改进B: 初始化序列使用保存的BOS token ID
            seq = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=device)
            
            # 逐步生成序列
            max_seq_length = self.decoder.max_seq_length
            for i in range(max_seq_length - 1):
                # 创建掩码
                seq_len = seq.size(1)
                tgt_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
                
                # 使用当前GM_t进行解码
                output = self.decoder(seq, memory, GM_t, tgt_mask)
                
                # 获取下一个token（取最后一个时间步的预测）
                next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(1)
                seq = torch.cat([seq, next_token], dim=1)
                
                # 改进B: 使用保存的EOS token ID而不是硬编码的2
                if (next_token == self.eos_id).all():  # 检查是否所有序列都生成了EOS token
                    break
                
                # 更新引导记忆（使用当前预测的嵌入表示作为y_prev）
                next_embed = self.decoder.embedding(next_token)  # (B, 1, D)
                y_prev = next_embed.mean(dim=0).mean(dim=0, keepdim=True)  # (1, D)
                GM_t, _ = self.gmg(GM_t, y_prev)
            
            return output, seq
        else:
            raise ValueError("mode必须是'forward'或'sample'")


if __name__ == "__main__":
    # 测试代码
    import types
    
    class MockTokenizer:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
            # 模拟特殊token和它们的ID
            self.BOS_TOKEN = "<bos>"
            self.EOS_TOKEN = "<eos>"
            self.PAD_TOKEN = "<pad>"
            self.token2idx = {
                self.BOS_TOKEN: 0,
                self.EOS_TOKEN: 1,
                self.PAD_TOKEN: 2,
                # 假设其他token从3开始
                "word1": 3,
                "word2": 4,
                # ...
            }
            # 确保vocab_size至少包含这些特殊token
            if vocab_size < len(self.token2idx):
                self.vocab_size = len(self.token2idx)
        
        def get_vocab_size(self):
            return self.vocab_size
    
    class MockArgs:
        def __init__(self):
            self.d_model = 512
            self.num_layers = 1
            self.num_heads = 8
            self.d_ff = 2048
            self.dropout = 0.1
            self.max_seq_length = 100
            self.H = 49
    
    vocab_size = 10000
    M = 49  # 补丁数量 (7x7)
    batch_size = 4
    seq_len = 10  # 使用正确的序列长度，不超过max_seq_length=100
    
    # 创建模拟参数和tokenizer
    args = MockArgs()
    tokenizer = MockTokenizer(vocab_size)
    
    # 创建模型
    model = EncoderDecoder(args, tokenizer)
    
    # 验证模型参数
    assert model.decoder.max_seq_length >= seq_len, f"序列长度{seq_len}超过最大限制{model.decoder.max_seq_length}"
    
    # 测试数据 - 确保维度正确
    fc_feats = torch.randn(batch_size, 512)  # 全局特征
    att_feats = torch.randn(batch_size, M, 2048)  # 注意力特征
    targets = torch.randint(1, vocab_size-1, (batch_size, seq_len))  # 文本序列，避免特殊token
    
    # 验证输入维度
    print(f"测试配置: batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}")
    print(f"输入维度检查:")
    print(f"  fc_feats: {fc_feats.shape}")
    print(f"  att_feats: {att_feats.shape}")
    print(f"  targets: {targets.shape}")
    
    try:
        # 测试训练模式
        print("开始训练模式测试...")
        output = model(fc_feats, att_feats, targets, mode='forward')
        
        print("编码器-解码器测试:")
        print(f"全局特征形状: {fc_feats.shape}")
        print(f"注意力特征形状: {att_feats.shape}")
        print(f"目标序列形状: {targets.shape}")
        print(f"输出预测形状: {output.shape}")
        
        # 验证输出
        expected_shape = (batch_size, 1, vocab_size)
        expected_shape = (batch_size, seq_len, vocab_size) # 更新为期望的完整序列输出形状
        assert output.shape == expected_shape, f"输出形状错误: 期望{expected_shape}, 实际{output.shape}"
        
        # 测试推理模式
        print("开始推理模式测试...")
        output_sample, seq = model(fc_feats, att_feats, mode='sample')
        print(f"采样模式输出形状: {output_sample.shape}")
        print(f"生成序列形状: {seq.shape}")
        
        # 测试梯度
        print("测试梯度计算...")
        loss = output.sum()
        loss.backward()
        
        # 验证梯度存在
        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"警告: {name} 没有梯度")
        
        print("✓ 所有基础测试通过！")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

    # ===== 测试GM生成器与解码器层集成 =====
    print("\n" + "="*50)
    print("测试GM生成器与解码器层集成")
    print("="*50)
    
    # 创建集成测试专用模型
    test_model = EncoderDecoder(args, tokenizer)
    
    # 测试GM生成器动态更新功能
    print("\n1. 测试GM生成器动态更新功能...")
    
    # 初始化引导记忆
    initial_gm = torch.randn(args.H, args.d_model)
    
    # 测试训练模式下的GM更新
    test_targets = torch.randint(0, vocab_size, (batch_size, 5))  # 短序列用于测试
    
    # 记录初始GM状态
    initial_gm_sum = initial_gm.sum().item()
    
    # 前向传播
    with torch.no_grad():
        test_output = test_model(fc_feats, att_feats, test_targets, mode='forward', GM_t=initial_gm)
    
    print(f"   初始GM总和: {initial_gm_sum:.4f}")
    print(f"   训练模式输出形状: {test_output.shape}")
    # 更新为期望的完整序列输出形状
    expected_test_output_shape = (batch_size, test_targets.size(1), vocab_size)
    assert test_output.shape == expected_test_output_shape, "训练模式输出形状错误"
    
    # 测试推理模式下的GM更新
    print("\n2. 测试推理模式下的GM更新...")
    
    with torch.no_grad():
        sample_output, generated_seq = test_model(fc_feats, att_feats, mode='sample', GM_t=initial_gm)
    
    print(f"   推理模式输出形状: {sample_output.shape}")
    print(f"   生成序列形状: {generated_seq.shape}")
    print(f"   生成序列示例: {generated_seq[0, :5].tolist()}")
    
    # 测试GM生成器参数更新
    print("\n3. 测试GM生成器参数更新...")
    
    # 确保GM生成器参数参与训练
    gm_params = list(test_model.gmg.parameters())
    print(f"   GM生成器参数数量: {len(gm_params)}")
    print(f"   第一个参数形状: {gm_params[0].shape}")
    
    # 测试前向传播和反向传播
    test_model.train()
    output_with_gm = test_model(fc_feats, att_feats, test_targets, mode='forward')
    loss = output_with_gm.sum()
    loss.backward()
    
    # 检查梯度
    has_grad = any(param.grad is not None for param in test_model.gmg.parameters())
    print(f"   GM生成器是否有梯度: {has_grad}")
    if has_grad:
        grad_norm = torch.norm(next(test_model.gmg.parameters()).grad).item()
        print(f"   梯度范数: {grad_norm:.6f}")
    
    # 测试不同批次大小
    print("\n4. 测试不同批次大小兼容性...")
    
    for test_batch in [1, 2, 4]:
        test_fc = torch.randn(test_batch, 512)
        test_att = torch.randn(test_batch, M, 2048)
        test_tgt = torch.randint(0, vocab_size, (test_batch, 3))
        
        with torch.no_grad():
            test_out = test_model(test_fc, test_att, test_tgt, mode='forward')
        
        assert test_out.shape[0] == test_batch, f"批次{test_batch}处理失败"
        print(f"   批次大小{test_batch}: 成功")
    
    # 测试GM_t维度验证
    print("\n5. 测试GM_t维度验证...")
    
    # 测试错误维度处理
    try:
        wrong_gm = torch.randn(args.H//2, args.d_model)  # 错误的高度
        test_model(fc_feats, att_feats, test_targets, mode='forward', GM_t=wrong_gm)
        print("   错误: 应该抛出维度不匹配异常")
    except Exception as e:
        print(f"   正确捕获维度错误: {type(e).__name__}")
    
    # 测试无GM_t输入（使用默认初始化）
    print("\n6. 测试无GM_t输入...")
    
    with torch.no_grad():
        default_output = test_model(fc_feats, att_feats, test_targets, mode='forward')
    
    print(f"   默认初始化输出形状: {default_output.shape}")
    # 更新为期望的完整序列输出形状
    expected_default_output_shape = (batch_size, test_targets.size(1), vocab_size)
    assert default_output.shape == expected_default_output_shape, "默认初始化失败"
    
    print("\n" + "="*50)
    print("所有GM集成测试通过！")
    print("="*50)