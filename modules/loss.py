import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -torch.log(input.gather(2, target.long().unsqueeze(2))+1e-6).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


def compute_loss(output, reports_ids, reports_masks):
    # 改进A: 使用CrossEntropyLoss处理logits而非概率分布
    # output是logits, shape (B, L, V)
    # reports_ids是token IDs, shape (B, L+1), 因为包含<bos>token
    # reports_masks是掩码, shape (B, L+1)
    
    # 省略<bos>的target
    target = reports_ids[:, 1:]
    mask = reports_masks[:, 1:]
    
    # 确保的output和target大小匹配
    seq_len = min(output.size(1), target.size(1))
    output = output[:, :seq_len]
    target = target[:, :seq_len]
    mask = mask[:, :seq_len]
    
    # 使用CrossEntropyLoss计算损失
    criterion = nn.CrossEntropyLoss(reduction='none')
    # 将output从 (B, L, V) 转为 (B*L, V)
    logits_flat = output.contiguous().view(-1, output.size(-1))
    # 将target从 (B, L) 转为 (B*L)
    targets_flat = target.contiguous().view(-1).long()
    # 计算未加权的损失
    losses = criterion(logits_flat, targets_flat)
    # 将损失重新转换为 (B, L) 并应用掩码
    losses = losses.view(target.size(0), target.size(1))
    # 使用mask进行加权
    masked_losses = losses * mask
    # 求平均损失
    loss = masked_losses.sum() / mask.sum().clamp(min=1e-5)
    
    return loss