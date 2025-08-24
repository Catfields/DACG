'''
Author: Catfield 2022211136@stu.hit.edu.cn
Date: 2025-08-13 09:53:39
LastEditors: Catfield 2022211136@stu.hit.edu.cn
LastEditTime: 2025-08-24 17:10:35
FilePath: /DACG/models/dacg.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
import numpy as np
from modules.visual_extractor import VisualExtractor
from modules.dacg import EncoderDecoder



class DACGModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(DACGModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            # 改进A: 返回生成的token序列，而非概率
            _, seq = self.encoder_decoder(fc_feats, att_feats, mode='sample')
            return seq  # <- 返回 token ids 序列而不是概率
        else:
            raise ValueError
        return output

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images.squeeze(1))
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            # 改进A: 返回生成的token序列，而非概率
            _, seq = self.encoder_decoder(fc_feats, att_feats, mode='sample')
            return seq  # <- 返回 token ids 序列而不是概率
        else:
            raise ValueError
        return output



