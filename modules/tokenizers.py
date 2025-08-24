import json
import re
import numpy as np
from collections import Counter


class Tokenizer(object):
    def __init__(self, args):
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name
        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
        else:
            self.clean_report = self.clean_report_mimic_cxr
        self.ann = json.loads(open(self.ann_path, 'r').read())
        
        # Define special tokens
        self.PAD_TOKEN = '<pad>'
        self.BOS_TOKEN = '<bos>' # Beginning of Sequence
        self.EOS_TOKEN = '<eos>' # End of Sequence
        self.UNK_TOKEN = '<unk>' # Unknown
        self.special_tokens = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        
        # Correctly create the vocabulary
        self.token2idx, self.idx2token = self.create_vocabulary()
        
        # 改进B: 存储特殊令牌的ID，便于其他组件使用
        self.pad_id = self.token2idx[self.PAD_TOKEN]
        self.bos_id = self.token2idx[self.BOS_TOKEN]
        self.eos_id = self.token2idx[self.EOS_TOKEN]
        self.unk_id = self.token2idx[self.UNK_TOKEN]

    def create_vocabulary(self):
        total_tokens = []
        for example in self.ann['train']:
            tokens = self.clean_report(example['report']).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        # Filter based on threshold, but do not add special tokens yet
        vocab_words = [k for k, v in counter.items() if v >= self.threshold]
        
        # Now, create the unified vocabulary starting with special tokens
        # The indices will be 0, 1, 2, 3...
        vocab = self.special_tokens + vocab_words
        vocab.sort() # Sorting is good practice for consistency
        
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx
            idx2token[idx] = token
            
        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                         replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                         .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report
    
    # --- The key fix is in __call__ and get_id_by_token ---
    def get_id_by_token(self, token):
        # Use .get() with a default value to gracefully handle unknown tokens
        return self.token2idx.get(token, self.token2idx[self.UNK_TOKEN])

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = [self.get_id_by_token(token) for token in tokens]
        # Prepend BOS and append EOS
        ids = [self.token2idx[self.BOS_TOKEN]] + ids + [self.token2idx[self.EOS_TOKEN]]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            # 确保idx是整数而不是列表
            if isinstance(idx, list):
                # 如果是列表，取第一个元素
                idx = idx[0] if idx else 0
                
            # Check for a valid token id, and stop decoding on EOS or PAD
            if idx in self.idx2token and idx != self.token2idx[self.PAD_TOKEN] and idx != self.token2idx[self.BOS_TOKEN] and idx != self.token2idx[self.EOS_TOKEN]:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            elif idx == self.token2idx[self.EOS_TOKEN] or idx == self.token2idx[self.PAD_TOKEN]:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            # Convert numpy array to list of integers
            if isinstance(ids, np.ndarray):
                ids = ids.tolist()
            # 确保ids是一维列表
            if ids and isinstance(ids[0], list):
                # 如果ids是二维列表，展平为一维列表
                ids = [item[0] if isinstance(item, list) and item else item for item in ids]
            out.append(self.decode(ids))
        return out