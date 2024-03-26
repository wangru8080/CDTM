#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   predict_cdtm.py
@Time    :   2024/02/19 09:50:52
@Author  :   wangru
@Version :   1.0
'''

# here put the import lib
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from torch.nn import functional as F
from models import CDTModel
from transformers import BertTokenizer
from script import BertUnDocTagTrainData
from torch.utils.data import DataLoader
from tqdm import tqdm
from trainer_cdtm import Trainer
torch.manual_seed(2024)

class Config(object):
    def __init__(self):
        self.pretrain_model_path = '/mnt/disk2/users/wangru/model_file/pretrained_PLM/pretrained_bert_base_chinese'
        self.doc_max_len = 128
        self.tag_max_len = 32 # 64
        self.per_gpu_batch_size = 256
        self.num_workers = 10
        self.train_mode = 'unsupervise'
        self.loss_fn = 'CLIPLoss' # ['CLIPLoss', 'InfoNCE']
        self.pooling_type = 'first_last_avg'
        # self.model_path = 'save/unsupervise/bsz-256-lr-3e-05-loss-InfoNCE/model-2.pth'
        # self.model_path = 'save/unsupervise/bsz-256-lr-3e-05-loss-CLIPLoss-v2/model-2.pth'
        # self.model_path = 'save/unsupervise/bsz-256-lr-0.00015000000000000001-loss-CLIPLoss-v3/model-1.pth'
        self.model_path = 'save/unsupervise/bsz-256-lr-0.0015-loss-CLIPLoss/model-8.pth'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data(args):
    train_path = 'data/doc_tag.csv'
    train_doc_list = []
    train_tag_list = []
    print('train dataset...')
    for line in tqdm(open(train_path, encoding='utf-8')):
        item = line.strip().split('\t')
        doc = item[1]
        tag = item[2]
        train_doc_list.append(doc)
        train_tag_list.append(tag)

    train_dataset = BertUnDocTagTrainData(args, doc_list=train_doc_list, tag_list=train_tag_list)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.per_gpu_batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  collate_fn=None)
    return train_dataloader

def get_token(args, doc, tag):
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)
    doc_token_dict = tokenizer(
        doc, 
        max_length=args.doc_max_len, 
        truncation=True, 
        padding='max_length',
        return_tensors='pt'
    )

    tag_token_dict = tokenizer(
        tag, 
        max_length=args.tag_max_len, 
        truncation=True, 
        padding='max_length',
        return_tensors='pt'
    )
    return doc_token_dict, tag_token_dict

if __name__ == '__main__':
    args = Config()

    model = CDTModel(args, pooling_type=args.pooling_type)
    trainer = Trainer(args, model)
    
    trainer.load_model(args.model_path)

    doc_token_dict, tag_token_dict = get_token(args, '你说的这两件事，缅甸政府第一件压根就做不到，不在它势力范围内。第二件它倒是想，问题是第一件如果做不到，第二件就免谈吧。|| 可以搞游客生意，但有两件事需要做好—— 1、提高治安水平 2、让中国修高铁', '缅甸柬埔寨来抢中国游客了')
    
    doc_ipt = {
        'input_ids': doc_token_dict['input_ids'].to(args.device), # [batch, max_len]
        'input_mask': doc_token_dict['attention_mask'].to(args.device),
        'segment_ids' : doc_token_dict['token_type_ids'].to(args.device)
    }
    
    tag_ipt = {
        'input_ids': tag_token_dict['input_ids'].to(args.device), # [batch, max_len]
        'input_mask': tag_token_dict['attention_mask'].to(args.device),
        'segment_ids' : tag_token_dict['token_type_ids'].to(args.device)
    }
 
    feature, logits_per_doc, logits_per_topic = trainer.model.get_similarity(doc_ipt, tag_ipt)

    q_feat = feature['doc']
    k_feat = feature['topic']
    batch_size = q_feat.shape[0]

    labels = torch.arange(batch_size).to(args.device)
    mask = F.one_hot(labels, num_classes=batch_size).to(args.device)
    
    logits_aa = q_feat @ q_feat.t() - mask * 1e12
    logits_bb = k_feat @ k_feat.t() - mask * 1e12

    logits_ab = q_feat @ k_feat.t()
    logits_ba = k_feat @ q_feat.t()


    logit_a = torch.cat([logits_ab, logits_aa], dim=1)
    print(logits_ab.shape, logit_a.shape, labels)

    loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels, reduction='none')
    loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels, reduction='none')
    print(loss_a, loss_b)

    out = torch.cat([q_feat, k_feat], dim=0) # [batch*2, ndim]
    sim = F.cosine_similarity(out.unsqueeze(1), out.unsqueeze(0), dim=-1) # [batch*2, batch*2]
    
    sim_ij = torch.diag(sim, batch_size)
    sim_ji = torch.diag(sim, -batch_size)
    pos = torch.cat([sim_ij, sim_ji], dim=0) # [batch*2]

    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(out.shape[0]).to(args.device) * 1e12
    nominator = torch.exp(pos) # [batch*2]
    denominator = torch.exp(sim) # [batch*2]

    loss = -torch.log(nominator / denominator.sum(dim=1))
    print(loss)


