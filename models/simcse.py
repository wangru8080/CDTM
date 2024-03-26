#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   simcse.py
@Time    :   2024/02/04 15:20:14
@Author  :   wangru
@Version :   1.0
'''

# here put the import lib
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertConfig, BertModel
from collections import OrderedDict

class UnSupLoss(nn.Module):
    '''
    无监督loss
    y_pred (tensor): bert的输出，[2*batch, hidden_size]：是根据bert(tokenize([ipt1, ipt2]))生成
    一条文本输入模型2次，由于dropout的随机性，构造正样本对。即[a,b]->[a1,a2,b1,b2]
    a1,a2为正样本对，b1,b2为正样本对
    '''
    def __init__(self, device, temp=0.05):
        super(UnSupLoss, self).__init__()
        self.device = device
        self.temp = temp

    def forward(self, out_feats):
        # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
        idx = torch.arange(out_feats.shape[0]).to(self.device)
        y_true = (idx - idx % 2 * 2) + 1
        # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
        sim = F.cosine_similarity(out_feats.unsqueeze(1), out_feats.unsqueeze(0), dim=-1)
        # 将相似度矩阵对角线置为很小的值, 消除自身的影响
        sim = sim - torch.eye(out_feats.shape[0]).to(self.device) * 1e12
        # 相似度矩阵除以温度系数
        sim = sim / self.temp
        # 计算相似度矩阵与y_true的交叉熵损失
        # 计算交叉熵，每个case都会计算与其他case的相似度得分，得到一个得分向量，目的是使得该得分向量中正样本的得分最高，负样本的得分最低
        loss = F.cross_entropy(sim, y_true)
        return torch.mean(loss)

class SimCSE(nn.Module):
    '''
    train_mode: ['supervise', 'unsupervise']
    '''
    def __init__(self, args, dropout=0.1, mode='unsupervise', pooling_type='cls'):
        super(SimCSE, self).__init__()
        self.args = args
        self.mode = mode
        self.pooling_type = pooling_type

        config = BertConfig.from_pretrained(args.pretrain_model_path)
        config.attention_probs_dropout_prob = dropout  # 修改config的dropout系数
        config.hidden_dropout_prob = dropout
        self.bert = BertModel.from_pretrained(args.pretrain_model_path, config=config)
        self.hidden_size = self.bert.config.hidden_size

    def mean_pooling(self, x, mask=None):
        if mask is not None:
            input_mask_expanded = mask.unsqueeze(-1).expand(x.size()).float() # [batch, max_len, hidden_size]
            sum_embeddings = torch.sum(x * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            return mean_embeddings
        else:
            return torch.mean(x, dim=1)

    def get_attention_pooling(self, last_encoder_layers, input_mask, attention_dim=768):
        attention = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(attention_dim, attention_dim)),
            ('layernorm', nn.LayerNorm(attention_dim)),
            ('gelu', nn.GELU()),
            ('fc2', nn.Linear(attention_dim, 1))
        ]))
        attention = attention.to(self.args.device)
        weight = attention(last_encoder_layers).float() # [batch, max_len, 1]
        mask = 1 - input_mask
        mask = mask.float().masked_fill(mask.eq(1), float('-inf')) # [batch, max_len]
        mask = mask.unsqueeze(-1) # [batch, max_len, 1]
        weight = weight + mask
        weight = torch.softmax(weight, dim=1) # [batch, max_len, 1]
        attention_pooling = torch.sum(weight * last_encoder_layers, dim=1) # [batch, hidden_size]
        return attention_pooling

    def get_pooling(self, bert_out, mask=None):
        last_encoder_layers = bert_out['last_hidden_state']
        pooled_output = bert_out['pooler_output']
        hidden_states = bert_out['hidden_states']
        embedding_output = hidden_states[0]
        all_encoder_layers = hidden_states[1:]

        out = pooled_output
        if self.pooling_type == 'pooler':
            out = pooled_output
        elif self.pooling_type == 'cls':
            out = last_encoder_layers[:, 0, :]
        elif self.pooling_type == 'first_last_avg':
            out = self.mean_pooling(all_encoder_layers[0] + all_encoder_layers[-1], mask)
        elif self.pooling_type == 'last_avg':
            out = self.mean_pooling(last_encoder_layers, mask)
        elif self.pooling_type == 'attention_pooling':
            out = self.get_attention_pooling(last_encoder_layers, mask, attention_dim=self.hidden_size)
        return out

    def forward(self, input_ids, input_mask, segment_ids):
        bert_out = self.bert(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, output_hidden_states=True)

        # Train: [batch*2, hidden_size] Test: [batch, hidden_size]
        pooled_output = self.get_pooling(bert_out, input_mask)
        if self.mode == 'unsupervise':
            loss_function = UnSupLoss(self.args.device)
            loss = loss_function(pooled_output)
            return pooled_output, loss
        return pooled_output
