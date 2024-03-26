#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   cdtm.py
@Time    :   2024/02/05 10:42:51
@Author  :   wangru
@Version :   1.0
'''

# here put the import lib
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from transformers import BertModel
from collections import OrderedDict

class CLIPLoss(nn.Module):
    def __init__(self, device):
        super(CLIPLoss, self).__init__()
        self.device = device
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, q_feat, k_feat):
        """
        q_feat: [batch, ndim], vector1
        k_feat: [batch, ndim], vector2
        """
        # normalized features
        q_feat = F.normalize(q_feat, p=2, dim=1)
        k_feat = F.normalize(k_feat, p=2, dim=1)
        
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logit_scale = logit_scale.mean()
        logits_per_q = logit_scale * q_feat @ k_feat.t() # [batch, batch]
        logits_per_k = logits_per_q.t()

        ground_truth = torch.arange(len(logits_per_q)).to(self.device)

        # compute loss
        q_loss = F.cross_entropy(logits_per_q, ground_truth)
        k_loss = F.cross_entropy(logits_per_k, ground_truth)
        loss = (q_loss + k_loss) / 2

        # compute acc
        acc = {}
        q2k_acc = (logits_per_q.argmax(-1) == ground_truth).sum() / len(logits_per_q)
        k2q_acc = (logits_per_k.argmax(-1) == ground_truth).sum() / len(logits_per_k)
        acc = {
            'q2k': q2k_acc,
            'k2q': k2q_acc
        }
        return loss, acc

class InfoNCE(nn.Module):
    """
    一个正样本对，通过encoder编码得到的向量q_feat， k_feat
    mean(-log(正样本的得分/所有样本的得分))
    https://blog.csdn.net/weixin_44966641/article/details/120382198
    """
    def __init__(self, device, temperature=0.1, reduction='mean'):
        super(InfoNCE, self).__init__()
        self.device = device
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, q_feat, k_feat):
        """
        q_feat: [batch, ndim], vector1
        k_feat: [batch, ndim], vector2
        """
        # normalized features
        q_feat = F.normalize(q_feat, p=2, dim=1)
        k_feat = F.normalize(k_feat, p=2, dim=1)
        batch_size = q_feat.shape[0]

        out = torch.cat([q_feat, k_feat], dim=0) # [batch*2, ndim]
        sim = F.cosine_similarity(out.unsqueeze(1), out.unsqueeze(0), dim=-1) # [batch*2, batch*2]
        
        sim_ij = torch.diag(sim, batch_size)
        sim_ji = torch.diag(sim, -batch_size)
        pos = torch.cat([sim_ij, sim_ji], dim=0) # [batch*2]

        # 将相似度矩阵对角线置为很小的值, 消除自身的影响
        sim = sim - torch.eye(out.shape[0]).to(self.device) * 1e12
        nominator = torch.exp(pos / self.temperature) # [batch*2]
        denominator = torch.exp(sim / self.temperature) # [batch*2]

        loss = -torch.log(nominator / denominator.sum(dim=1)) # [batch*2]

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'none':
            return loss

class NTXentLoss(nn.Module): # 计算结果与InfoNCE一样
    def __init__(self, device, temperature=0.1, reduction='mean'):
        super(InfoNCE, self).__init__()
        self.device = device
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, q_feat, k_feat):
        """
        q_feat: [batch, ndim], vector1
        k_feat: [batch, ndim], vector2
        """
        # normalized features
        q_feat = F.normalize(q_feat, p=2, dim=1)
        k_feat = F.normalize(k_feat, p=2, dim=1)
        batch_size = q_feat.shape[0]

        labels = torch.arange(batch_size).to(self.device)
        mask = F.one_hot(labels, num_classes=batch_size).to(self.device)

        # 将相似度矩阵对角线置为很小的值, 消除自身的影响
        logits_aa = (q_feat @ q_feat.t() - mask * 1e12) / self.temperature
        logits_bb = (k_feat @ k_feat.t() - mask * 1e12) / self.temperature
        logits_ab = q_feat @ k_feat.t() / self.temperature
        logits_ba = logits_ab.t() / self.temperature
        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels) # concat后 [batch, ndim*2]
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
        loss = (loss_a + loss_b) / 2
        return loss

class CDTModel(nn.Module):
    """
    Contrastive Document-Topic Model：针对微博的<微博正文，话题>的对比学习模型
    """
    def __init__(self, args, pooling_type='cls'):
        super(CDTModel, self).__init__()
        self.args = args
        self.pooling_type = pooling_type

        self.text_encoder = BertModel.from_pretrained(args.pretrain_model_path)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
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
    
    def forward(self, doc_ipt, topic_ipt):
        doc_out = self.text_encoder(input_ids=doc_ipt['input_ids'], attention_mask=doc_ipt['input_mask'], token_type_ids=doc_ipt['segment_ids'], output_hidden_states=True)
        topic_out = self.text_encoder(input_ids=topic_ipt['input_ids'], attention_mask=topic_ipt['input_mask'], token_type_ids=topic_ipt['segment_ids'], output_hidden_states=True)

        doc_embed = self.get_pooling(doc_out, doc_ipt['input_mask']) # [batch, ndim]
        topic_embed = self.get_pooling(topic_out, topic_ipt['input_mask']) # [batch, ndim]

        # normalized features
        doc_features = F.normalize(doc_embed, p=2, dim=1)
        topic_features = F.normalize(topic_embed, p=2, dim=1)

        sim_score = F.cosine_similarity(doc_features, topic_features, dim=-1)
        
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_doc = logit_scale * doc_features @ topic_features.t() # [batch, batch]
        logits_per_topic = logits_per_doc.t()

        ground_truth = torch.arange(len(logits_per_doc)).to(self.args.device)

        # compute loss
        doc_loss = F.cross_entropy(logits_per_doc, ground_truth)
        topic_loss = F.cross_entropy(logits_per_topic, ground_truth)
        loss = (doc_loss + topic_loss) / 2

        # compute acc
        acc = {}
        q2k_acc = (logits_per_doc.argmax(-1) == ground_truth).sum() / len(logits_per_doc)
        k2q_acc = (logits_per_topic.argmax(-1) == ground_truth).sum() / len(logits_per_topic)
        acc = {
            'q2k': q2k_acc,
            'k2q': k2q_acc
        }
        return sim_score, loss, acc

    def get_similarity(self, doc_ipt, topic_ipt):
        doc_out = self.text_encoder(input_ids=doc_ipt['input_ids'], attention_mask=doc_ipt['input_mask'], token_type_ids=doc_ipt['segment_ids'], output_hidden_states=True)
        topic_out = self.text_encoder(input_ids=topic_ipt['input_ids'], attention_mask=topic_ipt['input_mask'], token_type_ids=topic_ipt['segment_ids'], output_hidden_states=True)
        
        doc_embed = self.get_pooling(doc_out, doc_ipt['input_mask']) # [batch, ndim]
        topic_embed = self.get_pooling(topic_out, topic_ipt['input_mask']) # [batch, ndim]

        # normalized features
        doc_features = F.normalize(doc_embed, p=2, dim=1)
        topic_features = F.normalize(topic_embed, p=2, dim=1)
        feature = {
            'doc': doc_features,
            'topic': topic_features
        }

        logits_per_doc = doc_features @ topic_features.t() # [batch, batch]
        logits_per_topic = logits_per_doc.t()
        return feature, logits_per_doc, logits_per_topic
