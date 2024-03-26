#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_helper.py
@Time    :   2024/02/04 09:34:10
@Author  :   wangru
@Version :   1.0
'''

# here put the import lib
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

'''
输入样本对，有2种tokenize方式得到，设一个batch输入文本对为[[a0,a1],[b0,b1]]
a0,a1为相似文本；a1,b1为相似文本
batch_source=[a0,b0], source=a0,b0
batch_target=[a1,b1], target=a1,b1
1：分别对每个样本进行tokenize，即：
token_dict1 = tokenizer(batch_source)
token_dict2 = tokenizer(batch_target)
则可以得到source的编码与target的编码
source_input_id: [batch, max_len], target_input_id: [batch, max_len]
source_out = bert(source_input_id) # [a0_feat, b0_feat] [batch, ndim]
target_out = bert(target_input_id) # [a1_feat, b1_feat] [batch, ndim]
out = concat([source_out, target_out], dim=0) # [batch*2, max_len], [a0_feat, b0_feat, a1_feat, b1_feat] [batch*2, ndim]
2：对样本对进行tokenize，即：
token_dict = tokenizer([source, target])
input_id: [batch, 2, max_len]
source_input_id = input_id[:,0,:], target_input_id = input_id[:,1,:] 其input_id结果与第一种方式相同
input_id = input_id.view(-1, max_len) # [batch*2, max_len]
out = bert(input_id) # [a0_feat, a1_feat, b0_feat, b1_feat], [2*batch, ndim] 
source_out = out.view(-1, 2, out.shape[-1])[:, 0, :] # [a0_feat, b0_feat] [batch, ndim]
target_out = out.view(-1, 2, out.shape[-1])[:, 1, :] # [a1_feat, b1_feat] [batch, ndim]
concat([source_out, target_out], dim=0) # [a0_feat, b0_feat, a1_feat, b1_feat]
'''

class BertTokenData(Dataset):
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)

class BertUnSimTrainData(BertTokenData):
    """
    获取无监督训练语料
    """
    def __init__(self, args, text_list):
        super(BertUnSimTrainData, self).__init__(args)
        self.args = args

        self.text_list = text_list

    def __getitem__(self, index):
        text = self.text_list[index]

        token_dict = self.tokenizer(
            [text, text], 
            max_length=self.args.max_len, 
            truncation=True, 
            padding='max_length'
        )

        input_ids = token_dict['input_ids']
        input_mask = token_dict['attention_mask']
        segment_ids = token_dict['token_type_ids']

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        return input_ids, input_mask, segment_ids

    def __len__(self):
        return len(self.text_list)

class BertUnSimEvalData(BertTokenData):
    def __init__(self, args, source_text_list, target_text_list, label_list):
        super(BertUnSimEvalData, self).__init__(args)
        self.args = args

        self.source_text_list = source_text_list
        self.target_text_list = target_text_list
        self.label_list = label_list
    
    def __getitem__(self, index):
        source_text = self.source_text_list[index]
        target_text = self.target_text_list[index]
        label = self.label_list[index]
        label = torch.tensor(label, dtype=torch.float)

        token_dict = self.tokenizer(
            [source_text, target_text], 
            max_length=self.args.max_len, 
            truncation=True, 
            padding='max_length'
        )

        input_ids = token_dict['input_ids']
        input_mask = token_dict['attention_mask']
        segment_ids = token_dict['token_type_ids']

        input_ids = torch.tensor(input_ids, dtype=torch.long) # [batch, 2, max_len]
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        return input_ids, input_mask, segment_ids, label
    
    def __len__(self):
        return len(self.source_text_list)


class BertUnDocTagTrainData(BertTokenData):
    """
    获取无监督训练语料<doc, tag>
    """
    def __init__(self, args, doc_list, tag_list):
        super(BertUnDocTagTrainData, self).__init__(args)
        self.args = args

        self.doc_list = doc_list
        self.tag_list = tag_list

    def __getitem__(self, index):
        doc = self.doc_list[index]
        tag = self.tag_list[index]

        doc_token_dict = self.tokenizer(
            doc, 
            max_length=self.args.doc_max_len, 
            truncation=True, 
            padding='max_length',
            return_tensors='pt'
        )

        tag_token_dict = self.tokenizer(
            tag, 
            max_length=self.args.tag_max_len, 
            truncation=True, 
            padding='max_length',
            return_tensors='pt'
        )
        
        return doc_token_dict, tag_token_dict

    def __len__(self):
        return len(self.doc_list)
