#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   run_cdtm.py
@Time    :   2024/02/07 10:27:53
@Author  :   wangru
@Version :   1.0
'''

# here put the import lib
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from models import CDTModel
from script import BertUnDocTagTrainData
from torch.utils.data import DataLoader
from tqdm import tqdm
from trainer_cdtm import Trainer
torch.manual_seed(2024)

class Config(object):
    def __init__(self):
        # self.pretrain_model_path = '/mnt/disk2/users/wangru/model_file/pretrained_PLM/pretrained_bert_base_chinese'
        self.pretrain_model_path = '/mnt/disk2/users/wangru/model_file/pretrained_PLM/pretrained_chinese_roberta_wwm_ext'
        self.doc_max_len = 128
        self.tag_max_len = 32 # 64
        self.per_gpu_batch_size = 256
        self.num_workers = 10
        self.gradient_accumulation_steps = 50
        self.learning_rate = 3e-5 * self.gradient_accumulation_steps
        self.warmup_proportion = 0.1
        self.max_grad_norm = 1.0
        self.num_train_epochs = 8
        self.eval_step = 100
        self.train_mode = 'unsupervise'
        self.loss_fn = 'CLIPLoss' # ['CLIPLoss', 'InfoNCE']
        self.save_model_path = 'save/{}/bsz-{}-lr-{}-loss-{}/'.format(self.train_mode, self.per_gpu_batch_size, self.learning_rate, self.loss_fn)
        self.pooling_type = 'first_last_avg'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)

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

if __name__ == '__main__':
    args = Config()

    train_dataloader = get_data(args)

    dataloader_dict = {
        'train': train_dataloader
    }

    model = CDTModel(args, pooling_type=args.pooling_type)
    trainer = Trainer(args, model, dataloader_dict)
    trainer.train()
