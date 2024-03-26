#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   run.py
@Time    :   2024/02/04 15:53:58
@Author  :   wangru
@Version :   1.0
'''

# here put the import lib
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from models import SimCSE
from script import BertUnSimTrainData, BertUnSimEvalData
from torch.utils.data import DataLoader
from trainer import Trainer
from tqdm import tqdm
torch.manual_seed(2024)

# class Config(object):
#     def __init__(self):
#         self.pretrain_model_path = '/mnt/disk2/users/wangru/model_file/pretrained_PLM/bert-base-uncased'
#         self.max_len = 64
#         self.per_gpu_batch_size = 64
#         self.num_workers = 10
#         self.learning_rate = 3e-5
        # self.gradient_accumulation_steps = 1
        # self.warmup_proportion = 0.1
        # self.num_train_epochs = 2
        # self.max_grad_norm = 1.0
#         self.num_class = 10
#         self.eval_step = 100
#         self.dropout = 0.1
#         self.train_mode = 'unsupervise'
#         self.save_model_path = 'save/{}/bsz-{}-lr-{}-dropout-{}/'.format(self.train_mode, self.per_gpu_batch_size, self.learning_rate, self.dropout)
#         self.seed = 42
#         self.patience = 2

#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         if not os.path.exists(self.save_model_path):
#             os.makedirs(self.save_model_path)

class Config(object):
    def __init__(self):
        self.pretrain_model_path = '/mnt/disk2/users/wangru/model_file/pretrained_PLM/bert-base-uncased'
        self.max_len = 64
        self.per_gpu_batch_size = 64
        self.num_workers = 10
        self.learning_rate = 3e-5
        self.gradient_accumulation_steps = 1
        self.warmup_proportion = 0.1
        self.max_grad_norm = 1.0
        self.num_train_epochs = 2
        self.eval_step = 100
        self.dropout = 0.1
        self.train_mode = 'unsupervise'
        self.save_model_path = 'save/{}/bsz-{}-lr-{}-dropout-{}-v4/'.format(self.train_mode, self.per_gpu_batch_size, self.learning_rate, self.dropout)
        self.seed = 42

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)

def get_data(args):
    train_path = '/mnt/disk2/users/wangru/data/STS-B/wiki1m_for_simcse.txt'
    dev_path = '/mnt/disk2/users/wangru/data/STS-B/sts-dev.csv'
    test_path = '/mnt/disk2/users/wangru/data/STS-B/sts-test.csv'

    train_text_list = []
    print('train dataset...')
    for line in tqdm(open(train_path, encoding='utf-8')):
        line = line.strip()
        train_text_list.append(line)
    
    dev_source_text_list = []
    dev_target_text_list = []
    dev_label_list = []
    print('dev dataset...')
    for line in tqdm(open(dev_path, encoding='utf-8')):
        item = line.strip().split('\t')
        score = float(item[4])
        source_text = item[5].strip()
        target_text = item[6].strip()
        dev_source_text_list.append(source_text)
        dev_target_text_list.append(target_text)
        dev_label_list.append(score)
    
    test_source_text_list = []
    test_target_text_list = []
    test_label_list = []
    print('test dataset...')
    for line in tqdm(open(test_path, encoding='utf-8')):
        item = line.strip().split('\t')
        score = float(item[4])
        source_text = item[5].strip()
        target_text = item[6].strip()
        test_source_text_list.append(source_text)
        test_target_text_list.append(target_text)
        test_label_list.append(score)
    
    train_dataset = BertUnSimTrainData(args, train_text_list)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.per_gpu_batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  collate_fn=None)
    dev_dataset = BertUnSimEvalData(args, dev_source_text_list, dev_target_text_list, dev_label_list)
    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=256,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                collate_fn=None)

    test_dataset = BertUnSimEvalData(args, test_source_text_list, test_target_text_list, test_label_list)
    test_dataloader = DataLoader(test_dataset,
                                batch_size=256,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                collate_fn=None)

    return train_dataloader, dev_dataloader, test_dataloader

if __name__ == '__main__':
    args = Config()

    train_dataloader, dev_dataloader, test_dataloader = get_data(args)

    dataloader_dict = {
        'train': train_dataloader,
        'dev': dev_dataloader
    }

    model = SimCSE(args)
    trainer = Trainer(args, model, dataloader_dict)
    trainer.train()

    # trainer.load_model('save/unsupervise/bsz-64-lr-3e-05-dropout-0.1-v4/model-dev_score-0.808974.pth') # 0.7686
    # loss, score = trainer.evaluate(test_dataloader)
    # print(loss, score)
