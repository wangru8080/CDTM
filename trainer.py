#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   trainer.py
@Time    :   2024/02/06 09:35:57
@Author  :   wangru
@Version :   1.0
'''

# here put the import lib
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.nn as nn
import numpy as np
import random
from torch.nn import functional as F
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
from scipy.stats import spearmanr
from loguru import logger
import time
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(2024)
torch.cuda.manual_seed_all(2024)
torch.backends.cudnn.deterministic = True

class Trainer(object):
    def __init__(self, args, model, dataloader_dict=None):
        self.args = args

        if dataloader_dict:
            self.train_dataloader = dataloader_dict['train'] if 'train' in dataloader_dict else None
            self.dev_dataloader = dataloader_dict['dev'] if 'dev' in dataloader_dict else None

        self.model = model.to(self.args.device)
    
    def save_model(self, model, root_path, suffix):
        """
        suffix: 保存的模型名
        """
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        model_to_save = model.module if hasattr(model, 'module') else model
        path = root_path + suffix
        torch.save(model_to_save.state_dict(), path, _use_new_zipfile_serialization=False)
        logger.info('save model: {} ...'.format(suffix))

    def load_model(self, path):
        logger.info('loading model from {} ...'.format(path))
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint, strict=True)
        self.model = self.model.to(self.args.device)
    
    def log_info(self):
        logger.remove(handler_id=None)
        logger.add('{}/train-{}.log'.format(self.args.save_model_path, time.strftime("%Y%m%d%H%M%S", time.localtime())))
        logger.info('parameters: {}'.format(self.args.__dict__))
        logger.info('start training')
        logger.info('loading train data, len of train data: {}'.format(len(self.train_dataloader.dataset)))

        if self.dev_dataloader is not None:
            logger.info('loading dev data, len of dev data: {}'.format(len(self.dev_dataloader.dataset)))

    def train(self):
        self.log_info()

        self.optimizer = optim.AdamW(self.model.parameters(), 
                                    lr=self.args.learning_rate)

        num_training_steps = len(self.train_dataloader) * self.args.num_train_epochs

        start = time.time()
        best_score = float('-inf')
        self.model.zero_grad()
        for epoch in range(1, self.args.num_train_epochs+1):
            for step, data in enumerate(tqdm(self.train_dataloader)):
                self.model.train()
                input_ids, input_mask, segment_ids = data
                input_ids = input_ids.to(self.args.device) # [batch, 2, max_len]
                segment_ids = segment_ids.to(self.args.device)
                input_mask = input_mask.to(self.args.device)

                input_ids = input_ids.view(-1, self.args.max_len) # [batch*2, max_len]
                segment_ids = segment_ids.view(-1, self.args.max_len)
                input_mask = input_mask.view(-1, self.args.max_len)

                out, loss = self.model(input_ids, input_mask, segment_ids)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                iteration = (epoch - 1) * len(self.train_dataloader) + (step + 1)
                
                if iteration % self.args.eval_step == 0:
                    cur_time = time.time()
                    cost_time = cur_time - start
                    speed = iteration / cost_time
                    cost_iter_time = cost_time / (iteration // self.args.eval_step)

                    logger.info('epoch:{}/{} step:{}/{} lr:{:.6f} speed:{:.2f}it/s cost_iter_time:{:.2f}s cost_time:{:.2f}s【train】loss:{:.6f}'.format(
                        epoch,
                        self.args.num_train_epochs,
                        iteration,
                        num_training_steps,
                        lr,
                        speed,
                        cost_iter_time,
                        cost_time,
                        loss.item()
                    ))

                    if self.dev_dataloader is not None:
                        dev_loss, dev_score = self.evaluate(self.dev_dataloader)

                        logger.info('【dev】loss:{:.6f} metric:{:.6f}'.format(dev_loss, dev_score))

                        if dev_score > best_score:
                            best_score = dev_score
                            best_epoch = epoch
                            best_step = iteration
                            counter = 0
                            logger.info('best score: {} in step {} epoch {}, save model'.format(best_score, best_step, best_epoch))
                            self.save_model(model=self.model, root_path=self.args.save_model_path, suffix='model-dev_score-{:.6f}.pth'.format(dev_score))
                        else:
                            counter += 1
        logger.info('end training')
    
    def predict(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        dev_outputs = []
        dev_targets = []
        with torch.no_grad():
            pbar = tqdm(iterable=dataloader, ascii=True)
            for step, data in enumerate(dataloader):
                input_ids, input_mask, segment_ids, label = data
                input_ids = input_ids.to(self.args.device) # [batch, 2, max_len]
                input_mask = input_mask.to(self.args.device)
                segment_ids = segment_ids.to(self.args.device)
                label = label.to(self.args.device)

                input_ids = input_ids.view(-1, input_ids.shape[-1]) # [batch*2, max_len]
                input_mask = input_mask.view(-1, input_ids.shape[-1])
                segment_ids = segment_ids.view(-1, input_ids.shape[-1])

                pooled_output, loss = self.model(input_ids, input_mask, segment_ids) # [batch*2, hidden_size]
                pooled_output = pooled_output.view(-1, 2, pooled_output.shape[-1]) # [batch, 2, hidden_size]

                # 计算相似度：pooled_output[:,0,:]第一条样本结果，pooled_output[:,1,:]第二条样本结果
                pred_sim = F.cosine_similarity(pooled_output[:,0,:], pooled_output[:,1,:], dim=-1)

                total_loss += loss.item()

                dev_outputs.append(pred_sim)
                dev_targets.append(label)

                pbar.set_postfix(loss='%.6f' % loss.item())
                pbar.update()

        dev_outputs = torch.cat(dev_outputs)
        dev_targets = torch.cat(dev_targets)
        avg_loss = float('%.6f' % (total_loss / len(dataloader)))
        return avg_loss, dev_outputs, dev_targets
    
    def get_metrics(self, input: torch.Tensor, target: torch.Tensor):
        score = spearmanr(input.cpu().numpy(), target.cpu().numpy()).correlation
        return score

    def evaluate(self, dataloader):
        loss, outputs, targets = self.predict(dataloader)
        score = self.get_metrics(outputs, targets)
        return loss, score
