#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   trainer_cdtm.py
@Time    :   2024/02/07 13:46:31
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
        self.writer = SummaryWriter(self.args.save_model_path)

        self.optimizer = optim.AdamW(self.model.parameters(), 
                                    lr=self.args.learning_rate)

        num_training_steps = len(self.train_dataloader) * self.args.num_train_epochs
        num_warmup_steps = int(num_training_steps / self.args.gradient_accumulation_steps * self.args.warmup_proportion)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        start = time.time()
        self.model.zero_grad()
        for epoch in range(1, self.args.num_train_epochs+1):
            for step, data in enumerate(tqdm(self.train_dataloader)):
                self.model.train()
                source, target = data
                doc_ipt = {
                    'input_ids': source['input_ids'].squeeze(1).to(self.args.device), # [batch, max_len]
                    'input_mask': source['attention_mask'].squeeze(1).to(self.args.device),
                    'segment_ids' : source['token_type_ids'].squeeze(1).to(self.args.device)
                }
                
                tag_ipt = {
                    'input_ids': target['input_ids'].squeeze(1).to(self.args.device), # [batch, max_len]
                    'input_mask': target['attention_mask'].squeeze(1).to(self.args.device),
                    'segment_ids' : target['token_type_ids'].squeeze(1).to(self.args.device)
                }
                
                out = self.model(doc_ipt, tag_ipt)
                acc = None
                if len(out) == 2:
                    sim_score, loss = out
                elif len(out) == 3:
                    sim_score, loss, acc = out
                
                zero = torch.zeros_like(sim_score)
                one = torch.ones_like(sim_score)
                sim_pred = torch.where(sim_score >= 0.5, one, sim_score)
                sim_pred = torch.where(sim_score < 0.5, zero, sim_pred)
                sim_ratio = sim_pred.mean()

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                
                loss.backward()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

                self.model.logit_scale.data = torch.clamp(self.model.logit_scale.data, 0, 4.6052) # ln(100)=4.6052

                lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                iteration = (epoch - 1) * len(self.train_dataloader) + (step + 1)
                
                if iteration % self.args.eval_step == 0:
                    cur_time = time.time()
                    cost_time = cur_time - start
                    speed = iteration / cost_time
                    cost_iter_time = cost_time / (iteration // self.args.eval_step)

                    logger.info(
                        f'epoch:{epoch}/{self.args.num_train_epochs} | ' +
                        f'step:{iteration}/{num_training_steps} | ' +
                        f'lr:{lr:.6f} | ' +
                        f'speed:{speed:.2f}it/s | ' +
                        f'cost_iter_time:{cost_iter_time:.2f}s | ' +
                        f'cost_time:{cost_time:.2f}s | ' +
                        f'【train】loss:{loss.item():.6f} | ' +
                        f'sim_ratio:{sim_ratio.item():.6f} | ' +
                        f'logit_scale: {self.model.logit_scale.data:.3f}'
                    )

                    if acc is not None:
                        logger.info(
                            f'Doc2Topic Acc: {acc["q2k"].item() * 100:.2f}% | ' + 
                            f'Topic2Doc Acc: {acc["k2q"].item() * 100:.2f}%'
                        )

                    self.writer.add_scalar('loss', loss.item(), iteration)
                    self.writer.add_scalar('sim_ratio', sim_ratio.item(), iteration)

            if self.dev_dataloader is None:
                self.save_model(model=self.model, root_path=self.args.save_model_path, suffix='model-{}.pth'.format(epoch))
        logger.info('end training')
    
    def predict(self, dataloader):
        self.model.eval()
        dev_outputs = []
        with torch.no_grad():
            pbar = tqdm(iterable=dataloader, ascii=True)
            for step, data in enumerate(dataloader):
                source, target = data
                doc_ipt = {
                    'input_ids': source['input_ids'].squeeze(1).to(self.args.device), # [batch, max_len]
                    'input_mask': source['attention_mask'].squeeze(1).to(self.args.device),
                    'segment_ids' : source['token_type_ids'].squeeze(1).to(self.args.device)
                }
                
                tag_ipt = {
                    'input_ids': target['input_ids'].squeeze(1).to(self.args.device), # [batch, max_len]
                    'input_mask': target['attention_mask'].squeeze(1).to(self.args.device),
                    'segment_ids' : target['token_type_ids'].squeeze(1).to(self.args.device)
                }

                feature, logits_per_doc, logits_per_topic = self.model.get_similarity(doc_ipt, tag_ipt)

                cos_score = F.cosine_similarity(feature['doc'], feature['topic'])

                dev_outputs.append(cos_score)

                pbar.set_postfix()
                pbar.update()

        dev_outputs = torch.cat(dev_outputs)
        return dev_outputs
