import torch
import os
import shutil
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import dlutil as dl

from .controller import ArchBuilder, ArchSampler

class EnasListener:
    def __init__(self, run_name, data_gn, metrics):
        self.run_name = run_name
        self.dsgn, self.steps_per_epoch = data_gn
        self.metrics = metrics

    def begin(self, arch_sampler, arch_builder, builder_dir,  num_sampled_childs, device):
        self.sampler = arch_sampler
        self.builder = arch_builder
        self.model_dir = builder_dir
        self.num_sampled_childs = num_sampled_childs
        self.device = device
        self.summ_writer = SummaryWriter(
            log_dir=f'{self.model_dir}/{self.run_name}')

    def run(self, index):
        metric_dict_lst = []
        for i in range(self.num_sampled_childs):
            progress_desc = f'{self.run_name} Eval{index}[Arch{i+1}]'
            
            ds_iter = iter(self.dsgn)
            for metric in self.metrics:
                metric.reset()
            arch_seq = self.sampler()
            for _ in trange(self.steps_per_epoch, desc=progress_desc):
                bxs, bys = next(ds_iter)
                if self.device is not None:
                    bxs = [bx.cuda(self.device) for bx in bxs]
                    if type(bys) in (list, tuple):
                        bys = [by.cuda(self.device) for by in bys]
                    else:
                        bys = bys.cuda(self.device)
                with torch.no_grad():
                    by_ = self.builder(arch_seq, *bxs)
                for metric in self.metrics:
                    metric.update(bys, by_)
            metric_dict = {}
            print(f'Arch: {arch_seq}')
            for metric in self.metrics:
                result = metric.result
                print(f'{metric.name}: {result}')
                metric_dict[f'{self.run_name}_{metric.name}'] = result.detach().cpu().numpy().item()
            metric_dict_lst.append(metric_dict)
        mean_metric_dict = dict.fromkeys(metric_dict_lst[0].keys())
        for k in mean_metric_dict:
            mean_result = np.mean([m_dict[k] for m_dict in metric_dict_lst])
            mean_metric_dict[k] = mean_result
            self.summ_writer.add_scalar(k, mean_result, index)
        self.summ_writer.flush()
        return mean_metric_dict
    
    def close(self):
        self.summ_writer.close()


class EnasModel:
    def __init__(self, arch_sampler: ArchSampler, arch_builder: ArchBuilder, sampler_ckpt: dl.Checkpoint, builder_ckpt:dl.Checkpoint, device=None):
        self.sampler = arch_sampler
        self.builder = arch_builder
        self.sampler_ckpt = sampler_ckpt
        self.builder_ckpt = builder_ckpt
        if self.sampler_ckpt is not None:
            self.sampler_dir = self.sampler_ckpt.model_dir
        if self.builder_ckpt is not None:
            self.builder_dir = self.builder_ckpt.model_dir
        self.sampler_global_step = 0
        self.builder_global_step = 0
        self.device = device
    
    def reset_sampler(self):
        shutil.rmtree(self.sampler_dir)
        os.makedirs(self.sampler_dir, exist_ok=True)
        self.sampler_ckpt.reset()
        self.sampler_global_step = 0
    
    def reset_builder(self):
        shutil.rmtree(self.builder_dir)
        os.makedirs(self.builder_dir, exist_ok=True)
        self.builder_ckpt.reset()
        self.builder_global_step = 0

    
    def load_sampler_latest_checkpoint(self):
        state_dict, global_step = self.sampler_ckpt.load_state_dict_from_latest_checkpoint()
        self.sampler.load_state_dict(state_dict)
        self.sampler_global_step = global_step
    
    def load_builder_latest_checkpoint(self):
        state_dict, global_step = self.builder_ckpt.load_state_dict_from_latest_checkpoint()
        self.builder.load_state_dict(state_dict)
        self.builder_global_step = global_step


    def get_reward(self, sampled_arch, gntv, metric):
        dstv, steps_per_epoch = gntv
        ds_iter = iter(dstv)
        metric.reset()
        self.builder.eval()
        for _ in range(steps_per_epoch):
            bxs, bys = next(ds_iter)
            if self.device is not None:
                bxs = [bx.cuda(self.device) for bx in bxs]
                if type(bys) in (list, tuple):
                    bys = [by.cuda(self.device) for by in bys]
                else:
                    bys = bys.cuda(self.device)
            with torch.no_grad():
                by_ = self.builder(sampled_arch, *bxs)
            metric.update(bys, by_)
        reward = metric.result.detach().cpu().numpy().item()
        return reward

    def train(self, 
              gntr, 
              loss_func, 
              optimizer, 
              scheduler=None, 
              total_steps=5000, 
              ckpt_steps=500, 
              metrics=None, 
              summ_steps=100, 
              steps_before_sampling=999, num_sampled_childs=5, 
              listeners: EnasListener=None, 
              from_scratch=True,
              train_sampler=False, 
              sampler_optimizer=None,
              sampler_scheduler=None,
              sampler_gntv=None, 
              sampler_metric=None, steps_before_training_sampler=1000, 
              sampler_training_interval=500,
              sampler_total_steps=None, sampler_summ_steps=None,
              sampler_from_scratch=True):
        if not from_scratch:
            self.load_builder_latest_checkpoint()
        else:
            self.reset_builder()
        if not sampler_from_scratch:
            self.load_sampler_latest_checkpoint()
        else:
            self.reset_sampler()
        self.sampler.eval()
        assert num_sampled_childs >= 1, 'Must sample over 1 child network!'
        summ_writer = SummaryWriter(log_dir=f'{self.builder_dir}/train')
        dstr, steps_per_epoch = gntr
        num_ckpts = total_steps // ckpt_steps
        if listeners:
            for l in listeners:
                l.begin(self.sampler, self.builder, self.builder_dir, num_sampled_childs, self.device)
        ckpt_idx = self.builder_global_step // ckpt_steps
        for _ in range(num_ckpts):
            progress_desc = f'Checkpoint {ckpt_idx+1}'
            dstr_iter = iter(dstr)
            self.builder.train()
            for _ in trange(ckpt_steps, desc=progress_desc):
                bxs, bys = next(dstr_iter)
                if self.device is not None:
                    bxs = [bx.cuda(self.device) for bx in bxs]
                    if type(bys) in (list, tuple):
                        bys = [by.cuda(self.device) for by in bys]
                    else:
                        bys = bys.cuda(self.device)
                arch_seq = self.sampler()
                by_ = self.builder(arch_seq, *bxs)
                loss = loss_func()
                lr = optimizer.param_groups[0]['lr']
                if self.builder_global_step == 0:
                    summ_writer.add_scalar('train/loss', loss, self.builder_global_step)
                    summ_writer.add_scalar('train/lr', lr, self.builder_global_step)
                    if metrics is not None:
                        for metric in metrics:
                            metric.reset()
                            metric.update(bys, by_)
                            summ_writer.add_scalar(f'train/{metric.name}', metric.result, self.builder_global_step)
                    summ_writer.flush()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                self.builder_global_step += 1
                if self.builder_global_step % summ_steps == 0:
                    summ_writer.add_scalar('train/loss', loss, self.builder_global_step)
                    summ_writer.add_scalar('train/lr', lr, self.builder_global_step)
                    if metrics is not None:
                        for metric in metrics:
                            metric.reset()
                            metric.update(bys, by_)
                            summ_writer.add_scalar(
                                f'train/{metric.name}', metric.result, self.global_step)
                    summ_writer.flush()
            ckpt_idx = self.builder_global_step // ckpt_steps
            if train_sampler:
                sampler_summ_writer = SummaryWriter(log_dir=f'{self.sampler_dir}/train')
                if (self.builder_global_step >= steps_before_training_sampler) and (self.builder_global_step % sampler_training_interval == 0):
                    num_sampler_training = (self.builder_global_step - steps_before_training_sampler) // sampler_training_interval + 1
                    sampler_progress_desc = f'Sampler Checkpoint {num_sampler_training}'
                    self.sampler.train()
                    for _ in trange(sampler_total_steps, desc=sampler_progress_desc):
                        arch_seq = self.sampler()
                        reward = self.get_reward(arch_seq, sampler_gntv, sampler_metric)
                        sampler_loss = self.sampler.get_loss(reward)
                        sampler_lr = sampler_optimizer.param_groups[0]['lr']
                        if self.sampler_global_step == 0:
                            sampler_summ_writer.add_scalar('train/loss', sampler_loss, self.sampler_global_step)
                            sampler_summ_writer.add_scalar('train/lr', lr, self.sampler_global_step)
                            sampler_summ_writer.flush()
                        sampler_optimizer.zero_grad()
                        sampler_loss.backward()
                        sampler_optimizer.step()
                        if sampler_scheduler is not None:
                            sampler_scheduler.step()
                        self.sampler_global_step += 1
                        if self.sampler_global_step % sampler_summ_steps == 0:
                            sampler_summ_writer.add_scalar('train/loss', sampler_loss, self.sampler_global_step)
                            sampler_summ_writer.add_scalar('train/lr', lr, self.sampler_global_step)
                            sampler_summ_writer.flush()
            if self.builder_global_step >= steps_before_sampling:
                metric_dict_all = None
                if listeners:
                    self.sampler.eval()
                    self.builder.eval()
                    metric_dict_all = {}
                    for l in listeners:
                        metric_dict = l.run(ckpt_idx)
                        for m in metric_dict:
                            metric_dict_all[m] = metric_dict[m]
                self.builder_ckpt.save(self.builder, self.builder_global_step, metrics=metric_dict_all, given_index=ckpt_idx)
        summ_writer.close()
        if listeners:
            for l in listeners:
                l.close()
        if train_sampler:
            sampler_summ_writer.close()


        

        

