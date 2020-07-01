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



class EnasModel:
    def __init__(self, arch_sampler: ArchSampler, arch_builder: ArchBuilder, sampler_ckpt: dl.Checkpoint, builder_ckpt:dl.Checkpoint, device=None):
        self.arch_sampler = arch_sampler
        self.arch_builder = arch_builder
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

    @property
    def sampler(self):
        return self.sampler
    
    @property
    def builder(self):
        return self.builder

    
    def load_sampler_latest_checkpoint(self):
        state_dict, global_step = self.sampler_ckpt.load_state_dict_from_latest_checkpoint()
        self.sampler.load_state_dict(state_dict)
        self.sampler_global_step = global_step
    
    def load_builder_latest_checkpoint(self):
        state_dict, global_step = self.builder_ckpt.load_state_dict_from_latest_checkpoint()
        self.builder.load_state_dict(state_dict)
        self.builder_global_step = global_step


    def get_reward(self, sampled_arch, gntv, metric, idx=0):
        dstv, steps_per_epoch = gntv
        progress_desc = f'Reward #{idx}'
        ds_iter = iter(dstv)
        metric.reset()
        self.arch_builder.eval()
        for _ in trange(steps_per_epoch, desc=progress_desc):
            bxs, bys = next(ds_iter)
            if self.device is not None:
                bxs = [bx.cuda(self.device) for bx in bxs]
                if type(bys) in (list, tuple):
                    bys = [by.cuda(self.device) for by in bys]
                else:
                    bys = bys.cuda(self.device)
            with torch.no_grad():
                by_ = self.arch_builder(sampled_arch, *bxs)
            metric.update(bys, by_)
        reward = metric.result.detach().cpu().numpy().item()
        print(f'{metric.name}: {reward}')
        return reward

    # def train(self, gntr, )