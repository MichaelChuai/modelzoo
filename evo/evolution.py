import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import dlutil as dl
from collections import deque
import shutil
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from typing import Mapping, Sequence
from .search_space import *

class EvoListener:
    def __init__(self, run_name, data_gn, metrics):
        self.run_name = run_name
        self.dsgn, self.steps_per_epoch = data_gn
        self.metrics = metrics
    
    def begin(self, model_dir, model, device):
        self.model_dir = model_dir
        self.model = model
        self.device = device
        self.summ_writer = SummaryWriter(
            log_dir=f'{self.model_dir}/{self.run_name}')

    def run(self, epoch, archseq_dict):
        progress_desc = f'{self.run_name} evaluation {epoch}'
        ds_iter = iter(self.dsgn)
        for metric in self.metrics:
            metric.reset()
        for _ in trange(self.steps_per_epoch, desc=progress_desc):
            bxs, bys = next(ds_iter)
            if self.device is not None:
                bxs = [bx.cuda(self.device) for bx in bxs]
                if type(bys) in (list, tuple):
                    bys = [by.cuda(self.device) for by in bys]
                else:
                    bys = bys.cuda(self.device)
            with torch.no_grad():
                by_ = self.model(archseq_dict, *bxs)
            for metric in self.metrics:
                metric.update(bys, by_)
        metric_dict = {}
        for metric in self.metrics:
            result = metric.result
            print(f'{metric.name}: {result}')
            metric_dict[f'{self.run_name}_{metric.name}'] = result.detach(
            ).cpu().numpy().item()
            self.summ_writer.add_scalar(f'{metric.name}', result, epoch)
        self.summ_writer.flush()
        return metric_dict
    
    def close(self):
        self.summ_writer.close()

class EvoModel:
    def __init__(self, arch: nn.Module, ckpt: dl.Checkpoint, num_ops, num_rounds, device=None):
        self._model = arch
        self.ckpt = ckpt
        self.model_dir = None
        self.archseq_gen = ArchSeqGenerator(num_ops, num_rounds)
        self.device = device
        if self.ckpt is not None:
            self.model_dir = self.ckpt.model_dir
            self.device = self.ckpt.device
        self.global_step = 0
        self.model_loaded = False

    def reset(self):
        shutil.rmtree(self.model_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt.reset()
        self.global_step = 0

    @property
    def model(self):
        return self._model

    def load_latest_checkpoint(self):
        state_dict, global_step = self.ckpt.load_state_dict_from_latest_checkpoint()
        self._model.load_state_dict(state_dict)
        self.global_step = global_step

    def warm_up(self, gntr, loss_func, optimizer, scheduler=None, num_epochs=-1, total_steps=-1, ckpt_steps=-1, metrics=None, summ_steps=100, listeners: Sequence[EvoListener]=None, from_scratch=True):
        if not from_scratch:
            self.load_latest_checkpoint()
        else:
            self.reset()
        summ_writer = SummaryWriter(log_dir=f'{self.model_dir}/warmup/train')
        dstr, steps_per_epoch = gntr
        if steps_per_epoch == -1:
            num_epochs = total_steps // ckpt_steps
            steps_per_epoch = ckpt_steps
        if listeners:
            for l in listeners:
                l.begin(f'{self.model_dir}/warmup', self._model, self.device)

        epoch = self.global_step // steps_per_epoch
        for _ in range(num_epochs):
            progress_desc = f'Epoch {epoch + 1}'
            dstr_iter = iter(dstr)
            self._model.train()
            for _ in trange(steps_per_epoch, desc=progress_desc):
                bxs, bys = next(dstr_iter)
                archseq_dict = self.archseq_gen.gen_archseq()
                if self.device is not None:
                    bxs = [bx.cuda(self.device) for bx in bxs]
                    if type(bys) in (list, tuple):
                        bys = [by.cuda(self.device) for by in bys]
                    else:
                        bys = bys.cuda(self.device)

                by_ = self._model(archseq_dict, *bxs)
                loss = loss_func(by_, bys)
                lr = optimizer.param_groups[0]['lr']
                if self.global_step == 0:
                    summ_writer.add_scalar(
                        'warmup/train/loss', loss, self.global_step)
                    summ_writer.add_scalar(
                        'warmup/train/lr', lr, self.global_step)
                    if metrics is not None:
                        for metric in metrics:
                            metric.reset()
                            metric.update(bys, by_)
                            summ_writer.add_scalar(
                                f'warmup/train/{metric.name}', metric.result, self.global_step)
                    summ_writer.flush()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                self.global_step += 1
                if self.global_step % summ_steps == 0:
                    summ_writer.add_scalar(
                        'warmup/train/loss', loss, self.global_step)
                    summ_writer.add_scalar(
                        'warmup/train/lr', lr, self.global_step)
                    if metrics is not None:
                        for metric in metrics:
                            metric.reset()
                            metric.update(bys, by_)
                            summ_writer.add_scalar(
                                f'warmup/train/{metric.name}', metric.result, self.global_step)
                    summ_writer.flush()
            epoch = self.global_step // steps_per_epoch
            metric_dict_all = None
            if listeners:
                self._model.eval()
                archseq_dict = self.archseq_gen.gen_archseq()
                metric_dict_all = {}
                for l in listeners:
                    metric_dict = l.run(epoch, archseq_dict)
                    for m in metric_dict:
                        metric_dict_all[m] = metric_dict[m]
            self.ckpt.save(self._model, self.global_step,
                           metrics=metric_dict_all, given_index=epoch)
        summ_writer.close()
        if listeners:
            for l in listeners:
                l.close()

    def evaluate(self, archseq_dict, data_gn, metrics):
        dsgn, steps_per_epoch = data_gn
        progress_desc = f'Evaluation'
        ds_iter = iter(dsgn)
        for metric in metrics:
            metric.reset()
        self._model.eval()
        for _ in trange(steps_per_epoch, desc=progress_desc):
            bxs, bys = next(ds_iter)
            if self.device is not None:
                bxs = [bx.cuda(self.device) for bx in bxs]
                if type(bys) in (list, tuple):
                    bys = [by.cuda(self.device) for by in bys]
                else:
                    bys = bys.cuda(self.device)
            with torch.no_grad():
                by_ = self._model(archseq_dict, *bxs)
            for metric in metrics:
                metric.update(bys, by_)
        metric_dict = {}
        for metric in metrics:
            result = metric.result
            print(f'{metric.name}: {result}')
            metric_dict[f'{metric.name}'] = result.detach(
            ).cpu().numpy().item()
        return metric_dict

    def setup_population(self, num_pop):
        self.population = deque()
        self.pop_history = []
        for _ in range(num_pop):
            if len(self.population) > 0:
                archset = {str(ind['arch']) for ind in self.population}
                while True:
                    archseq_dict = self.archseq_gen.gen_archseq()
                    if str(archseq_dict) not in archset:
                        break
            else:
                archseq_dict = self.archseq_gen.gen_archseq()
            acdt = {'arch': archseq_dict, 'value': -1}
            self.population.append(acdt)
            self.pop_history.append(acdt)



    def evolve(self, total_individuals, ind_batch_size, gntr, gntv, loss_func, optimizer, scheduler=None, num_epochs=-1, total_steps=-1, ckpt_steps=-1, metric=None, summ_steps=100):
        summ_writer = SummaryWriter(log_dir=f'{self.model_dir}/evolve/train')
        # for
        dstr, steps_per_epoch = gntr
        if steps_per_epoch == -1:
            num_epochs = total_steps // ckpt_steps
            steps_per_epoch = ckpt_steps
        epoch = self.global_step // steps_per_epoch
        ind_sampler = dl.InfiniteRandomSampler(np.arange(len(self.population)))
        ind_loader = Data.DataLoader(np.arange(len(self.population)), sampler=ind_sampler, batch_size=ind_batch_size)
        ind_iter = iter(ind_loader)
        for _ in range(num_epochs):
            progress_desc = f'Epoch {epoch + 1}'
            dstr_iter = iter(dstr)
            self._model.train()
            for _ in trange(steps_per_epoch, desc=progress_desc):
                bxs, bys = next(dstr_iter)
                if self.device is not None:
                    bxs = [bx.cuda(self.device) for bx in bxs]
                    if type(bys) in (list, tuple):
                        bys = [by.cuda(self.device) for by in bys]
                    else:
                        bys = bys.cuda(self.device)
                inds = next(ind_iter).numpy()
                candidates = [self.population[i] for i in inds]
                losses = []
                for archseq_dict in candidates:
                    by_ = self._model(archseq_dict, *bxs)
                    loss = loss_func(by_, bys)
                    losses.append(loss)
                finial_loss = torch.mean(torch.cat(losses, dim=0), dim=0)
                lr = optimizer.param_groups[0]['lr']
                


                
                    
