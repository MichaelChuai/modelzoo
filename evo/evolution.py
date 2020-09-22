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
                           metrics=metric_dict_all, given_index=f'warmup_{epoch}')
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



    def evolve(self, evo_cycles, evo_sample_size,  individual_batch_size, gntr, gntv, loss_func, optimizer, scheduler=None, num_epochs=-1, total_steps=-1, ckpt_steps=-1, metrics=None, summ_steps=100):
        self.global_step = 0
        summ_writer = SummaryWriter(log_dir=f'{self.model_dir}/evolve/train')
        val_listener = EvoListener('val', gntv, metrics)
        val_listener.begin(f'{self.model_dir}/evolve', self._model, self.device)
        summ_cycle_writer = SummaryWriter(log_dir=f'{self.model_dir}/evolve/cycle')
        dstr, steps_per_epoch = gntr
        if steps_per_epoch == -1:
            num_epochs = total_steps // ckpt_steps
            steps_per_epoch = ckpt_steps
        ind_sampler = dl.InfiniteRandomSampler(np.arange(len(self.population)))
        ind_loader = Data.DataLoader(np.arange(len(self.population)), sampler=ind_sampler, batch_size=individual_batch_size)
        for i_c in range(evo_cycles):
            epoch = self.global_step // steps_per_epoch
            ind_iter = iter(ind_loader)
            for i_e in range(num_epochs):
                progress_desc = f'[Cycle {i_c+1}]Epoch {i_e+1}/{num_epochs}/{epoch+1}'
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
                    candidates = [self.population[i]['arch'] for i in inds]
                    losses = []
                    for archseq_dict in candidates:
                        by_ = self._model(archseq_dict, *bxs)
                        loss = loss_func(by_, bys)
                        losses.append(loss[None])
                    finial_loss = torch.mean(torch.cat(losses, dim=0), dim=0)
                    lr = optimizer.param_groups[0]['lr']
                    if self.global_step == 0:
                        summ_writer.add_scalar('train/loss', loss, self.global_step)
                        summ_writer.add_scalar('train/lr', lr, self.global_step)
                        if metrics is not None:
                            for metric in metrics:
                                metric.reset()
                                metric.update(bys, by_)
                                summ_writer.add_scalar(f'train/{metric.name}', metric.result, self.global_step)
                        summ_writer.flush()
                    optimizer.zero_grad()
                    finial_loss.backward()
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    self.global_step += 1
                    if self.global_step % summ_steps == 0:
                        summ_writer.add_scalar('train/loss', loss, self.global_step)
                        summ_writer.add_scalar('train/lr', lr, self.global_step)
                        if metrics is not None:
                            for metric in metrics:
                                metric.reset()
                                metric.update(bys, by_)
                                summ_writer.add_scalar(f'train/{metric.name}', metric.result, self.global_step)
                        summ_writer.flush()
                epoch = self.global_step // steps_per_epoch
                self._model.eval()
                archseq_dict = self.population[np.random.randint(len(self.population))]['arch']
                metric_dict = val_listener.run(epoch, archseq_dict)
                self.ckpt.save(self._model, self.global_step, metrics=metric_dict, given_index=f'evo_c{i_c+1}e{epoch}')
            ind_metric_lst = []
            print('Evaluating population...')
            pop_metric = metrics[0]  ### !!!Can be changed to support multiply criteria.
            for i, ind in enumerate(self.population):
                print(f'Evaluating individual {i+1} / {len(self.population)}...')
                archseq_dict = ind['arch']
                ind_metric = self.evaluate(archseq_dict, gntv, [pop_metric])[pop_metric.name]
                ind_metric_lst.append(ind_metric)
                self.population[i]['value'] = ind_metric
            ind_avg_metric = np.mean(ind_metric_lst)
            ind_metric_dict = {f'evo_{pop_metric.name}': ind_avg_metric}
            summ_cycle_writer.add_scalar(f'cycle/{pop_metric.name}', ind_avg_metric, i_c+1)
            summ_cycle_writer.flush()
            self.ckpt.save(self._model, self.global_step, metrics=ind_metric_dict, given_index=f'evo_c{i_c+1}')
            print('Beginning to mutate...')
            evo_candidates = np.random.choice(np.arange(len(self.population)), evo_sample_size, replace=False)
            evo_samples = [self.population[i] for i in evo_candidates]
            evo_parent = max(evo_samples, key=lambda acdt: acdt['value'])
            child_archseq_dict = self.archseq_gen.mutate_archseq(evo_parent['arch'])
            child_metric = self.evaluate(child_archseq_dict, gntv, [pop_metric])[pop_metric.name]
            child = {'arch': child_archseq_dict, 'value': child_metric}
            print(evo_parent)
            print(child)
            self.population.append(child)
            self.pop_history.append(child)
            self.population.popleft()
        
        summ_writer.close()
        val_listener.close()
        summ_cycle_writer.close()

                


                
                    
