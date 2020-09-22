import torch
import torch.nn as nn
import dlutil as dl
import numpy as np
from evo.evolution import *


def input_trans(bxs, bys):
    bx, = bxs
    by, = bys
    bx = bx.astype(np.float32).transpose([2, 0, 1])
    by = by.astype(np.int64).squeeze()
    return ((bx,), by)

num_rounds = 5
num_ops = 11
stem = nn.Sequential(
    ConvOp(3, 64, kernel_size=1),
    nn.ReLU())
arch = ArchBuilder(stem, 10, 64, [2,2,2], num_rounds=num_rounds).cuda(0)
wp_optimizer = torch.optim.Adam(arch.parameters())
optimizer = torch.optim.Adam(arch.parameters())
loss_func = nn.CrossEntropyLoss()
ckpt = dl.Checkpoint('temp/evo/e2', device=0)
acc = dl.MetricAccuracy(device=0, name='acc')

batch_size = 32
ds = dl.DataReader('/data/testdata/cifar10/cifar10.h5', transform_func=input_trans)

gntr = ds.common_cls_reader(batch_size, selected_classes=['tr'], shuffle=True)
gnte = ds.common_cls_reader(batch_size * 3, selected_classes=['te'], shuffle=False)
warmup_listeners = [EvoListener('wp/test', gnte, [acc])]


warmup_num_epochs = 10
num_pop = 200
evo_cycles = 200
evo_sample_size = 50
individual_batch_size = 5
evo_num_epochs = 2

emodel = EvoModel(arch, ckpt, num_ops, num_rounds, device=0)
emodel.warm_up(gntr, loss_func, wp_optimizer, num_epochs=warmup_num_epochs, metrics=[acc], listeners=warmup_listeners, from_scratch=True)
emodel.setup_population(num_pop)
emodel.evolve(evo_cycles=evo_cycles, evo_sample_size=evo_sample_size, individual_batch_size=individual_batch_size, gntr=gntr, gntv=gnte, loss_func=loss_func, optimizer=optimizer, num_epochs=evo_num_epochs, metrics=[acc])

import joblib

joblib.dump(emodel.pop_history, 'temp/evo/e2.pkl', compress=3)
