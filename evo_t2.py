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
optimizer = torch.optim.Adam(arch.parameters())
loss_func = nn.CrossEntropyLoss()
ckpt = dl.Checkpoint('temp/evo/e1', device=0)
acc = dl.MetricAccuracy(device=0, name='acc')

batch_size = 32
ds = dl.DataReader('/data/testdata/cifar10/cifar10_test.h5', transform_func=input_trans)

gntr = ds.common_cls_reader(batch_size, selected_classes=['tr'], shuffle=True)
gnte = ds.common_cls_reader(batch_size * 3, selected_classes=['te'], shuffle=False)

emodel = EvoModel(arch, ckpt, num_ops, num_rounds, device=0)
emodel.load_latest_checkpoint()
emodel.setup_population(5)

a = emodel.evolve(2, 3, 3, gnte, gnte, loss_func, optimizer, num_epochs=3, metrics=[acc])



