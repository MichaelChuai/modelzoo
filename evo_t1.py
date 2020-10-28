import torch
import torch.nn as nn
import dlutil as dl
import numpy as np
from evo.evolution import *
import cv2


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
ckpt = dl.Checkpoint('temp/evo/e1', max_to_keep=10, device=0)
acc = dl.MetricAccuracy(device=0, name='acc')


batch_size = 32
ds = dl.DataReader('/data/testdata/cifar10/cifar10_test.h5', transform_func=input_trans)
gntr = ds.common_cls_reader(batch_size, selected_classes=['tr'])
gnte = ds.common_cls_reader(batch_size * 3, selected_classes=['te'], shuffle=False)
listeners = [EvoListener('test', gnte, [acc])]
emodel = EvoModel(arch, ckpt, num_ops, num_rounds, device=0)
warmup_num_epochs = 10
emodel.warm_up(gntr, loss_func, optimizer, num_epochs=warmup_num_epochs, metrics=[acc], listeners=listeners, from_scratch=True)


