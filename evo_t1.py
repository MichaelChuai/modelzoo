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

ckpt = dl.Checkpoint('temp/evo/e1', max_to_keep=10, device=0)
acc = dl.MetricAccuracy(device=0, name='acc')

batch_size = 32
dstr = dl.DataReader('/data/testdata/cifar10/cifar10.h5', transform_func=input_trans)
gntr = dstr.common_cls_reader(batch_size, selected_classes=['tr'])

# it = iter(gntr[0])
# x, y = next(it)
# x[0] = x[0].cuda(0)
# archseq = EvoModel.gen_archseq(num_ops, num_rounds)
# t = arch(archseq, *x)

emodel = EvoModel(arch, ckpt, num_ops, num_rounds, device=0)

