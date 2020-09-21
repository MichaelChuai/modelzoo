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

ckpt = dl.Checkpoint('temp/evo/e1', device=0)
acc = dl.MetricAccuracy(device=0, name='acc')
batch_size = 32
ds = dl.DataReader('/data/testdata/cifar10/cifar10.h5', transform_func=input_trans)

gnte = ds.common_cls_reader(batch_size * 3, selected_classes=['te'], shuffle=False)
ArchSeqGenerator
emodel = EvoModel(arch, ckpt, num_ops, num_rounds, device=0)
emodel.load_latest_checkpoint()

ac_dict = emodel.archseq_gen.gen_archseq()
print(ac_dict)
# emodel.evaluate(ac_dict, gnte, [acc])

emodel.setup_population(200)
ind_batch_size = 5
ind_sampler = dl.InfiniteRandomSampler(np.arange(len(emodel.population)))
ind_loader = Data.DataLoader(np.arange(len(emodel.population)), sampler=ind_sampler, batch_size=ind_batch_size)
ind_iter = iter(ind_loader)
inds = next(ind_iter).numpy()
candidates = [emodel.population[i]['arch'] for i in inds]


