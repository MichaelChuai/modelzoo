import numpy as np
import torch
from meta.matching import *
import dlutil as dl

shots = 2
ways = 5

G = Embedding(1, 10)
context_embedding_network = MatchingNetwork.build_context_embedding_network(10, 64, 1)
network = MatchingNetwork(G, context_embedding_network, ways).cuda(0)

optimizer = torch.optim.Adam(network.parameters())

ckpt = dl.Checkpoint('results/matching/omniglot1', max_to_keep=10, device=0, save_best_only=True, saving_metric='test_acc')
acc = dl.MetricAccuracy(name='acc', device=0)


root = '/data/examples/omniglot'
batch_size = 32
def trans(bxs, bys):
    bx = bxs[0]
    by = bys[0]
    bx = bx.astype(np.float32) / 255.
    bx = np.expand_dims(bx, axis=1)
    by = np.squeeze(by.astype(np.int64))
    classes = sorted(list(set(by.tolist())))
    for i, c in enumerate(classes):
        by[by==c] = i
    inp_x = bx[:ways]
    sup_x = bx[ways:]
    inp_y = by[:ways]
    sup_y = by[ways:]
    bxs = [inp_x, sup_x, sup_y]
    bys = inp_y
    return (bxs, bys)
train_file = f'{root}/omniglot_bg.h5'
dstr = dl.DataReader(train_file, num_workers=5, transform_func=trans)
gntr = dstr.few_shot_reader(batch_size, shots+1, ways)

test_file = f'{root}/omniglot_eval.h5'
dste = dl.DataReader(test_file, num_workers=5, transform_func=trans)
gnte = dste.few_shot_seq_reader(batch_size * 2, shots=shots+1, selected_classes=[0,1,2,3,4])

gnte1 = dste.few_shot_seq_reader(batch_size * 2, shots=shots+1, selected_classes=[5,6,7,8,9])

listeners = [dl.Listener('test', gnte, [acc]), dl.Listener('test1', gnte1, [acc])]


def loss_func(y_, y):
    return nn.CrossEntropyLoss()(y_.transpose(-2, -1), y)

dlmodel = dl.DlModel(network, ckpt)
dlmodel.train(gntr, loss_func, optimizer, total_steps=200000, ckpt_steps=100, summ_steps=100, metrics=[acc], listeners=listeners, from_scratch=True)

