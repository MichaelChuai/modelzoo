import numpy as np
import torch
from meta.matching import *
import dlutil as dl



root = '/data/examples/omniglot/omniglot_bg.h5'
batch_size = 32
shots = 2
ways = 5
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

dstr = dl.DataReader(root, num_workers=5, transform_func=trans)

tr_reader = dstr.few_shot_reader(batch_size, shots+1, ways)

G = Embedding(1, 10)
context_embedding_network = MatchingNetwork.build_context_embedding_network(10, 64, 1)
network = MatchingNetwork(G, context_embedding_network, ways)

output = network(*x)


loss = nn.CrossEntropyLoss()(output.transpose(-2, -1), y)
