from enas import *
import torch
import torch.nn as nn
import dlutil as dl

class CatLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(CatLayer, self).__init__()
        assert kernel_size % 2 == 1
        padding_size = (kernel_size - 1) // 2
        self.l1 = nn.Conv2d(in_channel, out_channel//2, [1, kernel_size], padding=[0, padding_size])
        self.l2 = nn.Conv2d(in_channel, out_channel//2, [kernel_size, 1], padding=[padding_size, 0])
    
    def forward(self, x1, x2):
        r1 = self.l1(x1)
        r2 = self.l2(x2)
        r = torch.cat([r1, r2], dim=1)
        return r
    # def forward(self, x1):
    #     r1 = self.l1(x1)
    #     return r1

layers = {
    # 0:[
    #     CatLayer(8, 16, 3),
    #     CatLayer(8, 16, 5),
    #     CatLayer(8, 16, 7)
    # ],
    0:[
        nn.Conv2d(8, 16, [3, 3], padding=1),
        nn.Conv2d(8, 16, [5, 5], padding=2)
    ],
    1:[
        nn.Conv2d(16, 8, [3, 3], padding=1),
        nn.Conv2d(16, 8, [5, 5], padding=2),
        nn.Conv2d(16, 8, [7, 7], padding=3)
    ],
    2:[
        nn.MaxPool2d([3, 3], [2, 2], padding=1),
        nn.AvgPool2d([3, 3], [2, 2], padding=1)
    ],
    3:[
        nn.Conv2d(8, 1, [3, 3], padding=1),
        nn.Conv2d(8, 1, [5, 5], padding=2)
    ],
    4:[
        nn.MaxPool2d([3, 3], [2, 2], padding=1),
        nn.AvgPool2d([3, 3], [2, 2], padding=1)
    ],
}

lstm_size= 10
lstm_num_layers = 2
num_layers = 5
per_layer_types = 3

sampler = ArchSampler(
    lstm_size=lstm_size,
    lstm_num_layers=lstm_num_layers,
    num_layers=num_layers,
    per_layer_types=per_layer_types,
    bl_dec=0.1
).cuda()

inputs = torch.rand((4, 1, 28, 28), dtype=torch.float32).cuda()
# inputs2 = torch.rand((4, 8, 32, 32), dtype=torch.float32).cuda()
lc = LayerCollection(layers, num_layers, per_layer_types)
stem_layer = nn.Conv2d(1, 8, [1, 1])
output_layer = nn.Sequential(
    nn.Flatten(),
    nn.Linear(49, 10)
)
builder = ArchBuilder(stem_layer, lc, output_layer).cuda()



import numpy as np
def tran_func(bxs, bys):
    bx = bxs[0]
    by = bys[0]
    bx = bx.astype(np.float32) / 255.
    bx = np.expand_dims(bx, axis=0)
    by = np.squeeze(by.astype(np.int64))
    return (bx,), by

dstv = dl.DataReader('../data/mnist/mnist.h5', transform_func=tran_func, num_workers=0)

gntv = dstv.common_cls_reader(16, selected_classes=['valid'], shuffle=False)
metric = dl.MetricAccuracy(device=0, name='acc')


enas_model = EnasModel(sampler, builder, None, None, device=0)

arch_seqs = []
ps = []
sampler_losses = []
rs = []
optimizer = torch.optim.Adam(sampler.parameters())
gs = [sampler.g_emb.detach()]

for i in range(5):
    arch_seq = sampler()
    arch_seqs.append(arch_seq)
    ps.append(sampler.probs)
    sampler_losses.append(sampler.sample_log_prob)
    print(arch_seq)
    reward = enas_model.get_reward(arch_seq, gntv, metric, i+1)
    rs.append(reward)
    loss = sampler.get_loss(reward)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    gs.append(sampler.g_emb)
    print(sampler.baseline)
    print(loss)


