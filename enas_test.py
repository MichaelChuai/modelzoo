import numpy as np
import torch
import torch.nn as nn
import dlutil as dl
from enas import *

class CatLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(CatLayer, self).__init__()
        assert kernel_size % 2 == 1
        padding_size = (kernel_size - 1) // 2
        self.l1 = nn.Conv2d(in_channel, out_channel//2, [1, kernel_size], padding=[0, padding_size])
        self.l2 = nn.Conv2d(in_channel, out_channel//2, [kernel_size, 1], padding=[padding_size, 0])
    
    def forward(self, inputs):
        r1 = self.l1(inputs)
        r1 = nn.ReLU()(r1)
        r2 = self.l2(inputs)
        r2 = nn.ReLU()(r2)
        r = torch.cat([r1, r2], dim=1)
        return r

layers = {
    0:[
        nn.Sequential(nn.Conv2d(8, 16, [3, 3], padding=1), nn.ReLU()),
        nn.Sequential(nn.Conv2d(8, 16, [5, 5], padding=2), nn.ReLU())
    ],
    1:[
        nn.Sequential(nn.Conv2d(16, 8, [3, 3], padding=1), nn.ReLU()),
        nn.Sequential(nn.Conv2d(16, 8, [5, 5], padding=2), nn.ReLU()),
        nn.Sequential(nn.Conv2d(16, 8, [7, 7], padding=3), nn.ReLU())
    ],
    2:[
        nn.MaxPool2d([3, 3], [2, 2], padding=1),
        nn.AvgPool2d([3, 3], [2, 2], padding=1)
    ],
    3:[
        nn.Sequential(nn.Conv2d(8, 1, [3, 3], padding=1), nn.ReLU()),
        nn.Sequential(nn.Conv2d(8, 1, [5, 5], padding=2), nn.ReLU())
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
).cuda(0)

lc = LayerCollection(layers, num_layers, per_layer_types)
stem_layer = CatLayer(1, 8, kernel_size=3)
output_layer = nn.Sequential(
    nn.Flatten(),
    nn.Linear(49, 10)
)
builder = ArchBuilder(stem_layer, lc, output_layer).cuda(0)

def tran_func(bxs, bys):
    bx = bxs[0]
    by = bys[0]
    bx = bx.astype(np.float32) / 255.
    bx = np.expand_dims(bx, axis=0)
    by = np.squeeze(by.astype(np.int64))
    return (bx,), by



ds = dl.DataReader('../data/mnist/mnist.h5', transform_func=tran_func, num_workers=0)

gntr = ds.common_cls_reader(32, selected_classes=['train'], shuffle=True)
gntv = ds.common_cls_reader(32, selected_classes=['valid'], shuffle=False)
gnte = ds.common_cls_reader(32, selected_classes=['test'], shuffle=False)


optimizer = torch.optim.Adam(builder.parameters())
loss_func = nn.CrossEntropyLoss()
ckpt = dl.Checkpoint('temp/enas_t1/builder', max_to_keep=10, device=0, save_best_only=True, saving_metric='test_acc')
acc = dl.MetricAccuracy(device=0, name='acc')
total_steps = 20000
ckpt_steps = 1000
summ_steps = 50
listeners = [EnasListener('test', gnte, [acc])]

sampler_ckpt = dl.Checkpoint('temp/enas_t1/sampler', max_to_keep=10, device=0)
sampler_optim = torch.optim.Adam(sampler.parameters())
steps_before_training_sampler = 2000
sampler_training_interval = 1000
sampler_total_steps = 100
sampler_summ_steps = 10


enas_model = EnasModel(sampler, builder, sampler_ckpt, ckpt, device=0)

enas_model.train(gntr,
                 loss_func,
                 optimizer,
                 total_steps=total_steps,
                 ckpt_steps=ckpt_steps,
                 metrics=[acc],
                 summ_steps=summ_steps,
                 listeners=listeners,
                 from_scratch=True,
                 train_sampler=True,
                 sampler_optimizer=sampler_optim,
                 sampler_gntv=gntv,
                 sampler_metric=acc,
                 steps_before_training_sampler=steps_before_training_sampler,
                 sampler_training_interval=sampler_training_interval,
                 sampler_total_steps=sampler_total_steps,
                 sampler_summ_steps=sampler_summ_steps,
                 sampler_from_scratch=True
                 )




