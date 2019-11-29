import dlutil as dl
from nlp.transformer import *
import numpy as np




batch_size = 64
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
drop_prob = 0.1


def input_trans(bxs, bys):
    bx, = bxs
    by, = bys
    bx = bx.astype(np.int64)
    by = by.astype(np.int64)
    return ((bx, by[:-1]), by[1:])

file_root = '/data/examples/pt2en'
tokenizer = torch.load(f'{file_root}/tokenizer.pt')
enc_vocab_size = len(tokenizer[0][1]) + 2
dec_vocab_size = len(tokenizer[1][1]) + 2

dstr = dl.DataReader(f'{file_root}/pt2en_tr.h5', transform_func=input_trans, num_workers=5)
gntr = dstr.common_reader(batch_size)
dstv = dl.DataReader(f'{file_root}/pt2en_tv.h5', transform_func=input_trans, num_workers=5)
gntv = dstv.common_reader(batch_size * 3, shuffle=False)


network = Transformer(num_layers, num_heads, d_model, dff, enc_vocab_size, dec_vocab_size).cuda(0)

optimizer = torch.optim.Adam(network.parameters())

class CustomSchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(CustomSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        arg1 = 1. / (np.sqrt(self.last_epoch) + 1e-7)
        arg2 = self.last_epoch * (self.warmup_steps ** -1.5)
        lr = 1. / np.sqrt(self.d_model) * np.minimum(arg1, arg2)
        return [lr for _ in self.base_lrs]
scheduler = CustomSchedule(optimizer, d_model, warmup_steps=100)

ckpt = dl.Checkpoint('results/pt2en_transformer', max_to_keep=10, save_best_only=True, saving_metric='test_acc', device=0)
acc = dl.MetricAccuracy(name='acc', device=0)
listeners = [dl.Listener('test', gntv, [acc])]

def loss_func(predictions, labels):
    batch_size = predictions.size(0)
    loss_ = nn.CrossEntropyLoss(reduction='none')(predictions.transpose(-1, -2), labels)
    mask = (labels != 0).reshape((batch_size, -1)).type(loss_.dtype)
    loss_ *= mask
    return torch.mean(loss_)

dlmodel = dl.DlModel(network, ckpt)
dlmodel.train(gntr, loss_func, optimizer, num_epochs=100, metrics=[acc], listeners=listeners, from_scratch=True, scheduler=scheduler)

