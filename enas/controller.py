import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from tqdm import trange
import dlutil as dl

class Indexer(nn.Module):
    def __init__(self, lstm_size, lstm_num_layers):
        super(Indexer, self).__init__()
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.net = nn.LSTM(self.lstm_size, hidden_size=self.lstm_size, num_layers=self.lstm_num_layers, bias=False, batch_first=True)

    def forward(self, inputs, prev_state):
        if prev_state is None:
            next_h, next_state = self.net(inputs)
        else:
            next_h, next_state = self.net(inputs, prev_state)
        return next_h, next_state



class ArchSampler(nn.Module):
    def __init__(self, lstm_size, lstm_num_layers, num_layers, per_layer_types):
        super(ArchSampler, self).__init__()
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.num_layers = num_layers
        self.per_layer_types = per_layer_types
        self.g_emb = nn.Parameter(torch.empty((1, self.lstm_size), dtype=torch.float32), requires_grad=True)
        nn.init.xavier_normal_(self.g_emb)
        self.w_emb = nn.Parameter(torch.empty((self.num_layers, self.per_layer_types, self.lstm_size), dtype=torch.float32), requires_grad=True)
        nn.init.xavier_normal_(self.w_emb)
        self.w_soft = nn.Parameter(torch.empty((self.lstm_size, self.per_layer_types), dtype=torch.float32), requires_grad=True)
        nn.init.xavier_normal_(self.w_soft)
        self.indexer = Indexer(self.lstm_size, self.lstm_num_layers)
        

    def forward(self):
        self.arch_seq = []
        inputs = self.g_emb[None]
        prev_state = None
        log_probs = []
        for layer_i in range(self.num_layers):
            next_h, next_state = self.indexer(inputs, prev_state)
            prev_state = next_state
            logit = torch.matmul(next_h[:,0,:], self.w_soft)
            prob = torch.softmax(logit, dim=1)
            layer_type_id = torch.multinomial(prob[0], 1)
            log_prob = nn.CrossEntropyLoss()(logit, layer_type_id)
            log_probs.append(log_prob[None])
            inputs = self.w_emb[layer_i][layer_type_id[0]][None][None]
            if layer_type_id.device.type != 'cpu':
                self.arch_seq.append(layer_type_id.cpu().numpy())
            else:
                self.arch_seq.append(layer_type_id.numpy())

        self.sample_log_prob = torch.mean(torch.cat(log_probs))
        return self.arch_seq


class LayerCollection(nn.Module):
    def __init__(self, layers_lst_dict, num_layers, per_layer_types):
        '''
            `layers_lst_dict` shoule be dictionary containing lists of each layer with 0-based layer indices as keys.
            Examples:
                {
                    0: [
                        Conv1,
                        Conv2
                    ],
                    1: [
                        MaxPool1,
                        AvePool1
                    ],...
                }
        '''
        super(LayerCollection, self).__init__()
        self.layer_lst = []
        for i in range(num_layers):
            layer_lst = layers_lst_dict[i]
            assert len(layer_lst) <= per_layer_types, f'{per_layer_types} must not be less than number of types of each layer!'
            if len(layer_lst) == per_layer_types:
                self.layer_lst.append(nn.ModuleList(layer_lst))
            else:
                extend_layer_lst = []
                for j in range(per_layer_types):
                    idx = j % len(layer_lst)
                    if j == idx:
                        extend_layer_lst.append(layer_lst[idx])
                    else:
                        extend_layer_lst.append(deepcopy(layer_lst[idx]))
                assert len(extend_layer_lst) == per_layer_types
                self.layer_lst.append(nn.ModuleList(extend_layer_lst))
        self.layer_lst = nn.ModuleList(self.layer_lst)
    

    def get_layer(self, i, j):
        return self.layer_lst[i][j]

    def forward(self, x):
        return x



class ArchBuilder(nn.Module):
    def __init__(self, stem_layer, layer_col, output_layer):
        super(ArchBuilder, self).__init__()
        self.stem_layer = stem_layer
        self.layer_col = layer_col
        self.output_layer = output_layer

    def forward(self, arch_seq, *inputs):
        no_stem = False
        if self.stem_layer is not None:
            stem_output = self.stem_layer(*inputs)
        else:
            stem_output = inputs
            no_stem = True
        media_output = stem_output
        for i, seq in enumerate(arch_seq):
            print(i, seq)
            cur_layer = self.layer_col.get_layer(i, seq[0])
            if no_stem and (i==0):
                media_output = cur_layer(*media_output)
            else:
                media_output = cur_layer(media_output)
        if self.output_layer is not None:
            output = self.output_layer(media_output)
        else:
            output = media_output
        return output





class EnasController:
    def __init__(self, arch_sampler: ArchSampler, arch_builder: ArchBuilder, metric: dl.Metric, device=None):
        self.arch_sampler = arch_sampler
        self.arch_builder = arch_builder
        self.metric = metric
        self.device = device

    def get_reward(self, gntv, idx=0):
        dstv, steps_per_epoch = gntv
        progress_desc = f'Reward {idx}({self.metric.name})'
        ds_iter = iter(dstv)
        self.metric.reset()
        self.arch_builder.eval()
        # for _ in trange(steps_per_epoch, desc=progress_desc):
        #     bxs, bys = next(ds_iter)
        #     if self.device is not None:
        #         bxs = [bx.cuda(self.device) for bx in bxs]
        #         if type(bys) in (list, tuple):
        #             bys = [by.cuda(self.device) for by in bys]
        #         else:
        #             bys = bys.cuda(self.device)
        #     with torch.no_grad():
        #         by_ = 
            
        

lstm_size= 10
lstm_num_layers = 2
num_layers = 4
per_layer_types = 3

mc = ArchSampler(
    lstm_size=lstm_size,
    lstm_num_layers=lstm_num_layers,
    num_layers=num_layers,
    per_layer_types=per_layer_types
).cuda()

class CatLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CatLayer, self).__init__()
        self.l1 = nn.Conv2d(in_channel, out_channel//2, [1, 3], padding=[0, 1])
        self.l2 = nn.Conv2d(in_channel, out_channel//2, [3, 1], padding=[1, 0])
    
    def forward(self, x1, x2):
        r1 = self.l1(x1)
        r2 = self.l2(x2)
        r = torch.cat([r1, r2], dim=1)
        return r

layers = {
    0:[
        CatLayer(8, 8)
    ],
    1:[
        nn.Conv2d(8, 16, [3, 3], padding=1),
        nn.Conv2d(8, 16, [5, 5], padding=2)
    ],
    2:[
        nn.Conv2d(16, 32, [3, 3], padding=1),
        nn.Conv2d(16, 32, [5, 5], padding=2),
        nn.Conv2d(16, 32, [7, 7], padding=3)
    ],
    3:[
        nn.Conv2d(32, 64, [3, 3], padding=1),
        nn.Conv2d(32, 64, [5, 5], padding=2)
    ]
}

inputs = torch.rand((4, 8, 32, 32), dtype=torch.float32).cuda()
inputs2 = torch.rand((4, 8, 32, 32), dtype=torch.float32).cuda()
lc = LayerCollection(layers, num_layers, per_layer_types)
sampler = ArchBuilder(None, lc, None).cuda()

# ks = []
# ts = []
# slps = []
# for _ in range(5):
#     k = mc()
#     ks.append(k)
#     slps.append(mc.sample_log_prob)
#     t = sampler(k, inputs)
#     ts.append(t)


