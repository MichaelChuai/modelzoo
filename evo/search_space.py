import numpy as np
from copy import deepcopy
import networkx as nx
import torch
import torch.nn as nn
from collections import namedtuple

'''
Possible ops:
1.  none   
2.  1x3 & 3x1 conv
3.  1x7 & 7x1 conv   
4.  3x3 dconv
5.  3x3 avgpool
6.  3x3 maxpool
7.  1x1 conv
8.  3x3 conv
9.  3x3 sconv
10. 5x5 sconv
11. 7x7 sconv
'''

def cell_gen(num_ops, num_rounds=5):
    lst = [[0], [1]]
    whole_set = set(range(2+num_rounds))
    sampled_lst = []
    for i in range(2, num_rounds+2):
        nodes = np.random.choice(i, 2).tolist()
        ops = np.random.randint(0, num_ops, 2).tolist()
        sample = list(zip(nodes, ops))
        sample = [list(tup) for tup in sample]
        sampled_lst.extend(nodes)
        lst.append(sample)
    sampled_set = set(sampled_lst)
    out_set = whole_set - sampled_set
    out_lst = sorted(list(out_set))
    lst.append(out_lst)
    return lst

def mutate_cell(normal_cell, reduction_cell, num_ops=None):
    normal_cell = deepcopy(normal_cell)
    reduction_cell = deepcopy(reduction_cell)
    num_rounds = len(normal_cell) - 3
    mut_type = np.random.randint(2)  # 0 for hidden state mutation, 1 for op mutation
    mut_cell_type = np.random.randint(2) # 0 for normal cell, 1 for reduction cell
    mut_round = np.random.randint(num_rounds)
    mut_loc = np.random.randint(2)
    mut_cell = [normal_cell, reduction_cell][mut_cell_type]
    if mut_type == 0:
        cur_node = mut_cell[2 + mut_round][mut_loc][0]
        new_node = np.random.choice([i for i in range(2+mut_round) if i != cur_node])
        mut_cell[2 + mut_round][mut_loc][0] = new_node
        whole_set = set(range(2+num_rounds))
        sampled_set = set()
        for i in range(2, 2+num_rounds):
            sampled_set.add(mut_cell[i][0][0])
            sampled_set.add(mut_cell[i][1][0])
        out_set = whole_set - sampled_set
        out_lst = sorted(list(out_set))
        mut_cell[-1] = out_lst
    else:
        cur_op = mut_cell[2 + mut_round][mut_loc][1]
        new_op = np.random.choice([i for i in range(num_ops) if i != cur_op])
        mut_cell[2 + mut_round][mut_loc][1] = new_op
    return normal_cell, reduction_cell

def cell_to_graph(cell):
    num_rounds = len(cell) - 3
    g = nx.DiGraph()
    g.add_nodes_from(range(num_rounds+3))
    for i in range(2, num_rounds+2):
        g.add_edges_from([(cell[i][0][0], i), (cell[i][1][0], i)])
    g.add_edges_from([(i, 2+num_rounds) for i in cell[-1]])
    return g



def ConvOp(in_channels, 
           out_channels,
           kernel_size,
           stride=1,
           padding=0,
           dilation=1,
           groups=1):
           return nn.Sequential(
               nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups),
               nn.BatchNorm2d(out_channels)
           )

class IdentityOp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IdentityOp, self).__init__()

    def forward(self, x):
        return x


class Conv1331Op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1331Op, self).__init__()
        self.conv1 = ConvOp(in_channels, out_channels, kernel_size=[1, 3], padding=[0, 1])
        self.conv2 = ConvOp(out_channels, out_channels, kernel_size=[3, 1], padding=[1, 0])

    def forward(self, x):
        out = self.conv1(x)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        return out

class Conv1771Op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1771Op, self).__init__()
        self.conv1 = ConvOp(in_channels, out_channels, kernel_size=[1, 7], padding=[0, 3])
        self.conv2 = ConvOp(out_channels, out_channels, kernel_size=[7, 1], padding=[3, 0])

    def forward(self, x):
        out = self.conv1(x)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        return out

class DilConvOp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilConvOp, self).__init__()
        self.dilconv = ConvOp(in_channels, out_channels, kernel_size=[3, 3], padding=[2, 2], dilation=2)
    
    def forward(self, x):
        return self.dilconv(x)

class AvgPoolOp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AvgPoolOp, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=[3, 3], stride=[1, 1], padding=[1, 1])
    
    def forward(self, x):
        return self.avgpool(x)

class MaxPoolOp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MaxPoolOp, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=[3, 3], stride=[1, 1], padding=[1, 1])
    
    def forward(self, x):
        return self.maxpool(x)

class Conv11Op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv11Op, self).__init__()
        self.conv = ConvOp(in_channels, out_channels, kernel_size=1, stride=1)
    
    def forward(self, x):
        return self.conv(x)

class Conv33Op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv33Op, self).__init__()
        self.conv = ConvOp(in_channels, out_channels, kernel_size=[3, 3], padding=[1, 1])
    
    def forward(self, x):
        return self.conv(x)

class SepConv33Op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SepConv33Op, self).__init__()
        self.sepconv = ConvOp(in_channels, out_channels, kernel_size=[3, 3], padding=[1, 1], groups=in_channels)
    
    def forward(self, x):
        return self.sepconv(x)

class SepConv55Op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SepConv55Op, self).__init__()
        self.sepconv = ConvOp(in_channels, out_channels, kernel_size=[5, 5], padding=[2, 2], groups=in_channels)
    
    def forward(self, x):
        return self.sepconv(x) 

class SepConv77Op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SepConv77Op, self).__init__()
        self.sepconv = ConvOp(in_channels, out_channels, kernel_size=[7, 7], padding=[3, 3], groups=in_channels)
    
    def forward(self, x):
        return self.sepconv(x)


class LayerCollection(nn.Module):
    def __init__(self, out_channels, num_cells, num_rounds):
        super(LayerCollection, self).__init__()
        ops = [
            IdentityOp,
            Conv1331Op,
            Conv1771Op,
            DilConvOp,
            AvgPoolOp,
            MaxPoolOp,
            Conv11Op,
            Conv33Op,
            SepConv33Op,
            SepConv55Op,
            SepConv77Op
        ]
        num_locs = 2
        num_ops = len(ops)
        cell_lst = []
        for _ in range(num_cells):
            round_lst = []
            for _ in range(num_rounds):
                loc_lst = []
                for _ in range(num_locs):
                    op_lst = []
                    for l in range(num_ops):
                        op_lst.append(ops[l](out_channels, out_channels))
                    loc_lst.append(nn.ModuleList(op_lst))
                round_lst.append(nn.ModuleList(loc_lst))
            cell_lst.append(nn.ModuleList(round_lst))
        self.cell_lst = nn.ModuleList(cell_lst)
        cell_output_lst = []
        for _ in range(num_cells):
            multiple_lst = []
            for i in range(1, 2+num_rounds):
                multiple_lst.append(Conv11Op(i * out_channels, out_channels))
            cell_output_lst.append(nn.ModuleList(multiple_lst))
        self.cell_output_lst = nn.ModuleList(cell_output_lst)

    def forward(self, x):
        return x


class CellBuilder(nn.Module):
    def __init__(self, idx_cell, layer_col: LayerCollection):
        super(CellBuilder, self).__init__()
        self.layer_candidates = layer_col.cell_lst[idx_cell]
        self.layer_output_candidates = layer_col.cell_output_lst[idx_cell]

    def forward(self, cell_seq, x0, x1):
        nodes = [x0, x1]
        num_rounds = len(cell_seq) - 3
        for i in range(num_rounds):
            loc0, loc1 = cell_seq[2+i]
            n0, o0 = loc0
            inp0 = nodes[n0]
            l0 = self.layer_candidates[i][0][o0]
            n1, o1 = loc1
            inp1 = nodes[n1]
            l1 = self.layer_candidates[i][0][o1]
            out = nn.ReLU()(l0(inp0) + l1(inp1))
            nodes.append(out)
        out_nodes = [nodes[i] for i in cell_seq[-1]]
        out = torch.cat(out_nodes, dim=1)
        out_l = self.layer_output_candidates[len(out_nodes)-1]
        out = nn.ReLU()(out_l(out))
        return out

ArchSeq = namedtuple('ArchSeq', ['normal_cell', 'reduction_cell'])

class ArchBuilder(nn.Module):
    def __init__(self, stem_module, num_classes, out_channels, normal_cell_num_lst, num_rounds):
        super(ArchBuilder, self).__init__()
        self.stem_module = stem_module # Output channels should be `out_channels`
        num_cells = sum(normal_cell_num_lst) + len(normal_cell_num_lst) - 1
        self.layer_col = LayerCollection(out_channels, num_cells, num_rounds)
        assert len(normal_cell_num_lst) >= 2, 'Architecture must contain at least one reduction cell and two normal cells!'
        cell_lst = []
        idx_cell = -1
        self.cell_state = {} # 0 for normal cell; 1 for reduction cell
        for i in range(len(normal_cell_num_lst)):
            n = normal_cell_num_lst[i]
            for j in range(n):
                idx_cell += 1
                if i > 0 and j == 0:
                    inp0_layer = nn.Sequential(
                        ConvOp(out_channels, out_channels, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1]),
                        nn.ReLU()
                    )
                else:
                    inp0_layer = IdentityOp(out_channels, out_channels)
                inp1_layer = IdentityOp(out_channels, out_channels)
                normal_cell = CellBuilder(idx_cell, self.layer_col)
                self.cell_state[idx_cell] = 0
                cell_lst.append(nn.ModuleList([inp0_layer, inp1_layer, normal_cell]))
            if i != len(normal_cell_num_lst) - 1:
                idx_cell += 1
                inp0_layer = nn.Sequential(
                        ConvOp(out_channels, out_channels, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1]),
                        nn.ReLU()
                )
                inp1_layer = nn.Sequential(
                        ConvOp(out_channels, out_channels, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1]),
                        nn.ReLU()
                )
                reduction_cell = CellBuilder(idx_cell, self.layer_col)
                self.cell_state[idx_cell] = 1
                cell_lst.append(nn.ModuleList([inp0_layer, inp1_layer, reduction_cell]))
        self.cell_lst = nn.ModuleList(cell_lst)
        assert len(self.cell_lst) == num_cells, f'Fatal error: {len(self.cell_lst)} != {num_cells}'
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.final_layer = nn.Linear(out_channels, num_classes)
    
    def forward(self, archseq, x):
        x = self.stem_module(x)
        cell_nodes = [x, x]
        out = None
        for i in range(len(self.cell_lst)):
            inp0_layer, inp1_layer, cell = self.cell_lst[i]
            m_x0 = inp0_layer(cell_nodes[i])
            m_x1 = inp1_layer(cell_nodes[i+1])
            if self.cell_state[i] == 0:
                cell_seq = archseq.normal_cell
            elif self.cell_state[i] == 1:
                cell_seq = archseq.reduction_cell
            else:
                raise RuntimeError(f'Invalid cell state {self.cell_state[i]}')
            out = cell(cell_seq, m_x0, m_x1)
            cell_nodes.append(out)
        self.cell_nodes = cell_nodes
        out = self.avgpool(out)
        out = nn.Flatten()(out)
        out = self.final_layer(out)
        return out






# nc = cell_gen(11)
# rc = cell_gen(11)
# x = torch.rand(20, 3, 16, 16)
# y = torch.randint(5, (20,))
# stem = nn.Sequential(
#     ConvOp(3, 32, kernel_size=1),
#     nn.ReLU()
# )
# ab = ArchBuilder(stem, 5, 32, [2, 2, 2], 5).cuda(0)
# t = ab(nc, rc, x.cuda(0))

# anc, arc = mutate_cell(nc, rc, num_ops=11)
# t1 = ab(anc, arc, x)

# gnc = cell_to_graph(nc)
# ganc = cell_to_graph(anc)
# grc = cell_to_graph(rc)
# garc = cell_to_graph(arc)

# import matplotlib.pyplot as plt

# options = {
#     'node_color': 'red'
# }

# plt.subplot(221)
# g = gnc
# pos = nx.circular_layout(g)
# nx.draw(g, pos, with_labels=True, **options)
# plt.subplot(222)
# g = ganc
# pos = nx.circular_layout(g)
# nx.draw(g, pos, with_labels=True, **options)
# plt.subplot(223)
# g = grc
# pos = nx.circular_layout(g)
# nx.draw(g, pos, with_labels=True, **options)
# plt.subplot(224)
# g = garc
# pos = nx.circular_layout(g)
# nx.draw(g, pos, with_labels=True, **options)
# plt.show()




