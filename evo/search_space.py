import numpy as np
import random
from copy import deepcopy
import networkx as nx
import torch
import torch.nn as nn

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





def cell_gen(num_ops, num_round=5):
    lst = [[0], [1]]
    whole_set = set(range(2+num_round))
    sampled_lst = []
    for i in range(2, num_round+2):
        nodes = np.random.choice(i, 2).tolist()
        ops = np.random.randint(0, num_ops, 2).tolist()
        sample = sorted(zip(nodes, ops))
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
    num_round = len(normal_cell) - 3
    mut_type = np.random.randint(2)  # 0 for hidden state mutation, 1 for op mutation
    mut_cell_type = np.random.randint(2) # 0 for normal cell, 1 for reduction cell
    mut_round = np.random.randint(num_round)
    mut_loc = np.random.randint(2)
    mut_cell = [normal_cell, reduction_cell][mut_cell_type]
    if mut_type == 0:
        new_node = np.random.randint(2 + mut_round)
        mut_cell[2 + mut_round][mut_loc][0] = new_node
        mut_cell[2 + mut_round] = sorted(mut_cell[2 + mut_round])
        whole_set = set(range(2+num_round))
        sampled_set = set()
        for i in range(2, 2+num_round):
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
    num_round = len(cell) - 3
    g = nx.DiGraph()
    g.add_nodes_from(range(num_round+3))
    for i in range(2, num_round+2):
        g.add_edges_from([(cell[i][0][0], i), (cell[i][1][0], i)])
    g.add_edges_from([(i, 2+num_round) for i in cell[-1]])
    return g






# nc = cell_gen(11)
# rc = cell_gen(11)
# anc, arc = mutate_cell(nc, rc, 11)

# print(nc)
# print(anc)
# print(rc)
# print(arc)

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




