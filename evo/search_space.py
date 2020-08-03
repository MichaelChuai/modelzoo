import numpy as np
import random
from copy import deepcopy
import networkx as nx

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
9.  3x3 sconvl
10. 5x5 sconv
11. 7x7 sconv
'''


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




