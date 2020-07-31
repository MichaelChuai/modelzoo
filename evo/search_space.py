import numpy as np
import random
from copy import deepcopy

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
        nodes = np.random.choice(i, 2)
        ops = np.random.randint(0, num_ops, 2)
        sample = sorted(nodes.tolist()) + ops.tolist()
        sampled_lst.extend(nodes.tolist())
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
    # if mut_type == 0:

a = cell_gen(11)
b = deepcopy(a)

        





# a = cell_seq_gen(11)
# print(a)


