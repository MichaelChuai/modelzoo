import numpy as np
from evo.search_space import *
from collections import namedtuple, deque


Arch = namedtuple('Arch', ['normal_cell', 'reduction_cell'])

def gen_arch(num_ops, num_round=5):
    normal_cell = cell_gen(num_ops, num_round)
    reduction_cell = cell_gen(num_ops, num_round)
    arch = Arch(normal_cell=normal_cell, reduction_cell=reduction_cell)
    return arch

def train_and_eval(arch):
    def cal_value(cell):
        total = 0
        for l in sum(cell, []):
            if type(l) is int:
                total += l
            else:
                total += sum(l)
        return total
    return cal_value(arch.normal_cell) + cal_value(arch.reduction_cell)

def mutate_arch(prev_arch, num_ops):
    normal_cell, reduction_cell = prev_arch
    m_normal_cell, m_reduction_cell = mutate_cell(normal_cell, reduction_cell, num_ops)
    return Arch(normal_cell=m_normal_cell, reduction_cell=m_reduction_cell)



cycles = 3000
population_size = 200
sample_size = 50

history = []
population = deque()

for _ in range(population_size):
    arch = gen_arch(num_ops=11, num_round=5)
    value = train_and_eval(arch)
    dt = {'arch': arch, 'value': value}
    population.append(dt)
    history.append(dt)

while len(history) < cycles:
    candidates = np.random.choice(np.arange(len(population)), sample_size, replace=False)
    samples = [population[i] for i in candidates]
    parent = max(samples, key=lambda dt: dt['value'])
    child_arch = mutate_arch(parent['arch'], num_ops=11)
    child_value = train_and_eval(child_arch)
    child = {'arch': child_arch, 'value': child_value}
    population.append(child)
    history.append(child)
    population.popleft()

