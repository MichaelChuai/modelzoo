import random
import collections
from evo.search_space import *

dim = 100
noise_stdev = 0.01

class Model:
    def __init__(self):
        self.arch = None
        self.value = None
    
    def __str__(self):
        return f'{self.arch:b}'

    @classmethod
    def sum_bits(cls, arch):
        total = 0
        
        return total

    @classmethod
    def random_arch(cls):
        nc = cell_seq_gen(11)
        rc = cell_seq_gen(11)

    @classmethod
    def train_and_eval(cls, arch):
        accuracy = cls.sum_bits(arch) / dim
        accuracy += random.gauss(0.0, noise_stdev)
        accuracy = 0.0 if accuracy < 0.0 else accuracy
        accuracy = 1.0 if accuracy > 1.0 else accuracy
        return accuracy

    @classmethod
    def mutate_arch(cls, prev_arch):
        
        return arch

# 15821561938

cycles = 1000
population_size = 100
sample_size = 10

population = collections.deque()
history = []


# Initialize the population
while len(population) < population_size:
    model = Model()
    model.arch = Model.random_arch()
    model.accuracy = Model.train_and_eval(model.arch)
    population.append(model)
    history.append(model)

# Carry out evolution in cycles. Each cycle produces a model and removes another.

while len(history) < cycles:
    sample = []
    while len(sample) < sample_size:
        candidate = random.choice(list(population))
        sample.append(candidate)
    parent = max(sample, key=lambda i: i.accuracy)
    print(parent.arch, parent.accuracy)
    child = Model()
    child.arch = Model.mutate_arch(parent.arch)
    child.accuracy = Model.train_and_eval(child.arch)
    population.append(child)
    history.append(child)
    population.popleft()

