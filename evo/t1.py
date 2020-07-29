import random

dim = 100
noise_stdev = 0.01

class Model:
    def __init__(self):
        self.arch = None
        self.accuracy = None
    
    def __str__(self):
        return f'{self.arch:b}'

    @classmethod
    def sum_bits(cls, arch):
        total = sum(map(int, bin(arch)[2:]))
        return total

    @classmethod
    def random_arch(cls):
        return random.randint(0, 2 ** dim - 1)

    @classmethod
    def train_and_eval(cls, arch):
        accuracy = cls.sum_bits(arch) / dim
        accuracy += random.gauss(0.0, noise_stdev)
        accuracy = 0.0 if accuracy < 0.0 else accuracy
        accuracy = 1.0 if accuracy > 1.0 else accuracy
        return accuracy

    @classmethod
    def mutate_arch(cls, prev_arch):
        position = random.randint(0, dim-1)
        arch = prev_arch ^ (1 << position)
        return arch


q = Model.random_arch()
acc = Model.train_and_eval(q)
q2 = Model.mutate_arch(q)
acc2 = Model.train_and_eval(q2)