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
        total = 0
        for _ in range(dim):
            total += arch
