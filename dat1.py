import torch
import torch.nn as nn
import numpy as np
from darts.search_space import *



network = FactorizedReduction(10, 20)

x = torch.rand(3, 10, 20, 20)

q = network(x)
print(q[0].shape)
print(q[1].shape)



