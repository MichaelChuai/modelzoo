import numpy as np
import torch
from meta.matching import *
import dlutil as dl


root = '/data/examples/omniglot/omniglot_bg.h5'

dstr = dl.DataReader(root)

a, b = dstr.few_shot_reader(20,)

# G = Embedding(1, 10)



