import torch.utils.data as Data
import numpy as np
import dlutil as dl

sampler = dl.InfiniteRandomSampler(np.arange(100))

dlo = Data.DataLoader(np.arange(100), sampler=sampler, batch_size=5)

i = 0
for x in dlo:
    print(x)
    i += 1
    print(i)
    if i > 10:
        break
t = x.numpy()
s = [i for i in t]
