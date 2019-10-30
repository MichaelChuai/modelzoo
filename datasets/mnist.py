import numpy as np
import dlutil as dl

root = '/data/examples/mnist'

ds = np.load(f'{root}/mnist.npz')
xtr = ds['x_train']
ytr = ds['y_train']
xte = ds['x_test']
yte = ds['y_test']


dl.write_h5_from_arrays(f'{root}/mnist_tr.h5', [xtr], [ytr])
dl.write_h5_from_arrays(f'{root}/mnist_te.h5', [xte], [yte])

