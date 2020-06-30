import numpy as np
import dlutil as dl

# root = '/data/examples/mnist'
root = '../../data/mnist'

ds = np.load(f'{root}/mnist.npz')
xtr = ds['x_train'][:-5000]
ytr = ds['y_train'][:-5000]
xtv = ds['x_train'][-5000:]
ytv = ds['y_train'][-5000:]
xte = ds['x_test']
yte = ds['y_test']


dt = {
    'train': [[xtr], [ytr]],
    'valid': [[xtv], [ytv]],
    'test': [[xte], [yte]]
}

dl.write_classified_h5_from_arrays(f'{root}/mnist.h5', dt)
# dl.write_h5_from_arrays(f'{root}/mnist_tr.h5', [xtr], [ytr])
# dl.write_h5_from_arrays(f'{root}/mnist_te.h5', [xte], [yte])

