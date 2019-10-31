import os
import torch.utils.data as Data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class OmniglotRaw(Data.Dataset):
    def find_classes(self, root_dir):
        retour = []
        for (root, dirs, files) in os.walk(root_dir):
            for f in files:
                if f.endswith('png'):
                    r = root.split('/')
                    retour.append((f, r[-2]+'/' + r[-1], root))
        print(f'Found {len(retour)} items')
        return retour

    def index_classes(self, items):
        idx = {}
        for i in items:
            if (not i[1] in idx):
                idx[i[1]] = len(idx)
        print(f'Found {len(idx)} classes')
        return idx

    def __init__(self, root, transform=None, label_transform=None):
        self.root = root
        self.transform = transform
        self.label_transform = label_transform
        self.all_items = self.find_classes(root)
        self.idx_classes = self.index_classes(self.all_items)
        self.rev_idx_classes = {
            self.idx_classes[k]: k for k in self.idx_classes}

    def __len__(self):
        return len(self.all_items)

    def __getitem__(self, index):
        filename = self.all_items[index][0]
        img = f'{self.all_items[index][2]}/{filename}'
        label = self.idx_classes[self.all_items[index][1]]
        if self.transform is not None:
            img = self.transform(img)
        if self.label_transform is not None:
            label = self.label_transform(label)
        return img, label


root = '/data/examples/omniglot'


# def filenameToPILImage(x): return Image.open(x).convert('L')


# def PiLImageResize(x): return x.resize((28, 28))


# def np_reshape(x): return np.reshape(x, (28, 28))


# ds = OmniglotRaw(root, transform=transforms.Compose(
#     [filenameToPILImage, PiLImageResize, np_reshape]))

# temp = {}
# for (img, label) in ds:
#     if label in temp:
#         temp[label].append(img)
#     else:
#         temp[label] = [img]

# cs = []
# for c in temp:
#     cs.append(np.array(temp[c]))
# cs = np.array(cs)

# x = cs.reshape((-1, 28, 28))
# y = np.tile(np.arange(cs.shape[0])[None].T, 20).reshape((-1))

import joblib

# joblib.dump((cs, x, y, ds.rev_idx_classes), f'{root}/omniglot.pkl')


classes_per_set = 5 # way
samples_per_class = 10 # shot


cs, _, _, idx_dict = joblib.load(f'{root}/omniglot.pkl')
x = cs
num_classes = x.shape[0]
shuffle_classes = np.arange(num_classes)
np.random.shuffle(shuffle_classes)
x = x[shuffle_classes]

num_samples = classes_per_set * samples_per_class

data_cache = []
batch_size = 32
ds_size = 10
# for sample in range(ds_size):
support_set_x = np.zeros((batch_size, num_samples, 28, 28))
support_set_y = np.zeros((batch_size, num_samples))
target_x = np.zeros((batch_size, samples_per_class, 28, 28))
target_y = np.zeros((batch_size, samples_per_class))
for i in range(batch_size):
    pinds = np.random.permutation(num_samples)
    classes = np.random.choice(x.shape[0], classes_per_set, False)
    x_hat_class = np.random.choice(classes, samples_per_class, True)
    pinds_test = np.random.permutation(samples_per_class)
    ind = 0
    ind_test = 0
    for j, cur_class in enumerate(classes):
        if cur_class in x_hat_class:
            n_test_samples = np.sum(cur_class == x_hat_class)
            example_inds = np.random.choice(x.shape[1], samples_per_class + n_test_samples, False)
        else:
            example_inds = np.random.choice(x.shape[1], samples_per_class, False)
        
        for eind in example_inds[:samples_per_class]:
            support_set_x[i, pinds[ind], :, :] = x[cur_class][eind]
            support_set_y[i, pinds[ind]] = j
            ind += 1

        for eind in example_inds[samples_per_class:]:
            target_x[i, pinds_test[ind_test], :, :] = x[cur_class][eind]
            target_y[i, pinds_test[ind_test]] = j
            ind_test += 1

