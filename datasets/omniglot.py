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


def filenameToPILImage(x): return Image.open(x).convert('L')


def PiLImageResize(x): return x.resize((28, 28))


def np_reshape(x): return np.reshape(x, (28, 28))


ds = OmniglotRaw(root, transform=transforms.Compose(
    [filenameToPILImage, PiLImageResize, np_reshape]))

temp = {}
for (img, label) in ds:
    if label in temp:
        temp[label].append(img)
    else:
        temp[label] = [img]

ds = []
for c in temp:
    ds.append(np.array(temp[c]))
ds = np.array(ds)
# temp = []
