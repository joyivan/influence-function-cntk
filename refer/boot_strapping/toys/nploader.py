import numpy as np
import torch
from torch.utils.data import DataLoader

from ipdb import set_trace

class Dataset(object):
    def __init__(self):
        self._images = np.ones((1000,28,28,1))
        self._labels = np.ones((1000,47))

    def __getitem__(self, index):
        #return self._images[index], self._labels[index]
        image = np.transpose(self._images[index], (2,0,1))
        label = self._labels[index]
        return image, label

    def __len__(self):
        return len(self._images)

from datasets import dataset as dataset

dataset = dataset.Dataset('.', [], dict())

dataloader = DataLoader(dataset, 10, shuffle=True, num_workers=1)

for X,y in dataloader:
    set_trace()
    # check X,y numpy

    print(X.shape)


