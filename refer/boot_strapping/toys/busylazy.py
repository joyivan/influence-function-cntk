import os
import numpy as np
import cntk as C
import pickle as pkl
from torch.utils.data import DataLoader

#from datasets import dataset as dataset # FIXME
from datasets import dataset28 as dataset # FIXME
from models.nn import ResNet_18 as ConvNet
from learning.optimizers import MomentumOptimizer as Optimizer
from learning.evaluators import ErrorRateEvaluator as Evaluator

import time

from cntk.device import try_set_default_device, gpu
try_set_default_device(gpu(0))

from ipdb import set_trace

set_trace()

""" 1. Load and split datasets """
### FIXME ###
root_dir = '/Data/emnist/balanced/original/'

data_list, anno_dict = dataset.read_data_subset(root_dir, mode='test')
print(len(data_list))
busy_data_set = dataset.BusyDataset(root_dir, data_list, anno_dict)
lazy_data_set = dataset.LazyDataset(root_dir, data_list, anno_dict)

X,y = busy_data_set.__getitem__(0)

busydataloader = DataLoader(busy_data_set, 128, shuffle=False, num_workers=2)
lazydataloader = DataLoader(lazy_data_set, 128, shuffle=False, num_workers=2)

set_trace()

t1 = time.time()

ct = 0
for i in range(10):
    for X,y in lazydataloader:
        # check X,y
        #print(np.argmax(y))
        ct += 1
print(ct)
print('lazy one takes', time.time()-t1)

t2 = time.time()

ct = 0
for i in range(10):
    for X,y in busydataloader:
        # check X,y
        #print(X.shape)
        ct += 1
print(ct)
print('busy one takes', time.time()-t2)

print('EOS')
