import os
import json
import numpy as np
from skimage.io import imread
from skimage.transform import resize

from ipdb import set_trace
set_trace()

def read_skc_subset(root_dir, mode='train1', sample_size=None):
    # Read filename list of subset and annotation
    if mode == 'total':
        set_filename_list = []
        for set_txt_filename in ['train1.txt', 'validation1.txt', 'test.txt']:
            set_filename_path = os.path.join(root_dir, 'imageset', 'imageset1', set_txt_filename)
            with open(set_filename_path, 'r') as fid:
                set_filename_list += fid.read().split('\n')[:-1]
    else: # mode == 'train#' or 'validation#' or 'test'
        set_txt_filename = '{}.txt'.format(mode)
        set_filename_path = os.path.join(root_dir, 'imageset', 'imageset1', set_txt_filename)
        with open(set_filename_path, 'r') as fid:
            set_filename_list = fid.read().split('\n')[:-1]
    set_size = len(set_filename_list)

    if sample_size is not None and sample_size < set_size:
        # Randomly sample subset of data when sample_size is specified
        set_filename_list = np.random.choice(set_filename_list, size=sample_size, replace=False)
        set_size = sample_size

    # Read annotations
    anno_path = os.path.join(root_dir, 'annotation', 'annotation1.json')
    with open(anno_path, 'r') as fid:
        anno_dict = json.load(fid)
    # labels = anno_dict['images'][filename]['class']

    return set_filename_list, anno_dict

class Dataset(object):
    def __init__(self, root_dir, filename_list, anno_dict, binary=True, augment=None):
        self.root_dir = root_dir
        self.file_dir = os.path.join(self.root_dir, 'image')
        self.filename_list = filename_list
        self.anno_dict = anno_dict
        self.binary = binary
        self.augment = augment

    def __getitem__(self, index):
        return self.load_data(self.filename_list[index])

    def __len__(self):
        return len(self.filename_list)

    def load_data(self, filename):
        # Simply load data (X and y)
        # transpose, normalize, resize, augment, etc. will be done in minibatch-wise (NOT HERE)
        X = imread(os.path.join(self.file_dir, filename))
        label = self.anno_dict['images'][filename]['class'][0] # we only provide a single label case
        y = 1 if self.binary and label > 0 else label

        #X = np.transpose(X, [2,0,1])
        #y = np.transpose(y, [2,0,1])

        return X, y

from torch.utils.data import DataLoader

root_dir = '/Data/skc/20180424/original/' # FIXME

filename_list, anno_dict = read_skc_subset(root_dir, sample_size=None)
dataset = Dataset(root_dir, filename_list, anno_dict, binary=True, augment=None)

dataloader = DataLoader(dataset, 10, shuffle=True, num_workers=1)

for X,y in dataloader:
    set_trace()
    # check X, y numpy
