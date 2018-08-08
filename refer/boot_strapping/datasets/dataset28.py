import os
import json
import numpy as np
import torch
from skimage.io import imread
from skimage.transform import resize
import random
import time

def read_data_subset(root_dir, mode='train1', sample_size=None):
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

    # Shuffle filename list
    random.shuffle(set_filename_list)

    # Read annotations
    anno_path = os.path.join(root_dir, 'annotation', 'annotation1.json')
    with open(anno_path, 'r') as fid:
        anno_dict = json.load(fid)
    # labels = anno_dict['images'][filename]['class']

    return set_filename_list, anno_dict

class LazyDataset(object):
    def __init__(self, root_dir, filename_list, anno_dict, binary=False, augment=False):
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
        # transpose, augment, etc. will be done in minibatch-wise (NOT HERE)
        X = imread(os.path.join(self.file_dir, filename))
        X = np.float32(np.multiply(X, 1.0/255.0)) #<- resize will do
        #X = np.float32(resize(X, (224, 224), mode='constant')) ## RGB always?
        if len(X.shape) == 2:
            #X = np.tile(X, (3,1,1))
            #X = np.transpose(X, (1,2,0))
            X = np.expand_dims(X, axis=2)
        X = np.transpose(X, [2,0,1])
        if self.augment:
            # randomly rotate the image X in x, y axis
            rotate = pow(-1,np.random.randint(2,size=2))
            X = X[::,::rotate[0],::rotate[1]]
        
        label = self.anno_dict['images'][filename]['class'][0] # we only provide a single label case
        label = 1 if self.binary and label > 0 else label
        num_cls = 2 if self.binary else len(self.anno_dict['classes'])
        y = np.zeros(num_cls,dtype=np.float32)
        y[label] = 1.

        return X, y

class BusyDataset(object):
    def __init__(self, root_dir, filename_list, anno_dict, binary=False, augment=False):
        self.root_dir = root_dir
        self.file_dir = os.path.join(self.root_dir, 'image')
        self.filename_list = filename_list
        self.anno_dict = anno_dict
        self.binary = binary
        self.augment = augment

        self._images, self._labels = self.load_dataset(filename_list)

    def __getitem__(self, index):
        #return self.load_data(self.filename_list[index])
        #X = np.float32(resize(self._images[index], (224,224), mode='constant'))
        X = self._images[index]
        X = np.transpose(X, (2,0,1))
        if self.augment:
            # randomly rotate the image X in x, y axis
            rotate = pow(-1,np.random.randint(2,size=2))
            X = X[::,::rotate[0],::rotate[1]]
        y = self._labels[index]
        
        return X, y

    def __len__(self):
        return len(self.filename_list)

    def load_data(self, filename):
        # Simply load data (X and y)
        # transpose, augment, etc. will be done in minibatch-wise (NOT HERE)
        X = imread(os.path.join(self.file_dir, filename))
        X = np.float32(np.multiply(X, 1.0/255.0)) # <- resize will do
        #X = np.float32(resize(X, (224, 224), mode='constant')) ## RGB always? # FIXME
        if len(X.shape) == 2:
            #X = np.tile(X, (3,1,1))
            #X = np.transpose(X, (1,2,0))
            X = np.expand_dims(X, axis=2)
        #X = np.transpose(X, [2,0,1])
        
        label = self.anno_dict['images'][filename]['class'][0] # we only provide a single label case
        label = 1 if self.binary and label > 0 else label
        num_cls = 2 if self.binary else len(self.anno_dict['classes'])
        y = np.zeros(num_cls,dtype=np.float32)
        y[label] = 1.

        return X, y

    def load_dataset(self, filename_list):
        # FIXME
        t1 = time.time()
        images = []; labels = []
        for filename in filename_list:
            X,y = self.load_data(filename)
            images.append(X)
            labels.append(y)
        images = np.stack(images, axis=0)
        labels = np.stack(labels, axis=0)

        print('loading the overall raw dataset is done, and it takes {} sec'.format(time.time()-t1))

        return images, labels

#class Dataset(object):
#    def __init__(self, root_dir, filename_list, anno_dict, binary=False, augment=False):
#        self.root_dir = root_dir
#        self.file_dir = os.path.join(self.root_dir, 'image')
#        self.filename_list = filename_list
#        self.anno_dict = anno_dict
#        self.binary = binary
#        self.augment = augment
#
#        #self._images, self._labels = self.load_dataset(filename_list)
#        #import pandas as pd
#        #data = pd.read_csv('/Data/emnist/raw/emnist-balanced-train.csv')[0:1000]
#        #self._images = np.array(map(lambda x: x.reshape((28,28,1)), data[:,1:]))
#        #self._labels = data[:,0]
#
#        self._images = np.ones((len(filename_list), 28, 28,1))
#        self._labels = np.ones((len(filename_list), 47))
#
#    #def __getitem__(self, index):
#    #    #return self.load_data(self.filename_list[index])
#    #    X = np.float32(resize(self._images[index], (224,224), mode='constant'))
#    #    X = np.transpose(X, (2,0,1))
#    #    if self.augment:
#    #        # randomly rotate the image X in x, y axis
#    #        rotate = pow(-1,np.random.randint(2,size=2))
#    #        X = X[::,::rotate[0],::rotate[1]]
#    #    y = self._labels[index]
#    #    return torch.from_numpy(X), torch.from_numpy(y)
#
#    def __getitem__(self, index):
#        return self._images[index], self._labels[index]
#
#    def __len__(self):
#        return len(self._images)
#
#    #def load_data(self, filename):
#    #    # Simply load data (X and y)
#    #    # transpose, augment, etc. will be done in minibatch-wise (NOT HERE)
#    #    X = imread(os.path.join(self.file_dir, filename))
#    #    X = np.float32(np.multiply(X, 1.0/255.0)) # <- resize will do
#    #    #X = np.float32(resize(X, (224, 224), mode='constant')) ## RGB always? # FIXME
#    #    if len(X.shape) == 2:
#    #        X = np.tile(X, (3,1,1))
#    #        X = np.transpose(X, (1,2,0))
#    #        #X = np.expand_dims(X, axis=2)
#    #    #X = np.transpose(X, [2,0,1])
#    #    
#    #    label = self.anno_dict['images'][filename]['class'][0] # we only provide a single label case
#    #    label = 1 if self.binary and label > 0 else label
#    #    num_cls = 2 if self.binary else len(self.anno_dict['classes'])
#    #    y = np.zeros(num_cls,dtype=np.float32)
#    #    y[label] = 1.
#
#    #    return X, y
#
#    #def load_dataset(self, filename_list):
#    #    # FIXME
#    #    t1 = time.time()
#    #    images = []; labels = []
#    #    for filename in filename_list:
#    #        X,y = self.load_data(filename)
#    #        images.append(X)
#    #        labels.append(y)
#    #    images = np.stack(images, axis=0)
#    #    labels = np.stack(labels, axis=0)
#
#    #    print('loading the overall raw dataset is done, and it takes {} sec'.format(time.time()-t1))
#
#    #    return images, labels



#def random_rotate(images):
#    """
#    Perform random rotation to images.
#    :param images: np.ndarray, shape: (N, C, H, W).
#    :return: np.ndarray, shape: (N, C, H, W).
#    """
#    augmented_images = []
#    for image in images:    # image.shape: (C, H, W)
#        # Randomly reflect image horizontally/vertically
#        rotate = np.random.randint(4)
#        if rotate == 0:
#            image = image[:, ::-1, :]
#        elif rotate == 1:
#            image = image[:, :, ::-1]
#        elif rotate == 2:
#            image = image[:, ::-1, ::-1]
#        # else, keep the original image
#        augmented_images.append(image)
#    return np.stack(augmented_images)    # shape: (N, C, H, W)
#
#
#def reflect(images):
#    """
#    Perform reflection from images, resulting in 4x augmented images.
#    :param images: np.ndarray, shape: (N, C, H, W).
#    :return: np.ndarray, shape: (N, 4, C, H, W).
#    """
#    augmented_images = []
#    for image in images:    # image.shape: (C, H, W)
#        aug_image = np.array([image, image[:, ::-1, :], image[:, :, ::-1], image[:, ::-1, ::-1]])
#        augmented_images.append(aug_image)
#    return np.stack(augmented_images)    # shape: (N, 4, C, H, W)
#
#
#def preprocessed_dataloader(dataloader, split_size=1, augment=False):
#    # Load the preprocessed data in lazy way by using dataloader
#    # Split data into split_size and do augmentations
#    # We do this in order to process data efficiently using numpy operations
#
#    assert(type(split_size)==int, 'split_size must be integer')
#
#    batch_size = 0
#
#    for X, y in dataloader:
#        # Preprocessing
#        X, y = X.numpy(), y.numpy()
#        X = np.transpose(X, [2,0,1])
#
#        if augment == True:
#            X = random_rotate(X)
#
#        queue_size = len(X)
#
#        # Initialize batch_size 
#        if batch_size == 0:
#            assert((queue_size % split_size)==0, 'queue_size cannot be divided into split_size')
#            batch_size = queue_size / split_size
#
#        cursor = 0
#        while True:
#            next_cursor = min(cursor + batch_size, queue_size)
#            if next_cursor == cursor:
#                break
#            else:
#                yield X[cursor:next_cursor], y[cursor:next_cursor]
#            cursor = next_cursor
