import os
import json
import numpy as np
from skimage.io import imread
from skimage.transform import resize

def read_skc_subset(root_dir, binary=True, mode='train', sample_size=None):
    """
    Load the SKC data subset from disk in "lazy way".
    :param root_dir:  str, path to the root directory.
    :param binary: bool, whether to convert to binary classes.
    :param mode: str, type of subset to load: {'train', 'test', 'total'}
    :param sample_size: int, sample size specified when we are not using the entire set.
    :return: X_set: np.ndarray, shape: (N, C, H, W).
             y_set: np.ndarray, shape: (N, num_channels).
    """
    # Read filename list of subset
    if mode == 'total':
        set_filename_list = []
        for set_txt_filename in ['train1.txt', 'validation1.txt', 'test.txt']:
            set_filename_path = os.path.join(root_dir, 'imageset', 'imageset1', set_txt_filename)
            with open(set_filename_path, 'r') as fid:
                set_filename_list += fid.read().split('\n')[:-1]
        set_size = len(set_filename_list)
    else:    # mode == 'train' or mode == 'validation' or mode == 'test'
        set_txt_filename = '{}.txt'.format(mode)
        set_filename_path = os.path.join(root_dir, 'imageset', 'imageset1', set_txt_filename)
        with open(set_filename_path, 'r') as fid:
            set_filename_list = fid.read().split('\n')[:-1]
        set_size = len(set_filename_list)

    if sample_size is not None and sample_size < set_size:
        # Randomly sample subset of data when sample_size is specified
        set_filename_list = np.random.choice(set_filename_list, size=sample_size, replace=False)
        set_size = sample_size
    
    else:
        # Just shuffle the original filename list
        np.random.shuffle(set_filename_list)

    # Read annotations
    anno_path = os.path.join(root_dir, 'annotation', 'annotation1.json')
    with open(anno_path, 'r') as fid:
        anno_dict = json.load(fid)
    num_classes = len(anno_dict['classes']) if not binary else 2

    ### change this to lazy loader -> this output should be result_queue?
    ## Pre-allocate data arrays
    #X_set = np.zeros((set_size, 3, 224, 224), dtype=np.float32)    # (N, C, H, W)
    #y_set = np.zeros((set_size, num_classes), dtype=np.float32)    # (N, num_classes)

    ## Read actual images
    #image_dir = os.path.join(root_dir, 'image')
    #for idx, filename in enumerate(set_filename_list):
    #    if idx % 100 == 0:
    #        print('Reading subset data...{}/{}'.format(idx, set_size), end='\r', flush=True)
    #    file_path = os.path.join(image_dir, filename)
    #    img_np = np.multiply(imread(file_path), 1.0/255.0)    # clip into bound [0.0, 1.0]
    #    #print(idx,filename,img_np.shape)
    #    X_set[idx] = resize(img_np, (224, 224), mode='constant').transpose(2, 0, 1).astype(np.float32)
    #    # from shape: (3, 128, 128), range: [0, 255]
    #    # to shape: (3, 224, 224), range: [0.0, 1.0]
    #    labels = anno_dict['images'][filename]['class']
    #    for label in labels:
    #        label = 1 if binary and label > 0 else label
    #        y_set[idx, label] = 1.
    #print('\nDone')

    return X_set, y_set


def random_reflect(images):
    """
    Perform random reflection from images.
    :param images: np.ndarray, shape: (N, C, H, W).
    :return: np.ndarray, shape: (N, C, H, W).
    """
    augmented_images = []
    for image in images:    # image.shape: (C, H, W)
        # Randomly reflect image horizontally/vertically
        rotate = np.random.randint(4)
        if rotate == 0:
            image = image[:, ::-1, :]
        elif rotate == 1:
            image = image[:, :, ::-1]
        elif rotate == 2:
            image = image[:, ::-1, ::-1]
        # else, keep the original image
        augmented_images.append(image)
    return np.stack(augmented_images)    # shape: (N, C, H, W)


def reflect(images):
    """
    Perform reflection from images, resulting in 4x augmented images.
    :param images: np.ndarray, shape: (N, C, H, W).
    :return: np.ndarray, shape: (N, 4, C, H, W).
    """
    augmented_images = []
    for image in images:    # image.shape: (C, H, W)
        aug_image = np.array([image, image[:, ::-1, :], image[:, :, ::-1], image[:, ::-1, ::-1]])
        augmented_images.append(aug_image)
    return np.stack(augmented_images)    # shape: (N, 4, C, H, W)


class DataSet(object):
    def __init__(self, images, labels=None):
        """
        Construct a new DataSet object.
        :param images: np.ndarray, shape: (N, C, H, W).
        :param labels: np.ndarray, shape: (N, num_classes).
        """
        if labels is not None:
            assert images.shape[0] == labels.shape[0], (
                'Number of examples mismatch, between images and labels.'
            )
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels    # NOTE: this can be None, if not given.
        self._indices = np.arange(self._num_examples, dtype=np.uint)    # image/label indices(can be permuted)
        self._reset()

    def _reset(self):
        """Reset some variables."""
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size, shuffle=True, augment=True, is_train=True, fake_data=False):
        """
        Return the next `batch_size` examples from this dataset.
        :param batch_size: int, size of a single batch.
        :param shuffle: bool, whether to shuffle the whole set while sampling a batch.
        :param augment: bool, whether to perform data augmentation while sampling a batch.
        :param is_train: bool, current phase for sampling.
        :param fake_data: bool, whether to generate fake data (for debugging).
        :return: batch_images: np.ndarray, shape: (N, C, H, W).
                 batch_labels: np.ndarray, shape: (N, num_classes).
        """
        start_index = self._index_in_epoch

        # Shuffle the dataset, for the first epoch
        if self._epochs_completed == 0 and start_index == 0 and shuffle:
            np.random.shuffle(self._indices)

        # Go to the next epoch, if current index goes beyond the total number of examples
        if start_index + batch_size > self._num_examples:
            # Increment the number of epochs completed
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start_index
            indices_rest_part = self._indices[start_index:self._num_examples]

            # Shuffle the dataset, after finishing a single epoch
            if shuffle:
                np.random.shuffle(self._indices)

            # Start the next epoch
            start_index = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end_index = self._index_in_epoch
            indices_new_part = self._indices[start_index:end_index]

            images_rest_part = self.images[indices_rest_part]
            images_new_part = self.images[indices_new_part]
            batch_images = np.concatenate((images_rest_part, images_new_part), axis=0)
            if self.labels is not None:
                labels_rest_part = self.labels[indices_rest_part]
                labels_new_part = self.labels[indices_new_part]
                batch_labels = np.concatenate((labels_rest_part, labels_new_part), axis=0)
            else:
                batch_labels = None
        else:
            self._index_in_epoch += batch_size
            end_index = self._index_in_epoch
            indices = self._indices[start_index:end_index]
            batch_images = self.images[indices]
            if self.labels is not None:
                batch_labels = self.labels[indices]
            else:
                batch_labels = None

        if fake_data:
            fake_batch_labels = batch_labels.argmax(axis=1).reshape((-1, 1, 1, 1))
            fake_batch_images = np.ones((batch_size, 3, 224, 224), dtype=np.float32) * fake_batch_labels

            return fake_batch_images, batch_labels

        if augment and is_train:
            # Perform data augmentation, for training phase
            batch_images = random_reflect(batch_images)
        elif augment and not is_train:
            # Perform data augmentation, for evaluation phase(10x)
            batch_images = reflect(batch_images)

        return batch_images, batch_labels
