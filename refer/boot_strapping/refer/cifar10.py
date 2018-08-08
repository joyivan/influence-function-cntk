import os
import pickle
import numpy as np


def read_cifar10_subset(root_dir, is_train=True, one_hot=False):
    """
    Load CIFAR-10 data subset from disk.
    :param root_dir: str, path to the directory to read.
    :param is_train: bool, whether to load training(/validation) set.
    :param one_hot: bool, whether to return one-hot encoded labels.
    :return: X_set: np.ndarray, shape: (N, C, H, W).
             y_set: np.ndarray, shape: (N, num_channels) or (N,).
    """

    def unpickle(file):
        with open(file, 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
        return d

    if is_train:
        # Read training data set batches
        print('Reading training data...')
        set_batches = []
        for i in range(1, 6):
            set_batch_file_path = 'data_batch_{}'.format(i)
            set_batches.append(unpickle(os.path.join(root_dir, set_batch_file_path)))

        X_set = np.vstack([set_batch[b'data'] for set_batch in set_batches])    # (N, C*H*W)
        set_size = X_set.shape[0]
        set_labels = np.hstack([set_batch[b'labels'] for set_batch in set_batches])   # (N,)
    else:
        # Read test data set batch
        print('Reading test data...')
        test_set_batch = unpickle(os.path.join(root_dir, 'test_batch'))
        X_set = test_set_batch[b'data']    # (N, C*H*W)
        set_size = X_set.shape[0]
        set_labels = test_set_batch[b'labels']

    X_set = np.multiply(X_set.reshape(set_size, 3, 32, 32), 1.0/255.0).astype(np.float32)    # (N, C, H, W)
    y_set = set_labels

    if one_hot:
        # Convert labels to one-hot vectors, shape: (N, num_classes)
        y_set_oh = np.zeros((set_size, 10), dtype=np.int)
        y_set_oh[np.arange(set_size), y_set] = 1
        y_set = y_set_oh
    print('Done')

    return X_set, y_set.astype(np.float32)


def random_reflect(images):
    """
    Perform random reflection from images.
    :param images: np.ndarray, shape: (N, C, H, W).
    :return: np.ndarray, shape: (N, C, H, W).
    """
    augmented_images = []
    for image in images:    # image.shape: (C, H, W)
        # Randomly reflect image horizontally
        reflect = bool(np.random.randint(2))
        if reflect:
            image = image[:, :, ::-1]

        augmented_images.append(image)
    return np.stack(augmented_images)    # shape: (N, C, H, W)


def reflect(images):
    """
    Perform reflection from images, resulting in 2x augmented images.
    :param images: np.ndarray, shape: (N, C, H, W).
    :return: np.ndarray, shape: (N, 2, C, H, W).
    """
    augmented_images = []
    for image in images:    # image.shape: (C, H, W)
        aug_image = np.stack([image, image[:, :, ::-1]])    # (2, C, H, W)
        augmented_images.append(aug_image)
    return np.stack(augmented_images)    # shape: (N, 2, C, H, W)


class DataSet(object):
    def __init__(self, images, labels=None):
        """
        Construct a new DataSet object.
        :param images: np.ndarray, shape: (N, C, H, W).
        :param labels: np.ndarray, shape: (N, num_classes) or (N,).
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

    def next_batch(self, batch_size, shuffle=True, augment=True, is_train=True,
                   fake_data=False):
        """
        Return the next `batch_size` examples from this dataset.
        :param batch_size: int, size of a single batch.
        :param shuffle: bool, whether to shuffle the whole set while sampling a batch.
        :param augment: bool, whether to perform data augmentation while sampling a batch.
        :param is_train: bool, current phase for sampling.
        :param fake_data: bool, whether to generate fake data (for debugging).
        :return: batch_images: np.ndarray, shape: (N, C, h, w) or (N, 10, C, h, w).
                 batch_labels: np.ndarray, shape: (N, num_classes) or (N,).
        """
        if fake_data:
            fake_batch_images = np.random.random(size=(batch_size, 3, 32, 32))
            fake_batch_labels = np.zeros((batch_size, 10), dtype=np.int)
            fake_batch_labels[np.arange(batch_size), np.random.randint(10, size=batch_size)] = 1
            return fake_batch_images, fake_batch_labels

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

        if augment and is_train:
            # Perform data augmentation, for training phase
            batch_images = random_reflect(batch_images)
        elif augment and not is_train:
            # Perform data augmentation, for evaluation phase(10x)
            batch_images = reflect(batch_images)

        return batch_images, batch_labels
