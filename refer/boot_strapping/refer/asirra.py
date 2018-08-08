import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize


def read_asirra_subset(subset_dir, one_hot=True, sample_size=None):
    """
    Load the Asirra Dogs vs. Cats data subset from disk
    and perform preprocessing for training AlexNet.
    :param subset_dir: str, path to the directory to read.
    :param one_hot: bool, whether to return one-hot encoded labels.
    :param sample_size: int, sample size specified when we are not using the entire set.
    :return: X_set: np.ndarray, shape: (N, C, H, W).
             y_set: np.ndarray, shape: (N, num_channels) or (N,).
    """
    # Read trainval data
    filename_list = os.listdir(subset_dir)
    set_size = len(filename_list)

    if sample_size is not None and sample_size < set_size:
        # Randomly sample subset of data when sample_size is specified
        filename_list = np.random.choice(filename_list, size=sample_size, replace=False)
        set_size = sample_size
    else:
        # Just shuffle the filename list
        np.random.shuffle(filename_list)

    # Pre-allocate data arrays
    X_set = np.empty((set_size, 3, 256, 256), dtype=np.float32)    # (N, 3, H, W)
    y_set = np.empty((set_size), dtype=np.uint8)                   # (N,)
    for i, filename in enumerate(filename_list):
        if i % 1000 == 0:
            print('Reading subset data: {}/{}...'.format(i, set_size), end='\r')
        label = filename.split('.')[0]
        if label == 'cat':
            y = 0
        else:  # label == 'dog'
            y = 1
        file_path = os.path.join(subset_dir, filename)
        img = imread(file_path)    # shape: (H, W, 3), range: [0, 255]
        img = resize(img, (256, 256), mode='constant').transpose(2, 0, 1).astype(np.float32)
        # (3, 256, 256), [0.0, 1.0]
        X_set[i] = img
        y_set[i] = y

    if one_hot:
        # Convert labels to one-hot vectors, shape: (N, num_classes)
        y_set_oh = np.zeros((set_size, 2), dtype=np.uint8)
        y_set_oh[np.arange(set_size), y_set] = 1
        y_set = y_set_oh
    print('\nDone')

    return X_set, y_set.astype(np.float32)


def random_crop_reflect(images, crop_l):
    """
    Perform random cropping and reflection from images.
    :param images: np.ndarray, shape: (N, C, H, W).
    :param crop_l: int, a side length of crop region.
    :return: np.ndarray, shape: (N, C, h, w).
    """
    H, W = images.shape[2:]
    augmented_images = []
    for image in images:    # image.shape: (C, H, W)
        # Randomly crop patch
        y = np.random.randint(H-crop_l)
        x = np.random.randint(W-crop_l)
        image = image[:, y:y+crop_l, x:x+crop_l]    # (C, h, w)

        # Randomly reflect patch horizontally
        reflect = bool(np.random.randint(2))
        if reflect:
            image = image[:, :, ::-1]

        augmented_images.append(image)
    return np.stack(augmented_images)    # shape: (N, C, h, w)


def corner_center_crop_reflect(images, crop_l):
    """
    Perform 4 corners and center cropping and reflection from images,
    resulting in 10x augmented patches.
    :param images: np.ndarray, shape: (N, C, H, W).
    :param crop_l: int, a side length of crop region.
    :return: np.ndarray, shape: (N, 10, C, h, w).
    """
    H, W = images.shape[2:]
    augmented_images = []
    for image in images:    # image.shape: (C, H, W)
        aug_image_orig = []
        # Crop image in 4 corners
        aug_image_orig.append(image[:, :crop_l, :crop_l])
        aug_image_orig.append(image[:, :crop_l, -crop_l:])
        aug_image_orig.append(image[:, -crop_l:, :crop_l])
        aug_image_orig.append(image[:, -crop_l:, -crop_l:])
        # Crop image in the center
        aug_image_orig.append(image[:, H//2-(crop_l//2):H//2+(crop_l-crop_l//2),
                                    W//2-(crop_l//2):W//2+(crop_l-crop_l//2)])
        aug_image_orig = np.stack(aug_image_orig)    # (5, C, h, w)

        # Flip augmented images and add it
        aug_image_flipped = aug_image_orig[:, :, :, ::-1]    # (5, C, h, w)
        aug_image = np.concatenate((aug_image_orig, aug_image_flipped), axis=0)    # (10, C, h, w)
        augmented_images.append(aug_image)
    return np.stack(augmented_images)    # shape: (N, 10, C, h, w)


def center_crop(images, crop_l):
    """
    Perform center cropping of images.
    :param images: np.ndarray, shape: (N, H, W, C).
    :param crop_l: int, a side length of crop region.
    :return: np.ndarray, shape: (N, h, w, C).
    """
    H, W = images.shape[2:]
    cropped_images = []
    for image in images:    # image.shape: (H, W, C)
        # Crop image in the center
        cropped_images.append(image[H//2-(crop_l//2):H//2+(crop_l-crop_l//2),
                              W//2-(crop_l//2):W//2+(crop_l-crop_l//2)])
    return np.stack(cropped_images)


class DataSet(object):
    def __init__(self, images, labels=None, patch_side_len=227):
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
        self._patch_side_len = patch_side_len
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
        :return: batch_images: np.ndarray, shape: (N, C, H, W).
                 batch_labels: np.ndarray, shape: (N, num_classes) or (N,).
        """
        # if fake_data:
        #     fake_batch_images = np.random.random(size=(batch_size, 3, self._patch_side_len, self._patch_side_len))
        #     fake_batch_labels = np.zeros((batch_size, 2), dtype=np.float32)
        #     fake_batch_labels[np.arange(batch_size), np.random.randint(2, size=batch_size)] = 1
        #     return fake_batch_images, fake_batch_labels

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
            fake_batch_images = np.ones((batch_size, 3, self._patch_side_len, self._patch_side_len),
                                        dtype=np.float32) * fake_batch_labels

            return fake_batch_images, batch_labels

        if augment and is_train:
            # Perform data augmentation, for training phase
            batch_images = random_crop_reflect(batch_images, self._patch_side_len)
        elif augment and not is_train:
            # Perform data augmentation, for evaluation phase(10x)
            batch_images = corner_center_crop_reflect(batch_images, self._patch_side_len)
        else:
            # Don't perform data augmentation, generating center-cropped patches
            batch_images = center_crop(batch_images, self._patch_side_len)

        return batch_images, batch_labels
