import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize, rescale
from skimage.filters import threshold_otsu as thresholding_fn
from skimage.color import rgb2gray

NUM_AUGMENTATIONS = 3


def read_usim(root_dir, one_hot=True, patch_l=448, scale_factor=1.0,
              sample_size=None, **kwargs):

    def extract_and_crop_image_sets(sub_dir, sample_size=None, **kwargs):
        """
        Extract informative regions from each image and mask.
        :param sub_dir: Path of trainval/test set subdirectory.
        :param sample_size: int, sample size specified when we are not using the entire set.
        :return: X_set: np.ndarray, shape: (N, C, H, W).
                 y_set: np.ndarray, shape: (N, num_channels).
        """
        img_dir = os.path.join(sub_dir, 'images')
        mask_dir = os.path.join(sub_dir, 'segmasks')

        # Set cropping hyperparameters
        window_h = kwargs.pop('window_h', 121)
        window_w = kwargs.pop('window_w', 41)

        filename_list = os.listdir(img_dir)
        filename_wo_ext_list = ['.'.join(filename.split('.')[:-1]) for filename in filename_list]
        set_size = len(filename_wo_ext_list)

        if sample_size is not None and sample_size < set_size:
            # Randomly sample subset of data when sample_size is specified
            filename_wo_ext_list = np.random.choice(filename_wo_ext_list, size=sample_size, replace=False)
            set_size = sample_size
        else:
            # Just shuffle the filename list
            np.random.shuffle(filename_wo_ext_list)

        # 1. Read, rescale and crop images by thresholding
        img_np_list = []    # [0, 255]
        mask_inv_np_list = []    # [0, 255]
        for idx, filename_wo_ext in enumerate(filename_wo_ext_list):
            if idx % 20 == 0:
                print('Reading data...{}/{}'.format(idx, set_size), end='\r', flush=True)
            img_file_path = os.path.join(img_dir, filename_wo_ext + '.bmp')
            mask_file_path = os.path.join(mask_dir, filename_wo_ext + '_label.png')

            img_np = imread(img_file_path)    # (H, W), [0, 255]
            mask_np = imread(mask_file_path)  # (H, W), [0, 255]

            # Convert 'black-annotated' mask to 'white-annotated mask'
            mask_inv_np = np.zeros_like(mask_np)
            mask_inv_np[mask_np < 255] = 255

            # 1. Get binary mask by 'double-mean' thresholding
            img_mean = img_np.mean()
            img_lower_mean = img_np[img_np < img_mean].mean()
            bin_np = np.zeros_like(img_np)
            bin_np[img_np > img_mean] = 1.0
            bin_np[img_np < img_lower_mean] = 1.0

            # 2. Get horizontal upper/lower bounds,
            # according to its horizontal windowed sum of binary mask
            pad_h = window_h // 2
            bin_np_padded = np.pad(bin_np,
                                   ((pad_h, pad_h), (0, 0)), 'edge')
            bin_h_np = np.array([bin_np_padded[y-pad_h:y+pad_h+1, :].sum()
                                 for y in range(0+pad_h, bin_np_padded.shape[0]-pad_h)])    # (H,)
            bin_h_mean = bin_h_np.mean()
            h_outlier = bin_h_np > bin_h_mean
            h_ub = np.argwhere(~h_outlier).min()
            h_lb = np.argwhere(~h_outlier).max()

            # 3. Get vertical left/right bounds,
            # according to its vertical windowed sum of binary mask
            h_cropped_bin_np = bin_np[h_ub:h_lb]
            pad_w = window_w // 2
            h_cropped_bin_np_padded = np.pad(h_cropped_bin_np,
                                             ((0, 0), (pad_w, pad_w)), 'edge')
            h_cropped_bin_v_np = np.array([h_cropped_bin_np_padded[:, x-pad_w:x+pad_w+1].sum()
                                           for x in range(0+pad_w,
                                                          h_cropped_bin_np_padded.shape[1]-pad_w)])    # (W,
            h_cropped_bin_v_mean = h_cropped_bin_v_np.mean()
            v_outlier = h_cropped_bin_v_np > h_cropped_bin_v_mean
            v_lb = np.argwhere(~v_outlier).min()
            v_rb = np.argwhere(~v_outlier).max()

            # Crop images using its horizontal/vertical bounds
            cropped_img_np = img_np[h_ub:h_lb, v_lb:v_rb]
            img_np_list.append(cropped_img_np)

            # Crop seg mask using its horizontal/vertical bounds
            cropped_inv_mask_np = mask_inv_np[h_ub:h_lb, v_lb:v_rb]
            mask_inv_np_list.append(cropped_inv_mask_np)

        return img_np_list, mask_inv_np_list

    def get_label_from_patch_mask(patch_mask_np, fp_portion=0.10):
        patch_l = patch_mask_np.shape[0]   # patch_mask_np: (H, W, C), [0.0, 1.0]
        # Focused patch mask region
        padding = int(patch_l * fp_portion)
        focused_patch_mask_np = patch_mask_np[padding:-padding, padding:-padding]

        # Filtering rule 1: Label this patch as 'positive' and keep the patch
        #                   if at least a single pixel in the focused region is marked as 1.0.
        if focused_patch_mask_np.sum() > 0:
            return 1
        # Filtering rule 2: Drop this 'ambiguous' patch
        #                   if at least a single pixel in the padding is marked as 1.0.
        elif patch_mask_np.sum() > 0:
            return -1
        # Label this patch as 'negative' and keep the patch otherwise.
        else:
            return 0

    def extract_patches_and_labels(img_np_list, mask_inv_np_list, cropped=True,
                                   patch_l=428, one_hot=True, **kwargs):
        fp_portion = kwargs.pop('fp_portion', 0.10)
        scale_factor = kwargs.pop('scale_factor', 1.0)

        # Find min_h from cropped images
        if cropped:
            min_h = 1e4
            for img_np in img_np_list:
                h, w = img_np.shape
                if h < min_h: min_h = h

            assert patch_l <= min_h, 'patch side length is longer than minimum height of cropped images'

        patch_l = int(patch_l * scale_factor)
        stride = patch_l // 2    # 50% overlapping

        # Resize images and seg masks whose height to be patch_l,
        # and extract patches and labels from them
        # NOTE: image height ranges: (428, 512)
        X_patches_list, M_patches_list, y_patches_list = [], [], []
        for img_np, mask_np in zip(img_np_list, mask_inv_np_list):    # [0, 255], [0, 255]
            # Resize images to be fit in informative regions
            if cropped:
                h, w = img_np.shape
                resize_w = int(w * (min_h/h) * scale_factor)
                resize_h = int(min_h * scale_factor)
                img_np = resize(img_np, (resize_h, resize_w), mode='edge')    # [0.0, 1.0]
                mask_np = resize(mask_np, (resize_h, resize_w), mode='edge')    # [0.0, 1.0]
            # Just rescale images with the scaling factor
            else:
                img_np = rescale(img_np, scale_factor, mode='edge')    # [0.0, 1.0]
                mask_np = rescale(mask_np, scale_factor, mode='edge')    # [0.0, 1.0]

            # Resized h, w, patch_l and stride
            h, w = img_np.shape
            for y in range(0, h-stride, stride):
                for x in range(0, w-stride, stride):
                    if w-x < patch_l: x = w-patch_l   # NOTE: the rightmost 'narrow' patch
                    if h-y < patch_l: y = h-patch_l   # NOTE: the bottommost 'short' patch
                    y_true = get_label_from_patch_mask(mask_np[y:y+patch_l, x:x+patch_l], fp_portion=fp_portion)
                    if y_true == -1: continue    # Drop this patch
                    else:
                        X_patches_list.append(img_np[y:y+patch_l, x:x+patch_l])
                        M_patches_list.append(mask_np[y:y+patch_l, x:x+patch_l])
                        y_patches_list.append(y_true)

        X_set = np.expand_dims(np.stack(X_patches_list), axis=1).astype(np.float32)    # (N, C, H, W)
        M_set = np.expand_dims(np.stack(M_patches_list), axis=1).astype(np.int)    # (N, C, H, W)
        y_set = np.stack(y_patches_list).astype(np.int)             # (N,)
        n_patches = X_set.shape[0]
        if one_hot:
            y_set_onehot = np.zeros((n_patches, 2), dtype=np.int)
            y_set_onehot[np.arange(n_patches), y_set] = 1
            y_set = y_set_onehot

        return X_set, M_set, y_set

    print('-- Reading dataset...')
    img_np_list, mask_np_list = extract_and_crop_image_sets(root_dir,
                                                            sample_size=sample_size, **kwargs)
    # [0, 255], [0, 255]
    print('\n-- Extracting patches...')
    X_set, M_set, y_set = \
        extract_patches_and_labels(img_np_list, mask_np_list, cropped=True,
                                   scale_factor=scale_factor,
                                   patch_l=patch_l, one_hot=one_hot, **kwargs)
    set_size = X_set.shape[0]

    # Shuffle trainval data
    shuffled_idxs = np.arange(set_size)
    np.random.shuffle(shuffled_idxs)    # *SHUFFLE*
    X_set, M_set, y_set = X_set[shuffled_idxs], M_set[shuffled_idxs], y_set[shuffled_idxs]
    print('Done')

    return X_set, M_set, y_set


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
    def __init__(self, images, labels=None, num_classes=2):
        """
        Construct a new Usim DataSet.
        @params images: np.ndarray, shape: (N, C, H, W).
        @params labels: np.ndarray, shape: (N, num_classes) or (N,).
        """
        if labels is not None:
            assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape)
            )
        self._num_classes = num_classes
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels    # NOTE: this can be None, if not given.
        self._indices = np.arange(self._num_examples, dtype=np.uint)    # image/label indices(can be permuted)
        self._class_indices_list = [self._indices[labels[:, c] == 1] for c in range(self._num_classes)]
        self._num_class_examples_list = [class_indices.shape[0] for class_indices
                                         in self._class_indices_list]
        self.reset()

    def reset(self):
        """Reset some variables."""
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._class_epochs_completed_list = [0] * self._num_classes
        self._class_index_in_epoch_list = [0] * self._num_classes

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def num_classes(self):
        return self._num_classes

    def next_batch(self, batch_size, shuffle=True, augment=True, is_train=True):
        """
        Return the next `batch_size` examples from this dataset.
        When training, perform stratified sampling of positive/negative examples,
        while testing, perform uniform sampling of positive/negative examples.
        :param batch_size: int, size of a single batch.
        :param shuffle: bool, whether to shuffle the whole set while sampling a batch.
        :param augment: bool, whether to perform data augmentation while sampling a batch.
        :param is_train: bool, current phase for sampling.
        :return: batch_images: np.ndarray, shape: (N, C, H, W) or (N, 4, C, H, W).
                 batch_labels: np.ndarray, shape: (N, num_classes) or (N,).
        """

        if not is_train:
            # Perform uniform sampling
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

            if augment:
                # Perform data augmentation
                batch_images = reflect(batch_images)

        else:
            # Perform stratified sampling
            assert batch_size % self.num_classes == 0

            # Shuffle the dataset, for the first epoch
            if self._class_epochs_completed_list == [0]*self.num_classes \
                    and self._class_index_in_epoch_list == [0]*self.num_classes and shuffle:
                for class_indices in self._class_indices_list:
                    np.random.shuffle(class_indices)
            class_batch_size = batch_size // self.num_classes

            batch_images, batch_labels = [], []
            # Extract the same size of class-batches for each class
            for c in range(self.num_classes):
                class_start_index = self._class_index_in_epoch_list[c]
                num_class_examples = self._num_class_examples_list[c]

                # Go to the next epoch, if current index goes beyond the total number of examples for current class
                if class_start_index + class_batch_size > num_class_examples:
                    # Increment the number of epochs completed for current class
                    self._class_epochs_completed_list[c] += 1
                    # Get the rest examples in this epoch
                    rest_num_class_examples = num_class_examples - class_start_index
                    class_indices_rest_part = self._class_indices_list[c][class_start_index:num_class_examples]

                    # Shuffle the dataset, after finishing a single epoch
                    if shuffle:
                        np.random.shuffle(self._class_indices_list[c])

                    # Start the next epoch
                    class_start_index = 0
                    self._class_index_in_epoch_list[c] = class_batch_size - rest_num_class_examples
                    class_end_index = self._class_index_in_epoch_list[c]
                    class_indices_new_part = self._class_indices_list[c][class_start_index:class_end_index]

                    class_images_rest_part = self.images[class_indices_rest_part]
                    class_images_new_part = self.images[class_indices_new_part]
                    class_batch_images = np.concatenate((class_images_rest_part, class_images_new_part), axis=0)
                    if self.labels is not None:
                        class_labels_rest_part = self.labels[class_indices_rest_part]
                        class_labels_new_part = self.labels[class_indices_new_part]
                        class_batch_labels = np.concatenate((class_labels_rest_part, class_labels_new_part), axis=0)
                    else:
                        class_batch_labels = None
                else:
                    self._class_index_in_epoch_list[c] += class_batch_size
                    class_end_index = self._class_index_in_epoch_list[c]
                    class_indices = self._class_indices_list[c][class_start_index:class_end_index]
                    class_batch_images = self.images[class_indices]
                    if self.labels is not None:
                        class_batch_labels = self.labels[class_indices]
                    else:
                        class_batch_labels = None

                if augment:
                    # Perform data augmentation for current class examples
                    class_batch_images = random_reflect(class_batch_images)

                batch_images.append(class_batch_images)
                batch_labels.append(class_batch_labels)

            batch_images = np.concatenate(batch_images, axis=0)
            batch_labels = np.concatenate(batch_labels, axis=0)

        return batch_images, batch_labels
