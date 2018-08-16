import time
import cntk as C
import numpy as np
import matplotlib as mpl
from skimage.transform import resize
from skimage.color import rgba2rgb

from torch.utils.data import DataLoader

def grad_cam(model, data_set, class_idx, target_conv_layer_name, **kwargs):
    """
    Generate Grad-CAM visualization (for model-predicted class).
    :param model: ConvNet model.
    :param dataset: dataset class which will be imported in the main code
    :param root_dir
    :param filename_list: list of filename
    :param anno_dict: annotation dictionary
    :param class_idx: int, index of the target class.
    :param target_conv_layer_name: str, name of the target layer from which gradient is calculated.
    :param kwargs: dict, extra arguments for Grad-CAM visualization.
    :return: np.ndarray.
    """
    batch_size = kwargs.pop('batch_size', 256)
    augment_pred = kwargs.pop('augment_pred', False)
    cmap_name = kwargs.pop('cmap_name', 'jet')
    num_workers = kwargs.pop('num_workers', 4)

    dataloader = DataLoader(data_set, batch_size, shuffle=False, num_workers=num_workers)

    target_conv_layer = model.d[target_conv_layer_name]    # ((N, )D, h, w)
    logits = model.d['logits']    # ((N, )num_classes)
    H, W = 224, 224 # FIXME
    D, h, w = target_conv_layer.shape

    # Clone the existing network, starting from the target conv layer
    target_conv_layer_input = C.input_variable(target_conv_layer.shape, needs_gradient=True)
    logits_target = logits.clone('share', substitutions={target_conv_layer: target_conv_layer_input})
    logits_class = logits_target[class_idx]    # ((N, ))

    Xs = []
    f_grad_list = []
    f_activation_list = []
    for X, _ in dataloader:
        X = X.numpy()
        Xs.append(X)

        # Compute feature map gradients for each image
        f_activation = target_conv_layer.eval({model.X: X})
        f_activation_list.append(f_activation)
        f_grad = logits_class.grad({target_conv_layer_input: f_activation})    # (n, D, h, w)
        f_grad_list.append(f_grad)

    _f_activation = np.concatenate(f_activation_list, axis=0)
    _f_grad = np.concatenate(f_grad_list, axis=0)

    # _f_activation_shape, _f_grad.shape: (N, D, h, w)
    
    # Compute weights by global average pooling
    f_weights = _f_grad.mean(axis=(-1, -2), keepdims=True)    # (N, D, 1, 1) or (N, k, D, 1, 1)

    # f_weights = global_avg_pool(f_grad)    # (N, 1, 1, d) or (N, 1, 1, 1)
    # Compute weighted combination of feature maps,
    f_weighted_sum = np.sum(f_weights * _f_activation, keepdims=True, axis=-3)
    # and apply ReLU nonlinearity
    grad_cams = np.maximum(f_weighted_sum, 0)    # (N, k, 1, h, w) or (N, 1, h, w)

    # Resize((h,w)->(H,W), normalize and apply colormap to CAM (float number -> rgb color)
    def clarify_cam(cam):
        cmap = mpl.cm.get_cmap(cmap_name)
        # cam.shape: (1, h, w)
        if cam.max() - cam.min() != 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())     # normalize: [0.0, 1.0]
        else:
            cam = np.zeros_like(cam)
        cam = resize(cam.squeeze(), (H, W), mode='constant', preserve_range=True)  # resize: (H, W)
        return rgba2rgb(cmap(cam)).transpose(2, 0, 1)    # apply colormap: (3, H, W)

    cam_output_list = []
    for cam in grad_cams:
        cam_output_list.append(clarify_cam(cam))
    cam_output = np.stack(cam_output_list)    # (N, 3, H, W)

    return cam_output, np.concatenate(Xs, axis=0)


def guided_backprop(model, dataset, target_conv_layer_name, **kwargs):
    """
    Generate Guided backpropagation visualization (for selected feature maps).
    :param model: ConvNet model.
    :param dataset: DataSet.
    :param target_conv_layer_name: str, name of the target layer from which gradient is calculated.
    :param kwargs: dict, extra arguments for Guided-Backprop visualization.
    :return: np.ndarray.
    """
    batch_size = kwargs.pop('batch_size', 256)
    augment_pred = kwargs.pop('augment_pred', False)

    target_conv_layer = model.d[target_conv_layer_name]    # ((N, )D, h, w)
    ch, H, W = dataset.images.shape[1:]
    pred_size = dataset.num_examples
    num_steps = pred_size // batch_size

    # Clone the existing network, starting from the target conv layer
    X_input = C.input_variable(model.X.shape, dtype=np.float32, needs_gradient=True)
    target_conv_layer_clone = target_conv_layer.clone('share', substitutions={model.X: X_input})

    gb_grad_list = []
    for i in range(num_steps+1):
        if i == num_steps:
            _batch_size = pred_size - num_steps*batch_size
        else:
            _batch_size = batch_size
        X, _ = dataset.next_batch(_batch_size, shuffle=False,
                                  augment=augment_pred, is_train=False)
        # if augment_pred == True:  X.shape: (N, k, C, h, w)
        # else:                     X.shape: (N, C, h, w)

        # If performing augmentation during prediction:
        if augment_pred:
            k = X.shape[1]
            gb_grad_modes = np.empty((_batch_size, k, ch, H, W), dtype=np.float32)    # (n, k, C, H, W)
            # compute gb_grad for each of k modes,
            for idx in range(k):
                gb_grad_mode = target_conv_layer_clone.grad({X_input: X[:, idx]})
                gb_grad_modes[:, idx] = gb_grad_mode
            gb_grad_list.append(gb_grad_modes)
        else:
            # Compute gb_grad for each image
            gb_grad = target_conv_layer_clone.grad({X_input: X})    # (n, C, H, W)
            gb_grad_list.append(gb_grad)

    gb_grads = np.concatenate(gb_grad_list, axis=0)

    # Normalize GB
    gb_output_list = []
    for gb in gb_grads:    # gb: (C, H, W)
        if gb.max() - gb.min() != 0:
            gb = (gb - gb.min()) / (gb.max() - gb.min())     # normalize: [0.0, 1.0]
        else:
            gb = np.zeros_like(gb)
        gb_output_list.append(gb)

    gb_output = np.stack(gb_output_list)    # (N, C, H, W)

    return gb_output


def get_feature_maps(model, data_set, target_conv_layer_name, **kwargs):
    """
    Get feature maps on selected convolutional layer.
    :param model: ConvNet model.
    :param dataset: dataset class which will be imported in the main code
    :param root_dir
    :param filename_list: list of filename
    :param anno_dict: annotation dictionary
    :param target_conv_layer_name: str, name of the target layer from which feature map is retrieved.
    :param kwargs: dict, extra arguments for feature map visualization.
    :return: np.ndarray.
    """
    batch_size = kwargs.pop('batch_size', 256)
    augment_pred = kwargs.pop('augment_pred', False)
    num_workers = kwargs.pop('num_workers', 4)

    dataloader = DataLoader(data_set, batch_size, shuffle=False, num_workers=num_workers)

    target_conv_layer = model.d[target_conv_layer_name]    # ((N, )D, h, w)
    D, h, w = target_conv_layer.shape

    f_map_list = []
    X_list = []; y_list = []

    for X, y in dataloader:
        X = X.numpy(); y = y.numpy()
        
        # Compute feature maps for each image
        f_map = target_conv_layer.eval({model.X: X}) # (n, D, h, w)
        f_map_list.append(f_map)

        # Gathering image data for visualization
        X_temp = np.transpose(X, [0, 2, 3, 1]) #(N, C, H, W) -> (N, H, W, C)
        resize28 = lambda x: resize(x, (28,28), mode='constant') # resize to 28x28 image due to memory issue
        X_temp = list(map(resize28, X_temp))
        X_list.append(X_temp)
        y_list.append(y)

    f_maps = np.concatenate(f_map_list, axis=0)
    Xs = np.transpose(np.concatenate(X_list, axis=0), [0, 3, 1, 2]) # (N,H,W,C) -> (N,C,H,W)
    ys = np.concatenate(y_list, axis=0)

    # Normalize feature maps
    f_map_output_list = []
    for f_map in f_maps:    # f_map.shape: (D, h, w)
        if f_map.max() - f_map.min() != 0:
            f_map = (f_map - f_map.min()) / (f_map.max() - f_map.min())    # normalize: [0.0, 1.0]
        else:
            f_map = np.zeros_like(f_map)
        f_map_output_list.append(f_map)
    f_map_output = np.stack(f_map_output_list)    # (N, D, h, w)

    return f_map_output, Xs, ys
