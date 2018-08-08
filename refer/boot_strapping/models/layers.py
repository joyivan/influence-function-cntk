import numpy as np
import cntk as C
from cntk.ops.functions import UserFunction
from cntk import output_variable, user_function


def max_pool(x, side_l, stride, padding='SAME'):
    """
    Performs max pooling on given input.
    :param x: shape: ((N, )C, H, W).
    :param side_l: int, the side length of the pooling window for each dimension.
    :param stride: int, the stride of the sliding window for each dimension.
    :param padding: str, either 'SAME' or 'VALID',
                         the type of padding algorithm to use.
    :return: cntk.ops.functions.Function.
    """
    pad = True if padding == 'SAME' else False    # padding == 'VALID'

    pool = C.layers.MaxPooling((side_l, side_l), strides=stride, pad=pad)(x)
    return pool


def conv_layer(x, side_l, stride, out_depth, padding='SAME', bias=True, **kwargs):
    """
    Add a new convolutional layer.
    :param x: shape: ((N, )C, H, W).
    :param side_l: int, the side length of the filters for each dimension.
    :param stride: int, the stride of the filters for each dimension.
    :param out_depth: int, the total number of filters to be applied.
    :param padding: str, either 'SAME' or 'VALID',
                         the type of padding algorithm to use.
    :param bias: bool, whether to add biases.
    :param kwargs: dict, extra arguments, including weights/biases initialization hyperparameters.
        - weight_stddev: float, standard deviation of Normal distribution for weights.
        - biases_value: float, initial value for biases.
    :return: cntk.ops.functions.Function.
    """
    weights_init_type = kwargs.pop('weights_init_type', 'normal')
    biases_value = kwargs.pop('biases_value', 0.0)

    if weights_init_type == 'he_normal':
        weights_init = C.initializer.he_normal()
    else:    # weights_init_type == 'normal'
        weights_stddev = kwargs.pop('weights_stddev', 0.01)
        weights_init = C.initializer.normal(weights_stddev)
    pad = True if padding == 'SAME' else False    # padding == 'VALID'

    conv_bias = C.layers.Convolution((side_l, side_l), out_depth, sequential=False,
                                     activation=None, init=weights_init, pad=pad, strides=stride,
                                     bias=bias, init_bias=biases_value)(x)
    return conv_bias


def fc_layer(x, out_dim, bias=True, **kwargs):
    """
    Add a new fully-connected layer.
    :param x: shape: ((N, )C, H, W) or ((N, )D).
    :param out_dim: int, the dimension of output vector.
    :param bias: bool, whether to add biases.
    :param kwargs: dict, extra arguments, including weights/biases initialization hyperparameters.
        - weight_stddev: float, standard deviation of Normal distribution for weights.
        - biases_value: float, initial value for biases.
    :return: cntk.ops.functions.Function.
    """
    weights_stddev = kwargs.pop('weights_stddev', 0.01)
    biases_value = kwargs.pop('biases_value', 0.1)

    weights_init = C.initializer.normal(weights_stddev)

    fc_bias = C.layers.Dense(out_dim, activation=None, init=weights_init,
                             bias=bias, init_bias=biases_value)(x)
    return fc_bias


def bn_layer(x, **kwargs):
    """
    Add a new batch normalization layer(after convolutional layer or fully-connected layer).
    :param x: shape: ((N, )C, H, W) or ((N, )D).
    :param kwargs: dict, extra arguments.
    :return: cntk.ops.functions.Function.
    """
    bn = C.layers.BatchNormalization(map_rank=1, init_scale=1,
                                     normalization_time_constant=4096,
                                     disable_regularization=True)(x)    # FIXME
    return bn


def residual_block(x, out_depth, **kwargs):
    """
    Add a new residual building block:
    Conv_3/Sxc-BN-ReLU-Conv_3/1xc-BN+identity-ReLU.
    :param x: shape: ((N, )C, H, W).
    :param out_depth: int, the total number of filters in the two convolutional layers.
    :param kwargs: dict, extra arguments.
    :return: cntk.ops.functions.Function.
        - guided_backprop: bool, whether to perform guided backpropagation(only after training).
    """
    guided_backprop = kwargs.pop('guided_backprop', False)
    relu_fn = C.relu if not guided_backprop else guided_relu

    in_depth = x.shape[0]
    stride = 2 if out_depth > in_depth else 1    # double the initial stride if depth increases
    # out_depth: D, stride: S

    # conv_3/SxD - bn - relu
    conv1 = conv_layer(x, 3, stride, out_depth, padding='SAME', bias=False, **kwargs)
    bn1 = bn_layer(conv1, **kwargs)
    relu1 = relu_fn(bn1)    # ((N, )D, H/S, W/S)

    # conv_3/1xD - bn
    conv2 = conv_layer(relu1, 3, 1, out_depth, padding='SAME', bias=False, **kwargs)
    bn2 = bn_layer(conv2, **kwargs)    # ((N, )D, H/S, W/S)

    # +residual - relu
    if stride > 1:    # should increase x's depth, by 1x1 convolution with doubled stride
        x = conv_layer(x, 1, stride, out_depth, padding='SAME', bias=False, **kwargs)    # ((N, )D, H/S, W/S)
    x_p_residual = x + bn2
    relu3 = relu_fn(x_p_residual)    # ((N, )D, H, W) or ((N, )D, H/S, W/S)

    return relu3


def bottleneck_block(x, mid_depth, out_depth, stride=1, **kwargs):
    """
    Add a new bottleneck building block:
    Conv_1/SxD-BN-ReLU-Conv_3/1xD-BN-ReLU-Conv_1/1xC-BN+identity-ReLU.
    :param x: shape: ((N, )C, H, W).
    :param mid_depth: int, the total number of filters in the first two convolutional layers.
    :param out_depth: int, the total number of filters in the last convolutional layers.
    :param stride: int, the stride of the filters in the 2nd convolutional layer.
    :param kwargs: dict, extra arguments.
    :return: cntk.ops.functions.Function.
        - guided_backprop: bool, whether to perform guided backpropagation(only after training).
    """
    guided_backprop = kwargs.pop('guided_backprop', False)
    relu_fn = C.relu if not guided_backprop else guided_relu

    in_depth = x.shape[0]
    # out_depth: D, mid_depth: d, stride: S

    # conv_1/Sxd - bn - relu
    conv1 = conv_layer(x, 1, 1, mid_depth, padding='SAME', bias=False, **kwargs)
    bn1 = bn_layer(conv1, **kwargs)
    relu1 = relu_fn(bn1)    # ((N, )d, H, W)

    # conv_3/1xd - bn - relu
    conv2 = conv_layer(relu1, 3, stride, mid_depth, padding='SAME', bias=False, **kwargs)
    bn2 = bn_layer(conv2, **kwargs)
    relu2 = relu_fn(bn2)    # ((N, )d, H/S, W/S)

    # conv_1/1xD - bn
    conv3 = conv_layer(relu2, 1, 1, out_depth, padding='SAME', bias=False, **kwargs)
    bn3 = bn_layer(conv3, **kwargs)    # ((N, )D, H/S, W/S)

    # +residual - relu
    if in_depth < out_depth:    # should increase x's depth, by 1x1 convolution
        x = conv_layer(x, 1, stride, out_depth, padding='SAME', bias=False, **kwargs)    # ((N, )D, H/S, W/S)
    x_p_residual = x + bn3
    relu3 = relu_fn(x_p_residual)    # ((N, )D, H, W) or ((N, )D, H/S, W/S)

    return relu3


def guided_relu(x):
    """
    Add a new Guided ReLU nonlinearity.
    :param x: shape: ((N, )C, H, W).
    :return: cntk.ops.functions.Function.
    """

    class GuidedRelu(UserFunction):
        def __init__(self, arg, name='GuidedRelu'):
            super(GuidedRelu, self).__init__([arg], name=name)

        def forward(self, argument, device=None, outputs_to_retain=None):
            return argument, np.maximum(argument, 0)

        def backward(self, state, root_gradients):
            relu_grad = np.where(state > 0, root_gradients, np.zeros_like(root_gradients))
            grad = np.where(root_gradients > 0, relu_grad, np.zeros_like(root_gradients))
            return grad

        def infer_outputs(self):
            return [output_variable(self.inputs[0].shape, self.inputs[0].dtype,
                                    self.inputs[0].dynamic_axes)]

        @staticmethod
        def deserialize(inputs, name, state):
            return GuidedRelu(inputs[0], name)

    return user_function(GuidedRelu(x))






