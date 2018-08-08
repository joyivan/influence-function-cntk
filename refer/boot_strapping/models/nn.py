import time
from abc import abstractmethod, abstractproperty
import cntk as C
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from models.layers import max_pool, conv_layer, fc_layer, bn_layer, \
                          residual_block, bottleneck_block, guided_relu


class ConvNet(object):
    """Base class for Convolutional Neural Networks."""

    def __init__(self, input_shape, num_classes, **kwargs):
        """
        Model initializer.
        :param input_shape: tuple, the shape of inputs (C, H, W), ranged [0.0, 1.0].
        :param num_classes: int, the number of classes.
        """
        self.X = C.input_variable(input_shape, dtype=np.float32, name='features')    # ((N, )C, H, W)
        self.y = C.input_variable((num_classes), dtype=np.float32, name='labels')    # ((N, )num_classes)
        # self.is_train = tf.placeholder(tf.bool)

        # Build model and loss function
        self.d = self._build_model(**kwargs)
        self.logits = self.d['logits']
        self.pred = self.d['pred']
        self.loss = self._build_loss(**kwargs)

    @abstractmethod
    def _build_model(self, **kwargs):
        """
        Build model.
        This should be implemented.
        """
        pass

    @abstractmethod
    def _build_loss(self, **kwargs):
        """
        Build loss function for the model training.
        This should be implemented.
        """
        pass

    def predict(self, dataset, verbose=False, **kwargs):
        """
        Make predictions for the given dataset.
        :param dataset: DataSet.
        :param verbose: bool, whether to print details during prediction.
        :param kwargs: dict, extra arguments for prediction.
            - batch_size: int, batch size for each iteration.
            - augment_pred: bool, whether to perform augmentation for prediction.
        :return 
          _y_preds: np.ndarray, shape: (N, num_classes) 
          score: error rate score, scalar
          conf_mat: confusion matrix, (num_classes, num_classes)
        """
        batch_size = kwargs.pop('batch_size', 256)
        augment_pred = kwargs.pop('augment_pred', False)
        num_workers = kwargs.pop('num_workers', 4)

        dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers)
        num_classes = len(dataset.anno_dict['classes'])
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int)
        eye = np.eye(num_classes)

        if verbose:
            print('Running prediction loop...')

        # Start prediction loop
        temp_scores = []
        ys = []
        y_preds = []
        start_time = time.time()

        # prediction in batchwise
        for X, y in dataloader:
            X = X.numpy(); y = y.numpy()
            ys.append(y)
            y_pred = self.pred.eval({self.X: X})
            y_preds.append(y_pred)

            y_ = y.argmax(axis=1)
            y_pred_ = y_pred.argmax(axis=1)
            y_pred_onehot = eye[y_pred_]
            
            temp_score = 1. - accuracy_score(y_, y_pred_)
            temp_scores.append(temp_score)
            
            # confusion matrix in batchwise
            conf = np.sum(np.expand_dims(y,axis=2) * np.expand_dims(y_pred_onehot,axis=1), axis=0)
            conf = np.array(conf, dtype=np.int)
            confusion_matrix += conf

        if verbose:
            print('Total prediction time(sec): {}'.format(time.time() - start_time))

        _ys = np.concatenate(ys, axis=0)
        _y_preds = np.concatenate(y_preds, axis=0)
        score = np.mean(temp_scores)

        return _ys, _y_preds, score, confusion_matrix


class SuaNet(ConvNet):
    """SuaNet class."""

    def _build_model(self, **kwargs):
        """
        Build model.
        :param kwargs: dict, extra arguments for building SuaNet.
            - image_mean: np.ndarray, mean image for each input channel, shape: (C,).
            - guided_backprop: bool, whether to perform guided backpropagation(only after training).
        :return d: dict, containing outputs on each layer.
        """
        guided_backprop = kwargs.pop('guided_backprop', False)
        relu_fn = C.relu if not guided_backprop else guided_relu

        d = dict()    # Dictionary to save intermediate values returned from each layer.
        X_mean = kwargs.pop('image_mean', 0.0)
        num_classes = int(self.y.shape[-1])

        # input
        X_input = C.minus(self.X, X_mean)    # perform mean subtraction

        # conv1 - relu1 - pool1
        d['conv1'] = conv_layer(X_input, 11, 4, 96, padding='SAME',
                                weights_stddev=0.01, biases_value=0.0)
        print('conv1.shape', d['conv1'].shape)
        d['relu1'] = relu_fn(d['conv1'])
        # (3, 224, 224) --> (96, 56, 56)
        d['pool1'] = max_pool(d['relu1'], 3, 2, padding='SAME')
        # (96, 56, 56) --> (96, 28, 28)
        print('pool1.shape', d['pool1'].shape)

        # conv2 - relu2 - pool2
        d['conv2'] = conv_layer(d['pool1'], 5, 1, 256, padding='SAME',
                                weights_stddev=0.01, biases_value=0.1)
        print('conv2.shape', d['conv2'].shape)
        d['relu2'] = relu_fn(d['conv2'])
        # (96, 28, 28) --> (256, 28, 28)
        d['pool2'] = max_pool(d['relu2'], 3, 2, padding='SAME')
        # (256, 28, 28) --> (256, 14, 14)

        # conv3 - relu3
        d['conv3'] = conv_layer(d['pool2'], 3, 1, 256, padding='SAME',
                                weights_stddev=0.01, biases_value=0.0)
        print('covn3.shape', d['conv3'].shape)
        d['relu3'] = relu_fn(d['conv3'])
        # (256, 14, 14) --> (256, 14, 14)

        # pool3: global average pooling
        d['pool3'] = C.layers.GlobalAveragePooling()(d['relu3'])
        print('pool3.shape', d['pool3'].shape)
        # (256, 14, 14) --> (256, 1, 1)

        # Flatten feature maps: (256, 1, 1) --> (256)
        # fc4
        d['logits'] = fc_layer(d['pool3'], num_classes,
                               weights_stddev=0.01, biases_value=0.0)
        # (256) --> (num_classes)

        # softmax
        d['pred'] = C.softmax(d['logits'])

        return d

    def _build_loss(self, **kwargs):
        """
        Build loss function for the model training.
        :param kwargs: dict, extra arguments.
        :return cntk.ops.functions.Function.
        """
        # Softmax cross-entropy loss function
        softmax_loss = C.cross_entropy_with_softmax(self.logits, self.y)

        return softmax_loss


class AlexNet(ConvNet):
    """AlexNet class."""

    def _build_model(self, **kwargs):
        """
        Build model.
        :param kwargs: dict, extra arguments for building AlexNet.
            - image_mean: np.ndarray, mean image for each input channel, shape: (C,).
            - dropout_prob: float, the probability of dropping out each unit in FC layer.
            - guided_backprop: bool, whether to perform guided backpropagation(only after training).
        :return d: dict, containing outputs on each layer.
        """
        guided_backprop = kwargs.pop('guided_backprop', False)
        relu_fn = C.relu if not guided_backprop else guided_relu

        d = dict()    # Dictionary to save intermediate values returned from each layer.
        X_mean = kwargs.pop('image_mean', 0.0)
        dropout_prob = kwargs.pop('dropout_prob', 0.0)
        num_classes = int(self.y.shape[-1])

        # input
        X_input = C.minus(self.X, X_mean)    # perform mean subtraction

        # conv1 - relu1 - pool1
        d['conv1'] = conv_layer(X_input, 11, 4, 96, padding='VALID',
                                weights_stddev=0.01, biases_value=0.0)
        print('conv1.shape', d['conv1'].shape)
        d['relu1'] = relu_fn(d['conv1'])
        # (3, 227, 227) --> (96, 55, 55)
        d['pool1'] = max_pool(d['relu1'], 3, 2, padding='VALID')
        # (96, 55, 55) --> (96, 27, 27)
        print('pool1.shape', d['pool1'].shape)

        # conv2 - relu2 - pool2
        d['conv2'] = conv_layer(d['pool1'], 5, 1, 256, padding='SAME',
                                weights_stddev=0.01, biases_value=0.1)
        print('conv2.shape', d['conv2'].shape)
        d['relu2'] = relu_fn(d['conv2'])
        # (96, 27, 27) --> (256, 27, 27)
        d['pool2'] = max_pool(d['relu2'], 3, 2, padding='VALID')
        # (256, 27, 27) --> (256, 13, 13)
        print('pool2.shape', d['pool2'].shape)

        # conv3 - relu3
        d['conv3'] = conv_layer(d['pool2'], 3, 1, 384, padding='SAME',
                                weights_stddev=0.01, biases_value=0.0)
        print('conv3.shape', d['conv3'].shape)
        d['relu3'] = relu_fn(d['conv3'])
        # (256, 13, 13) --> (384, 13, 13)

        # conv4 - relu4
        d['conv4'] = conv_layer(d['relu3'], 3, 1, 384, padding='SAME',
                                weights_stddev=0.01, biases_value=0.1)
        print('conv4.shape', d['conv4'].shape)
        d['relu4'] = relu_fn(d['conv4'])
        # (384, 13, 13) --> (384, 13, 13)

        # conv5 - relu5 - pool5
        d['conv5'] = conv_layer(d['relu4'], 3, 1, 256, padding='SAME',
                                weights_stddev=0.01, biases_value=0.1)
        print('conv5.shape', d['conv5'].shape)
        d['relu5'] = relu_fn(d['conv5'])
        # (384, 13, 13) --> (256, 13, 13)
        d['pool5'] = max_pool(d['relu5'], 3, 2, padding='VALID')
        # (256, 13, 13) --> (256, 6, 6)
        print('pool5.shape', d['pool5'].shape)

        # Flatten feature maps: (256, 6, 6) --> (9216)
        # fc6
        d['fc6'] = fc_layer(d['pool5'], 4096,
                            weights_stddev=0.005, biases_value=0.1)
        d['relu6'] = relu_fn(d['fc6'])
        d['drop6'] = C.dropout(d['relu6'], dropout_rate=dropout_prob)
        # (9216) --> (4096)
        print('drop6.shape', d['drop6'].shape)

        # fc7
        d['fc7'] = fc_layer(d['drop6'], 4096,
                            weights_stddev=0.005, biases_value=0.1)
        d['relu7'] = relu_fn(d['fc7'])
        d['drop7'] = C.dropout(d['relu7'], dropout_rate=dropout_prob)
        # (4096) --> (4096)
        print('drop7.shape', d['drop7'].shape)

        # fc8
        d['logits'] = fc_layer(d['drop7'], num_classes,
                               weights_stddev=0.01, biases_value=0.0)
        # (4096) --> (num_classes)

        # softmax
        d['pred'] = C.softmax(d['logits'])

        return d

    def _build_loss(self, **kwargs):
        """
        Build loss function for the model training.
        :param kwargs: dict, extra arguments.
        :return cntk.ops.functions.Function.
        """
        # Softmax cross-entropy loss function
        softmax_loss = C.cross_entropy_with_softmax(self.logits, self.y)

        return softmax_loss


class CIFARNet(ConvNet):
    """CIFARNet class(for CIFAR-10 classification)."""

    def _build_model(self, **kwargs):
        """
        Build model.
        :param kwargs: dict, extra arguments for building CIFARNet.
            - image_mean: np.ndarray, mean image for each input channel, shape: (C,).
        :return d: dict, containing outputs on each layer.
        """
        guided_backprop = kwargs.pop('guided_backprop', False)
        relu_fn = C.relu if not guided_backprop else guided_relu

        d = dict()    # Dictionary to save intermediate values returned from each layer.
        X_mean = kwargs.pop('image_mean', 0.0)
        num_classes = int(self.y.shape[-1])

        # input
        X_input = C.minus(self.X, X_mean)    # perform mean subtraction

        # conv1 - relu1 - pool1
        d['conv1'] = conv_layer(X_input, 5, 1, 32, padding='SAME',
                                weights_stddev=0.01, biases_value=0.0)
        print('conv1.shape', d['conv1'].shape)
        d['relu1'] = relu_fn(d['conv1'])
        # (3, 32, 32) --> (32, 32, 32)
        d['pool1'] = max_pool(d['relu1'], 3, 2, padding='VALID')
        # (32, 32, 32) --> (32, 15, 15)
        print('pool1.shape', d['pool1'].shape)

        # conv2 - relu2 - pool2
        d['conv2'] = conv_layer(d['pool1'], 5, 1, 32, padding='SAME',
                                weights_stddev=0.01, biases_value=0.1)
        print('conv2.shape', d['conv2'].shape)
        d['relu2'] = relu_fn(d['conv2'])
        # (32, 15, 15) --> (32, 15, 15)
        d['pool2'] = max_pool(d['relu2'], 3, 2, padding='VALID')
        # (32, 15, 15) --> (32, 7, 7)
        print('pool2.shape', d['pool2'].shape)

        # conv3 - relu3 - pool3
        d['conv3'] = conv_layer(d['pool2'], 5, 1, 64, padding='SAME',
                                weights_stddev=0.01, biases_value=0.0)
        print('conv3.shape', d['conv3'].shape)
        d['relu3'] = relu_fn(d['conv3'])
        # (32, 7, 7) --> (64, 7, 7)
        d['pool3'] = max_pool(d['relu3'], 3, 2, padding='VALID')
        # (64, 7, 7) --> (64, 3, 3)
        print('pool3.shape', d['pool3'].shape)

        # Flatten feature maps: (64, 3, 3) --> (576)
        # fc4
        d['fc4'] = fc_layer(d['pool3'], 64,
                            weights_stddev=0.01, biases_value=0.1)
        d['relu4'] = relu_fn(d['fc4'])
        # (576) --> (64)
        print('drop4.shape', d['relu4'].shape)

        # fc5
        d['logits'] = fc_layer(d['relu4'], num_classes,
                               weights_stddev=0.01, biases_value=0.0)
        # (64) --> (num_classes)

        # softmax
        d['pred'] = C.softmax(d['logits'])

        return d

    def _build_loss(self, **kwargs):
        """
        Build loss function for the model training.
        :param kwargs: dict, extra arguments.
        :return cntk.ops.functions.Function.
        """
        # Softmax cross-entropy loss function
        softmax_loss = C.cross_entropy_with_softmax(self.logits, self.y)

        return softmax_loss


class ResNet_18(ConvNet):
    """ResNet-18 class."""

    def _build_model(self, **kwargs):
        """
        Build model.
        :param kwargs: dict, extra arguments for building ResNet-50.
            - image_mean: np.ndarray, mean image for each input channel, shape: (C,).
        :return d: dict, containing outputs on each layer.
        """

        guided_backprop = kwargs.pop('guided_backprop', False)
        relu_fn = C.relu if not guided_backprop else guided_relu

        d = dict()    # Dictionary to save intermediate values returned from each layer.
        X_mean = kwargs.pop('image_mean', 0.0)
        num_classes = int(self.y.shape[-1])

        # input
        X_input = C.minus(self.X, X_mean)    # perform mean subtraction

        # conv1 - bn - relu1 - pool1
        d['conv1'] = conv_layer(X_input, 7, 2, 64, padding='SAME',
                                weights_init_type='he_normal')
        print('conv1.shape', d['conv1'].shape)
        d['conv1_bn'] = bn_layer(d['conv1'], **kwargs)
        d['conv1_relu'] = relu_fn(d['conv1_bn'])
        # (3, 224, 224) --> (64, 112, 112)
        d['pool1'] = max_pool(d['conv1_relu'], 3, 2, padding='SAME')
        # (64, 112, 112) --> (64, 56, 56)
        print('pool1.shape', d['pool1'].shape)

        # conv2_x
        for b in range(2):
            prev = 'pool1' if b == 0 else 'conv2_{}'.format(b)
            d['conv2_{}'.format(b+1)] = residual_block(d[prev], 64,
                                                       weights_init_type='he_normal', **kwargs)
        print('conv2_2.shape', d['conv2_2'].shape)
        # (64, 56, 56) --> (64, 56, 56)

        # conv3_x
        for b in range(2):
            prev = 'conv2_2' if b == 0 else 'conv3_{}'.format(b)
            d['conv3_{}'.format(b+1)] = residual_block(d[prev], 128,
                                                       weight_init_type='he_normal', **kwargs)
        print('conv3_2.shape', d['conv3_2'].shape)
        # (64, 56, 56) --> (128, 28, 28)

        # conv4_x
        for b in range(2):
            prev = 'conv3_2' if b == 0 else 'conv4_{}'.format(b)
            d['conv4_{}'.format(b+1)] = residual_block(d[prev], 256,
                                                       weight_init_type='he_normal', **kwargs)
        print('conv4_2.shape', d['conv4_2'].shape)
        # (128, 28, 28) --> (256, 14, 14)

        # conv5_x
        for b in range(2):
            prev = 'conv4_2' if b == 0 else 'conv5_{}'.format(b)
            d['conv5_{}'.format(b+1)] = residual_block(d[prev], 512,
                                                       weight_init_type='he_normal', **kwargs)
        print('conv5_2.shape', d['conv5_2'].shape)
        # (256, 14, 14) --> (512, 7, 7)

        # pool6: global average pooling
        d['pool6'] = C.layers.GlobalAveragePooling()(d['conv5_2'])
        print('pool6.shape', d['pool6'].shape)
        # (2048, 7, 7) --> (2048, 1, 1)

        # Flatten feature maps: (2048, 1, 1) --> (2048)
        # fc7
        d['logits'] = fc_layer(d['pool6'], num_classes,
                               weights_stddev=0.01, biases_value=0.0)
        # (4096) --> (num_classes)

        # softmax
        d['pred'] = C.softmax(d['logits'])

        return d

    def _build_loss(self, **kwargs):
        """
        Build loss function for the model training.
        :param kwargs: dict, extra arguments.
        :return cntk.ops.functions.Function.
        """
        # Softmax cross-entropy loss function
        softmax_loss = C.cross_entropy_with_softmax(self.logits, self.y)

        return softmax_loss


class ResNet_50(ConvNet):
    """ResNet-50 class."""

    def _build_model(self, **kwargs):
        """
        Build model.
        :param kwargs: dict, extra arguments for building ResNet-50.
            - image_mean: np.ndarray, mean image for each input channel, shape: (C,).
        :return d: dict, containing outputs on each layer.
        """
        guided_backprop = kwargs.pop('guided_backprop', False)
        relu_fn = C.relu if not guided_backprop else guided_relu

        d = dict()    # Dictionary to save intermediate values returned from each layer.
        X_mean = kwargs.pop('image_mean', 0.0)
        num_classes = int(self.y.shape[-1])

        # input
        X_input = C.minus(self.X, X_mean)    # perform mean subtraction

        # conv1 - bn - relu1 - pool1
        d['conv1'] = conv_layer(X_input, 7, 2, 64, padding='SAME',
                                weights_init_type='he_normal')
        print('conv1.shape', d['conv1'].shape)
        d['conv1_bn'] = bn_layer(d['conv1'], **kwargs)
        d['conv1_relu'] = relu_fn(d['conv1_bn'])
        # (3, 224, 224) --> (64, 112, 112)
        d['pool1'] = max_pool(d['conv1_relu'], 3, 2, padding='SAME')
        # (64, 112, 112) --> (64, 56, 56)
        print('pool1.shape', d['pool1'].shape)

        # conv2_x
        for b in range(3):
            stride = 1
            prev = 'pool1' if b == 0 else 'conv2_{}'.format(b)
            d['conv2_{}'.format(b+1)] = bottleneck_block(d[prev], 64, 256, stride=stride,
                                                         weights_init_type='he_normal', **kwargs)
        print('conv2_3.shape', d['conv2_3'].shape)
        # (64, 56, 56) --> (256, 56, 56)

        # conv3_x
        for b in range(4):
            stride = 2 if b == 0 else 1
            prev = 'conv2_3' if b == 0 else 'conv3_{}'.format(b)
            d['conv3_{}'.format(b+1)] = bottleneck_block(d[prev], 128, 512, stride=stride,
                                                         weights_init_type='he_normal', **kwargs)
        print('conv3_4.shape', d['conv3_4'].shape)
        # (256, 56, 56) --> (512, 28, 28)

        # conv4_x
        for b in range(6):
            stride = 2 if b == 0 else 1
            prev = 'conv3_4' if b == 0 else 'conv4_{}'.format(b)
            d['conv4_{}'.format(b+1)] = bottleneck_block(d[prev], 256, 1024, stride=stride,
                                                         weights_init_type='he_normal', **kwargs)
        print('conv4_6.shape', d['conv4_6'].shape)
        # (512, 28, 28) --> (1024, 14, 14)

        # conv5_x
        for b in range(3):
            stride = 2 if b == 0 else 1
            prev = 'conv4_6' if b == 0 else 'conv5_{}'.format(b)
            d['conv5_{}'.format(b+1)] = bottleneck_block(d[prev], 512, 2048, stride=stride,
                                                         weights_init_type='he_normal', **kwargs)
        print('conv5_3.shape', d['conv5_3'].shape)
        # (1024, 14, 14) --> (2048, 7, 7)

        # pool6: global average pooling
        d['pool6'] = C.layers.GlobalAveragePooling()(d['conv5_3'])
        print('pool6.shape', d['pool6'].shape)
        # (2048, 7, 7) --> (2048, 1, 1)

        # Flatten feature maps: (2048, 1, 1) --> (2048)
        # fc7
        d['logits'] = fc_layer(d['pool6'], num_classes,
                               weights_stddev=0.01, biases_value=0.0)
        # (4096) --> (num_classes)

        # softmax
        d['pred'] = C.softmax(d['logits'])

        return d

    def _build_loss(self, **kwargs):
        """
        Build loss function for the model training.
        :param kwargs: dict, extra arguments.
        :return cntk.ops.functions.Function.
        """
        # Softmax cross-entropy loss function
        softmax_loss = C.cross_entropy_with_softmax(self.logits, self.y)

        return softmax_loss

class VGG_28(ConvNet):
    """VGG like 28 x 28 input network"""

    def _build_model(self, **kwargs):
        """
        Build model.
        :param kwargs: dict, extra arguments for building VGG like network.
            - image_mean: np.ndarray, mean image for each input channel, shape: (C,).
        :return d: dict, containing outputs on each layer.
        """

        guided_backprop = kwargs.pop('guided_backprop', False)
        relu_fn = C.relu if not guided_backprop else guided_relu

        d = dict()    # Dictionary to save intermediate values returned from each layer.
        X_mean = kwargs.pop('image_mean', 0.0)
        num_classes = int(self.y.shape[-1])

        # input
        X_input = C.minus(self.X, X_mean)    # perform mean subtraction

        # conv1-1 - relu1-1 - conv1-2 - relu1-2 - pool1
        d['conv1-1'] = conv_layer(X_input, 3, 1, 64, padding='SAME')
        print('conv1-1.shape', d['conv1-1'].shape)
        d['conv1-1_relu'] = relu_fn(d['conv1-1'])
        # (1, 28, 28) --> (64, 28, 28)
        d['conv1-2'] = conv_layer(d['conv1-1_relu'], 3, 1, 64, padding='SAME')
        print('conv1-2.shape', d['conv1-2'].shape)
        d['conv1-2_relu'] = relu_fn(d['conv1-2'])
        # (64, 28, 28) --> (64, 28, 28)
        d['pool1'] = max_pool(d['conv1-2_relu'], 2, 2, padding='SAME')
        # (64, 28, 28) --> (64, 14, 14)
        print('pool1.shape', d['pool1'].shape)

        # conv2-1 - relu2-1 - conv2-2 - relu2-2 - pool2
        d['conv2-1'] = conv_layer(d['pool1'], 3, 1, 128, padding='SAME')
        print('conv2-1.shape', d['conv2-1'].shape)
        d['conv2-1_relu'] = relu_fn(d['conv2-1'])
        # (64, 14, 14) --> (128, 14, 14)
        d['conv2-2'] = conv_layer(d['conv2-1_relu'], 3, 1, 128, padding='SAME')
        print('conv2-2.shape', d['conv2-2'].shape)
        d['conv2-2_relu'] = relu_fn(d['conv2-2'])
        # (128, 14, 14) --> (128, 14, 14)
        d['pool2'] = max_pool(d['conv2-2_relu'], 2, 2, padding='SAME')
        # (128, 14, 14) --> (128, 7, 7)
        print('pool2.shape', d['pool2'].shape)

        # conv3-1 - relu3-1 - conv3-2 - relu3-2 - conv3-3 - relu3-3 - conv3-4 - relu3-4 - pool3
        d['conv3-1'] = conv_layer(d['pool2'], 3, 1, 256, padding='SAME')
        print('conv3-1.shape', d['conv3-1'].shape)
        d['conv3-1_relu'] = relu_fn(d['conv3-1'])
        # (128, 7, 7) --> (256, 7, 7)
        d['conv3-2'] = conv_layer(d['conv3-1_relu'], 3, 1, 256, padding='SAME')
        print('conv3-2.shape', d['conv3-2'].shape)
        d['conv3-2_relu'] = relu_fn(d['conv3-2'])
        # (256, 7, 7) --> (256, 7, 7)
        d['conv3-3'] = conv_layer(d['conv3-2_relu'], 3, 1, 256, padding='SAME')
        print('conv3-3.shape', d['conv3-3'].shape)
        d['conv3-3_relu'] = relu_fn(d['conv3-3'])
        # (256, 7, 7) --> (256, 7, 7)
        d['conv3-4'] = conv_layer(d['conv3-3_relu'], 3, 1, 256, padding='SAME')
        print('conv3-4.shape', d['conv3-4'].shape)
        d['conv3-4_relu'] = relu_fn(d['conv3-4'])
        # (256, 7, 7) --> (256, 7, 7)
        d['pool3'] = max_pool(d['conv3-4_relu'], 2, 2, padding='SAME')
        # (256, 7, 7) --> (256, 4, 4)
        print('pool3.shape', d['pool3'].shape)

        # flatten and dense
        d['logits'] = fc_layer(d['pool3'], num_classes,
                weights_stddev=0.01, biases_value=0.0)

        # softmax
        d['pred'] = C.softmax(d['logits'])

        return d

    def _build_loss(self, **kwargs):
        """
        Build loss function for the model training.
        :param kwargs: dict, extra arguments.
        :return cntk.ops.functions.Function.
        """
        # Softmax cross-entropy loss function
        softmax_loss = C.cross_entropy_with_softmax(self.logits, self.y)

        return softmax_loss

class VGG(ConvNet):
    """Simplest VGG network"""

    def _build_model(self, **kwargs):
        """
        Build model.
        :param kwargs: dict, extra arguments for building VGG like network.
            - image_mean: np.ndarray, mean image for each input channel, shape: (C,).
        :return d: dict, containing outputs on each layer.
        """

        guided_backprop = kwargs.pop('guided_backprop', False)
        relu_fn = C.relu if not guided_backprop else guided_relu

        d = dict()    # Dictionary to save intermediate values returned from each layer.
        X_mean = kwargs.pop('image_mean', 0.0)
        num_classes = int(self.y.shape[-1])

        # input
        X_input = C.minus(self.X, X_mean)    # perform mean subtraction

        # conv1 - relu1 - pool1
        d['conv1'] = conv_layer(X_input, 3, 1, 64, padding='SAME')
        print('conv1.shape', d['conv1'].shape)
        d['conv1_relu'] = relu_fn(d['conv1'])
        # (1, 28, 28) --> (64, 28, 28)
        d['pool1'] = max_pool(d['conv1_relu'], 2, 2, padding='SAME')
        # (64, 28, 28) --> (64, 14, 14)
        print('pool1.shape', d['pool1'].shape)

        # conv2 - relu2 - pool2
        d['conv2'] = conv_layer(d['pool1'], 3, 1, 128, padding='SAME')
        print('conv2.shape', d['conv2'].shape)
        d['conv2_relu'] = relu_fn(d['conv2'])
        # (64, 14, 14) --> (128, 14, 14)
        d['pool2'] = max_pool(d['conv2_relu'], 2, 2, padding='SAME')
        # (128, 14, 14) --> (128, 7, 7)
        print('pool2.shape', d['pool2'].shape)

        # conv3 - relu3 - pool3
        d['conv3'] = conv_layer(d['pool2'], 3, 1, 256, padding='SAME')
        print('conv3.shape', d['conv3'].shape)
        d['conv3_relu'] = relu_fn(d['conv3'])
        # (128, 7, 7) --> (256, 7, 7)
        #d['conv3-2'] = conv_layer(d['conv3'], 3, 1, 256, padding='SAME')
        #print('conv3-2.shape', d['conv3-2'].shape)
        #d['conv3-2_relu'] = relu_fn(d['conv3-2'])
        ## (128, 7, 7) --> (256, 7, 7)
        d['pool3'] = max_pool(d['conv3_relu'], 2, 2, padding='SAME')
        # (256, 7, 7) --> (256, 4, 4)
        print('pool3.shape', d['pool3'].shape)

        # flatten and dense
        d['dense1'] = fc_layer(d['pool3'], 2048,
                weights_stddev=0.01, biases_value=0.0)
        d['dense1_dropout'] = C.layers.Dropout(dropout_rate=0.5)(d['dense1'])
        d['logits'] = fc_layer(d['dense1_dropout'], num_classes,
                weights_stddev=0.01, biases_value=0.0)

        # softmax
        d['pred'] = C.softmax(d['logits'])

        return d

    def _build_loss(self, **kwargs):
        """
        Build loss function for the model training.
        :param kwargs: dict, extra arguments.
        :return cntk.ops.functions.Function.
        """
        beta = kwargs.pop('beta', 1.0)
        method = kwargs.pop('method', 'soft')
        stop = kwargs.pop('stop', True)

        # Softmax cross-entropy loss function
        softmax_loss = C.cross_entropy_with_softmax(self.logits, self.y)

        # boot-strapping loss
        if method == 'soft':
            boot_loss = -C.reduce_sum(C.element_times(self.pred, C.log(self.pred)), axis=0)
        elif method == 'hard':
            boot_loss = -C.reduce_sum(C.element_times(C.hardmax(self.pred), C.log(self.pred)), axis=0)

        if stop:
            boot_loss = C.stop_gradient(boot_loss)

        loss = beta * softmax_loss + (1-beta) * boot_loss

        return loss

