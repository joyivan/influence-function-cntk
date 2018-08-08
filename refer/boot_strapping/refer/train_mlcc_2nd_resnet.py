import os
import numpy as np
import cntk as C
from datasets import mlcc_2nd as dataset
from models.nn import ResNet_18 as ConvNet
from learning.optimizers import MomentumOptimizer as Optimizer
from learning.evaluators import ErrorRateEvaluator as Evaluator

from cntk.device import try_set_default_device, gpu
try_set_default_device(gpu(0))

from ipdb import set_trace

set_trace()

""" 1. Load and split datasets """
#root_dir = os.path.join('/', 'mnt', 'sdb2', 'New-ML-Data', '006_samsung_electro_mechanics', 'MLCC',
#                        '2차', 'original_crop192x128')    # FIXME
root_dir = '/Data/006_samsung_electro_mechanics/original_crop192x128'

# Load trainval set and split into train/val sets
#X_trainval, y_trainval = dataset.read_mlcc_2nd_subset(root_dir, binary=True, mode='train')
#                                                      # sample_size=2000)    # FIXME
#trainval_size = X_trainval.shape[0]
#val_size = int(trainval_size * 0.2)    # FIXME 
## ::: we've already divide train & validation set at imageset1
#val_set = dataset.DataSet(X_trainval[:val_size], y_trainval[:val_size])
#train_set = dataset.DataSet(X_trainval[val_size:], y_trainval[val_size:])

X_train, y_train = dataset.read_mlcc_2nd_subset(root_dir, binary=True, mode='train')
X_val, y_val = dataset.read_mlcc_2nd_subset(root_dir, binary=True, mode='validation')
train_set = dataset.DataSet(X_train, y_train)
val_set = dataset.DataSet(X_val, y_val)

# Sanity check
print('Training set stats:')
print(train_set.images.shape)
print(train_set.images.min(), train_set.images.max())
print((train_set.labels[:, 1] == 0).sum(), (train_set.labels[:, 1] == 1).sum())
print('Validation set stats:')
print(val_set.images.shape)
print(val_set.images.min(), val_set.images.max())
print((val_set.labels[:, 1] == 0).sum(), (val_set.labels[:, 1] == 1).sum())


""" 2. Set training hyperparameters """
hp_d = dict()
channel_mean = train_set.images.mean(axis=(0, 2, 3), keepdims=True)    # mean image, shape: (C,)
image_mean = np.ones((3, 224, 224), dtype=np.float32) * channel_mean[0]    # shape: (C, H, W)
np.save('/tmp/mlcc_2nd_mean.npy', image_mean)    # save mean image
hp_d['image_mean'] = image_mean

# FIXME: Training hyperparameters
hp_d['batch_size'] = 64
hp_d['num_epochs'] = 300

hp_d['augment_train'] = True
hp_d['augment_pred'] = False

hp_d['init_learning_rate'] = 0.1
hp_d['momentum'] = 0.9
hp_d['learning_rate_patience'] = 30
hp_d['learning_rate_decay'] = 0.1
hp_d['eps'] = 1e-8

# FIXME: Regularization hyperparameters
hp_d['weight_decay'] = 0.0001

# FIXME: Evaluation hyperparameters
hp_d['score_threshold'] = 1e-4


""" 3. Start training """
model = ConvNet([3, 224, 224], 2, **hp_d)
evaluator = Evaluator()
optimizer = Optimizer(model, train_set, evaluator, val_set=val_set, **hp_d)

print(model.logits.parameters)
train_results = optimizer.train(details=True, verbose=True, **hp_d)
