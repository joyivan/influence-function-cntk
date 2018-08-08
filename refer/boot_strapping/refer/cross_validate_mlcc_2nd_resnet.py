#hwehee's branch
import os
import numpy as np
import cntk as C
import pickle as pkl
from sklearn.model_selection import KFold
from datasets import mlcc_2nd as dataset
from models.nn import ResNet_18 as ConvNet
from learning.optimizers import MomentumOptimizer as Optimizer
from learning.evaluators import ErrorRateEvaluator as Evaluator

from ipdb import set_trace

set_trace()

########################
# it does not use predifined test set
# it only loads all of data in imageset and do KFold among train/test dataset 
# (inner) validation set is defined only once

# this is reasonable when the external testing set is not given.

# if we have an external testing set, then 
# split train set into K chunks in order to do KFold, (5 times training)
# and finally training over unsplited train set and check with test set (1 time training)
########################


""" 1. Load and split datasets """
#root_dir = os.path.join('/', 'mnt', 'sdb2', 'New-ML-Data', '006_samsung_electro_mechanics', 'MLCC',
#                        '2차', 'original_crop192x128')    # FIXME
root_dir = '/Data/006_samsung_electro_mechanics/original_crop192x128'

# Load trainval set and split into train/val sets
X_total, y_total = dataset.read_mlcc_2nd_subset(root_dir, binary=True, mode='total')
total_size = X_total.shape[0]

K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=0)    # FIXME
kf.get_n_splits(X_total)

test_result_dict = {}
for fold_idx, (trainval_idx, test_idx) in enumerate(kf.split(X_total)):
    print('===========================')
    print('**Fold {}'.format(fold_idx))
    test_result_dict[fold_idx] = {}
    X_trainval_fold, y_trainval_fold = X_total[trainval_idx], y_total[trainval_idx]
    X_test_fold, y_test_fold = X_total[test_idx], y_total[test_idx]

    for trainval_ratio in [1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125]:
        trainval_size = int(X_trainval_fold.shape[0] * trainval_ratio)
        print('*Trainval ratio: {}, size: {}'.format(trainval_ratio, trainval_size))
        X_trainval, y_trainval = X_trainval_fold[:trainval_size], y_trainval_fold[:trainval_size]

        val_size = int(trainval_size * 0.2)    # FIXME
        val_set = dataset.DataSet(X_trainval[:val_size], y_trainval[:val_size])
        train_set = dataset.DataSet(X_trainval[val_size:], y_trainval[val_size:])
        test_set = dataset.DataSet(X_test_fold, y_test_fold)

        # Sanity check
        print('Training set stats:')
        print(train_set.images.shape)
        print(train_set.images.min(), train_set.images.max())
        print((train_set.labels[:, 1] == 0).sum(), (train_set.labels[:, 1] == 1).sum())
        print('Validation set stats:')
        print(val_set.images.shape)
        print(val_set.images.min(), val_set.images.max())
        print((val_set.labels[:, 1] == 0).sum(), (val_set.labels[:, 1] == 1).sum())
        print('Test set stats:')
        print(test_set.images.shape)
        print(test_set.images.min(), test_set.images.max())
        print((test_set.labels[:, 1] == 0).sum(), (test_set.labels[:, 1] == 1).sum())

        """ 2. Set training hyperparameters """
        hp_d = dict()
        channel_mean = train_set.images.mean(axis=(0, 2, 3), keepdims=True)    # mean image, shape: (C,)
        image_mean = np.ones((3, 224, 224), dtype=np.float32) * channel_mean[0]    # shape: (C, H, W)
        # np.save('/tmp/mlcc_2nd_mean_fold{}.npy'.format(fold_idx), image_mean)    # save mean image
        hp_d['image_mean'] = image_mean

        # FIXME: Training hyperparameters
        hp_d['batch_size'] = 64
        #hp_d['num_epochs'] = 100
        hp_d['num_epochs'] = 50

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

        """ 4. Start test """
        test_y_pred = model.predict(test_set, **hp_d)
        test_score = evaluator.score(test_set.labels, test_y_pred)
        print('Test error rate: {}'.format(test_score))

        test_result_dict[fold_idx][trainval_ratio] = test_score

with open('cv_mlcc_2nd_resnet_test_result.pkl', 'wb') as fid:
    pkl.dump(test_result_dict, fid)
print('Done.')
