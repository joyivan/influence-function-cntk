import os
import numpy as np
import cntk as C
import pickle as pkl
from datasets import skc_5th as dataset
from models.nn import ResNet_18 as ConvNet
from learning.optimizers import MomentumOptimizer as Optimizer
from learning.evaluators import ErrorRateEvaluator as Evaluator

from cntk.device import try_set_default_device, gpu
try_set_default_device(gpu(0))

from ipdb import set_trace

set_trace()

######
# In skc_5th data, the testing set is given. 

# At first, we do not use test.txt data
# Instead of that, we use train(1..5).txt and validation(1..5).txt to check 5 fold CV
# During that, we save model weights w.r.t fold index
# Finally, we'll check performance w.r.t test set among 5 models.
######

""" 1. Load and split datasets """
### FIXME ###
root_dir = '/Data/skc/20180424/original/'

# 5 fold cross validation
test_result_dict = {}
for fold_idx in range(1,6):
    print('===========================')
    print('**Fold {}'.format(fold_idx))
    test_result_dict[fold_idx] = {}
    # Load train and validation set
    X_trainval_fold, y_trainval_fold = dataset.read_skc_5th_subset(root_dir, binary=False, mode='train{}'.format(fold_idx))
    #X_trainval_fold, y_trainval_fold = dataset.read_skc_5th_subset(root_dir, binary=False,
    #        mode='train{}'.format(fold_idx), sample_size=1000)
    X_test_fold, y_test_fold = dataset.read_skc_5th_subset(root_dir, binary=False, mode='validation{}'.format(fold_idx))
    
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
        # np.save('/tmp/skc_5th_mean_fold{}.npy'.format(fold_idx), image_mean)    # save mean image
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
        ### FIXME ###
        # 2 -> num_class
        set_trace()
        print(X_trainval.shape[1:])
        model = ConvNet([3, 224, 224], y_trainval.shape[1], **hp_d)
        evaluator = Evaluator()
        optimizer = Optimizer(model, train_set, evaluator, val_set=val_set, **hp_d)

        print(model.logits.parameters)
        train_results = optimizer.train(details=True, verbose=True, **hp_d)

        """ 4. Start test """
        test_y_pred = model.predict(test_set, **hp_d)
        test_score = evaluator.score(test_set.labels, test_y_pred)
        print('Test error rate: {}'.format(test_score))

        test_result_dict[fold_idx][trainval_ratio] = test_score

with open('cv_skc_5th_resnet_test_result.pkl', 'wb') as fid:
    pkl.dump(test_result_dict, fid)
print('Done.')
