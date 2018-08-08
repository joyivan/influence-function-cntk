import os
import numpy as np
import cntk as C
import pickle as pkl
from torch.utils.data import DataLoader

from datasets import dataset as dataset # FIXME
from models.nn import ResNet_18 as ConvNet
from learning.optimizers import MomentumOptimizer as Optimizer
from learning.evaluators import ErrorRateEvaluator as Evaluator

from cntk.device import try_set_default_device, gpu
try_set_default_device(gpu(0))

from ipdb import set_trace

set_trace()

""" 1. Load and split datasets """
### FIXME ###
root_dir = '/Data/emnist/balanced/original/'

# 5 fold cross validation
test_result_dict = {}
for fold_idx in range(1,6):
    print('===========================')
    print('**Fold {}'.format(fold_idx))
    test_result_dict[fold_idx] = {}
    # Load train and validation set
    trainval_list, anno_dict = dataset.read_data_subset(root_dir,\
            mode='train{}'.format(fold_idx))
    test_list, _ = dataset.read_data_subset(root_dir,\
            mode='validation{}'.format(fold_idx))
    #trainval_list, anno_dict = dataset.read_data_subset(root_dir,\
    #        mode='train{}'.format(fold_idx),sample_size=1000)
    #test_list, _ = dataset.read_data_subset(root_dir,\
    #        mode='validation{}'.format(fold_idx),sample_size=1000)
    
    for trainval_ratio in [1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125]:
        trainval_size = int(len(trainval_list) * trainval_ratio)
        print('*Trainval ratio: {}, size: {}'.format(trainval_ratio, trainval_size))
        trainval_list_ratio = trainval_list[:trainval_size]
        
        val_size = int(trainval_size * 0.2) # FIXME
        val_set = dataset.Dataset(root_dir, trainval_list_ratio[:val_size], anno_dict)
        train_set = dataset.Dataset(root_dir, trainval_list_ratio[val_size:], anno_dict)
        test_set = dataset.Dataset(root_dir, test_list, anno_dict)

        mean = np.zeros(3)

        # Sanity check

        print('Training set stats:')
        lbs = anno_dict['classes'].keys()
        lbs_dict = {key:0 for key in lbs}
        dataloader = DataLoader(train_set, 512, shuffle=False, num_workers=6)
        mn = 1; mx = 0 # Since we've normalized image
        mean = []
        print(len(dataloader))
        it = 0
        for X,y in dataloader:
            X = X.numpy()
            print(it); it += 1
            mn = min(X.min(),mn)
            mx = max(X.max(),mx)
            mean.append(X.mean(axis=(0,2,3))) # channel mean; shape:(C,)
            for y_ in y:
                lb = np.argmax(y_)
                lbs_dict[str(int(lb))] += 1
            break ######### remove this
        mean = np.mean(mean,axis=0) # shape: (N, C) -> (C,)
        print('X shape:', X.shape)
        print('y shape:', y.shape)
        print('min/max value of X',mn,mx)
        print('channel mean value of X', mean)
        #for ld in lbs_dict:
        #    print(anno_dict['classes'][ld], lbs_dict[ld])

        """ 2. Set training hyperparameters """
        hp_d = dict()
        # fold1, trainval1 -> mean: [0.62663692 0.58812195 0.50663531]
        image_mean = np.transpose(np.tile(mean,(224,224,1)),[2,0,1])    # shape: (C, H, W)
        # np.save('/tmp/lg_innotek_mean_fold{}.npy'.format(fold_idx), image_mean)    # save mean image
        hp_d['image_mean'] = image_mean

        # FIXME: Spec hyperparameters
        hp_d['num_workers'] = 6

        # FIXME: Training hyperparameters
        hp_d['batch_size'] = 64
        hp_d['num_epochs'] = 100 # FIXME

        hp_d['augment_train'] = True

        hp_d['init_learning_rate'] = 0.10
        hp_d['momentum'] = 0.9
        hp_d['learning_rate_patience'] = 30
        hp_d['learning_rate_decay'] = 0.1
        hp_d['eps'] = 1e-8

        # FIXME: Regularization hyperparameters
        hp_d['weight_decay'] = 0.0001

        # FIXME: Evaluation hyperparameters
        hp_d['score_threshold'] = 1e-4

        """ 3. Start training """
        #model = ConvNet([3, 224, 224], len(anno_dict['classes']), **hp_d)
        model = ConvNet(image_mean.shape, len(anno_dict['classes']), **hp_d)
        evaluator = Evaluator()
        optimizer = Optimizer(model, train_set, evaluator, val_set=val_set,
                fold_idx=fold_idx, trainval_ratio=trainval_ratio, **hp_d)

        print(model.logits.parameters)
        train_results = optimizer.train(details=True, verbose=True, **hp_d)

        """ 4. Start test """
        _, _, test_score, confusion_matrix = model.predict(test_set, **hp_d)
        print('Test error rate: {}'.format(test_score))
        print('Confusion matrix: {}'.format(confusion_matrix))

        test_result_dict[fold_idx][trainval_ratio] = test_score

        print('intermediate test_result_dict value:\n{}'.format(test_result_dict))
        with open('./output/intermediate_cv_emnist_resnet_test_result.pkl', 'wb') as fid:
            pkl.dump(test_result_dict, fid)

with open('./output/cv_emnist_resnet_test_result.pkl', 'wb') as fid:
    pkl.dump(test_result_dict, fid)
print('Done.')
