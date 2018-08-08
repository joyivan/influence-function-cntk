import os
import numpy as np
import cntk as C
import pickle as pkl
from torch.utils.data import DataLoader

from datasets import skc as dataset
from models.nn import ResNet_18 as ConvNet
from learning.optimizers import MomentumOptimizer as Optimizer
from learning.evaluators import ErrorRateEvaluator as Evaluator

from cntk.device import try_set_default_device, gpu
try_set_default_device(gpu(0))

from ipdb import set_trace

#set_trace()

######
# In skc data, the testing set is given. 

# At first, we do not use test.txt data
# Instead of that, we use train(1..5).txt and validation(1..5).txt to check 5 fold CV
# During that, we save model weights w.r.t fold index
# Finally, we'll check performance w.r.t test set among 5 models.
######

""" 1. Load and split datasets """
### FIXME ###
root_dir = '/Data/skc/20180424/original/'

test_list, anno_dict = dataset.read_skc_subset(root_dir, mode='validation5')
test_set = dataset.Dataset(root_dir, test_list, anno_dict)

""" 2. Set training hyperparameters """
hp_d = dict()
mean = np.array([0.3711091, 0.3711091, 0.3711091], dtype=np.float32)
image_mean = np.transpose(np.tile(mean,(224,224,1)),[2,0,1])    # shape: (C, H, W)
# np.save('/tmp/skc_5th_mean_fold{}.npy'.format(fold_idx), image_mean)    # save mean image
hp_d['image_mean'] = image_mean

# FIXME: Spec hyperparameters
hp_d['num_workers'] = 6

# FIXME: Training hyperparameters
hp_d['batch_size'] = 64
hp_d['num_epochs'] = 100

hp_d['augment_train'] = True

hp_d['init_learning_rate'] = 0.05
hp_d['momentum'] = 0.9
hp_d['learning_rate_patience'] = 10
hp_d['learning_rate_decay'] = 0.1
hp_d['eps'] = 1e-8

# FIXME: Regularization hyperparameters
hp_d['weight_decay'] = 0.0001

# FIXME: Evaluation hyperparameters
hp_d['score_threshold'] = 1e-4

""" 3. Load model """
set_trace()
model = ConvNet([3, 224, 224], len(anno_dict['classes']), **hp_d)
model.logits.restore('/Data/checkpts/model_trainval1.dnn')
#evaluator = Evaluator()
#optimizer = Optimizer(model, train_set, evaluator, val_set=val_set, **hp_d)

print(model.logits.parameters)

""" 4. Start test """
_, _, test_score, confusion_matrix = model.predict(test_set, **hp_d)
print('Test error rate: {}'.format(test_score))
print('Confusion matrix:\n {}'.format(confusion_matrix))

with open('confusion matrix.pkl', 'wb') as fid:
    pkl.dump(confusion_matrix, fid)
print('Done.')
