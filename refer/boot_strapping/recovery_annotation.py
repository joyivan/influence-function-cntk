# recover noisy label with several policies

import glob
import json
import numpy as np
import random
import os
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from datasets import dataset28 as dataset
from models.nn import VGG as ConvNet
from learning.evaluators import ErrorRateEvaluator as Evaluator

from ipdb import set_trace

import time

def review(ratio, method):
    # ratio: the ratio of reviewed data. reviewing is done by oracle. the label may or may not be
    # changed. (i.e. if a single reviewed data has correct label, the label won't be changed, and
    # vice versa)
    # method: the methodology of selecting data torch be reviewed. 
    # this can be 'random', 'loss', 'entropy'
    t1 = time.time()

    # FIXME
    anno_dir = '/Data/emnist/balanced/original/annotation/'
    root_dir = '/Data/emnist/balanced/original/'
    checkpt_dir = '/Data/github/interview/save/dropout_0.5_noaugmentation/model_fold_1_trainval_ratio_0.3.dnn'

    with open(anno_dir + 'annotation1.json', 'r') as fid:
        annotation = json.load(fid)

    with open(anno_dir + 'annotation1_wp_0.3.json', 'r') as fid:
        noisy_annotation = json.load(fid)

    image_list = list(noisy_annotation['images'].keys())
    num_image = len(image_list)

    ## sorting
    if method == 'random':
        # random policy
        image_list_random = image_list[:]
        random.shuffle(image_list_random)
        review_list = image_list_random[0:int(num_image * ratio)]
        #review_list_random = np.random.choice(image_list, int(num_image * ratio), replace=False)
    
    else:
        # loss
        image_set = dataset.LazyDataset(root_dir, image_list, annotation)
        model = ConvNet(image_set.__getitem__(0)[0].shape, len(annotation['classes']))
        model.logits.restore(checkpt_dir)
        evaluator = Evaluator()
        
        # extract loss, entropy
        t1_measure = time.time()
        loss, entropy = network_based_measure(model, image_set)
        t2_measure = time.time()
        print('measure extraction takes {}'.format(t2_measure-t1_measure))
        # check data // filename[0] and __getitem__[0] and dataloader first instance
        # -> all of them are same. in other word, we can use an index information
        
        if method == 'loss':
            # loss ascending policy
            idx_loss = np.argsort(loss)[::-1]
            image_list_loss = [image_list[i] for i in idx_loss]
            review_list = image_list_loss[0:int(num_image*ratio)]
        
        elif method == 'entropy':
            # entropy ascending policy
            idx_entropy = np.argsort(entropy)[::-1]
            image_list_entropy = [image_list[i] for i in idx_entropy]
            review_list = image_list_entropy[0:int(num_image*ratio)]

    ## correcting
    num_corrected = 0
    for fname in review_list:
        correct_class = annotation['images'][fname]['class']
        noisy_class = noisy_annotation['images'][fname]['class']
        if noisy_class != correct_class:
            noisy_annotation['images'][fname]['class'] = correct_class
            num_corrected += 1

    # write
    if not os.path.exists(anno_dir+'annotation1_wp_0.3'):
        os.makedirs(anno_dir+'annotation1_wp_0.3')
    h = json.dumps(noisy_annotation, sort_keys=True, indent=4, separators=(',',':'), ensure_ascii=False)
    with open(anno_dir+'annotation1_wp_0.3/annotation1_wp_0.3_review_{}_{}.json'.format(ratio,method), 'w') as f:
        f.write(h)

    t2 = time.time()
    print('{} ratio: {} is done, {}/{} are corrected, it takes {} secs'.format(\
            method, ratio, num_corrected, int(num_image * ratio), t2-t1))

    return num_corrected/int(0.3*len(image_list))

def network_based_measure(model, dataset):
    # return loss and entropy
    batch_size = 256
    num_workers = 6
    dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers)
    
    num_classes = len(dataset.anno_dict['classes'])

    loss = np.empty(0)
    entropy = np.empty(0)
    
    # prediction in batchwise
    for X, y in dataloader:
        X = X.numpy(); y = y.numpy()
        y_pred = model.pred.eval({model.X: X})
        loss_batch = -np.log(np.sum(y_pred * y, axis=1))
        entropy_batch = -np.sum(y_pred * np.log(y_pred), axis=1)
        loss = np.concatenate((loss,loss_batch), axis=0)
        entropy = np.concatenate((entropy, entropy_batch))

    return loss, entropy

def check():
    # print ratio that how much label is corrected and its accuracy w.r.t original one
    return

# main code
set_trace()
rnd = []; ls = []; etp = []

#x = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
#x = [0.1]
x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for ratio in x:
    rnd.append(review(ratio, 'random'))
    ls.append(review(ratio, 'loss'))
    etp.append(review(ratio, 'entropy'))

# draw a graph
set_trace()
fig, ax = plt.subplots(1,1, figsize=(9,9))
_ = ax.plot(x, rnd, color='b', label='random')
_ = ax.plot(x, ls, color='g', label='loss')
_ = ax.plot(x, etp, color='r', label='entropy') 
_ = ax.set_title('Recovery results')
_ = ax.set_ylabel('Ratio of corrected labeled: $Num_{corrected}/Num_{mislabeled}$')
_ = ax.set_xlabel('Ratio of reviewed data: $Num_{reviewed}/Num_{total}$')
_ = ax.set_xticks(x)
_ = plt.legend()
#plt.savefig('./notebooks/images/recovery results.png', bbox_inches='tight')
plt.show()
