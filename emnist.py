import os, sys
sys.path.append('./refer/datasets-analysis-cntk')
import json
import numpy as np
import time
import scipy.misc

from datasets import dataset
from models.nn import cntk_ConvNet as ConvNet

from modules.influence import get_inverse_hvp_cg, get_inverse_hvp_se, get_influence_val
from modules.utils import visualize_topk_samples 

from cntk.device import try_set_default_device, gpu
try_set_default_device(gpu(0))

############# FIXME ########################
num_sample = 1000
root_dir = '/Data/emnist/balanced/original/'
save_dir = '/Data/result/influence-emnist-sample/{}/'.format(num_sample)
mean_dir = './refer/datasets-analysis-cntk/output/mean_emnist.npy'
net_dir = '/Data/checkpts/emnist/model_fold_1_trainval_ratio_1.0.dnn'
idx_tests = range(0,1)
# check try/except w.r.t save_dir
############# FIXME ########################

from ipdb import set_trace
set_trace()

try:
    trainval_list = np.load(save_dir+'trainval_list.npy')
    test_list = np.load(save_dir+'test_list.npy')
    with open(root_dir+'annotation/annotation1.json', 'r') as fid:
        anno_dict = json.load(fid)
except:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    trainval_list, anno_dict = dataset.read_data_subset(root_dir, mode='train1', sample_size=num_sample)
    test_list, _ = dataset.read_data_subset(root_dir, mode='test')
    np.save(save_dir+'trainval_list.npy', trainval_list)
    np.save(save_dir+'test_list.npy', test_list)

# SNAITY CHECK
print('trainval_list', trainval_list[0:5])

train_set = dataset.LazyDataset(root_dir, trainval_list, anno_dict, rescale=False)
test_set = dataset.LazyDataset(root_dir, test_list, anno_dict, rescale=False)

Ch, H, W = test_set.__getitem__(0)[0].shape

# EMNIST NETWORK
hp_d = dict() # hyperparameters for a network
mean = np.load(mean_dir)
hp_d['image_mean'] = np.transpose(np.tile(mean,(H,W,1)),(2,0,1))

net = ConvNet((Ch,H,W), len(anno_dict['classes']), **hp_d)
net.logits.restore(net_dir)

# EMNIST NETWORK: SANITY CHECK
start_time = time.time()
ys, y_preds, test_score, confusion_matrix = net.predict(test_set, **hp_d)
total_time = time.time() - start_time

print('Test error rate: {}'.format(test_score))
print('Total tack time(sec): {}'.format(total_time))
print('Tact time per image(sec): {}'.format(total_time / len(test_list)))
print('Confusion matrix: \n{}'.format(confusion_matrix))

# SHOW TOP-K SAMPLES

for idx_test in idx_tests:
    # Set a singel test image
    name_test = test_list[idx_test]
    img_test, lb_test = test_set.__getitem__(idx_test)
    lb_true = anno_dict['classes'][str(np.argmax(lb_test))]
    lb_pred = anno_dict['classes'][str(np.argmax(net.logits.eval({net.X:img_test})))]
    print('test image: {} \ntrue label:{} \npredicted label: {}'.format(name_test, lb_true, lb_pred))

    save_path = os.path.join(save_dir, name_test.split('.')[0])

    # Make a folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    scipy.misc.imsave(save_path+'/test_reference_true_{}_pred_{}.png'.format(lb_true,lb_pred),\
            np.squeeze(np.transpose(img_test,(1,2,0))))

    # Calculate with freezed network
    params = net.loss.parameters
    p_ftex = net.d['dense1'].parameters
    p_logreg = tuple(set(params) - set(p_ftex))
    print(p_logreg)
    # FIXME
    from modules.influence import disable_dropout
    net_loss_without_dropout = disable_dropout(net.loss)
    v_logreg = net_loss_without_dropout.grad({net.X:img_test, net.y:lb_test}, wrt=p_logreg) # randomness
    # FIXME

    # Calculate influence function value with some methods

    # Conjugate Gradient
    t1 = time.time()
    ihvp = get_inverse_hvp_cg(net, net.loss, v_logreg, train_set, **{'damping':0.0, 'maxiter':100})
    if_val = get_influence_val(net, net.loss, ihvp, train_set)
    print('CG takes {} sec, and its max/min value {}'.format(time.time()-t1,\
            [max(if_val), min(if_val)]))
    np.save(save_path+'/if_val_cg.npy', if_val)
    visualize_topk_samples(if_val, train_set, num_sample=5, save_path=save_path+'/cg')

