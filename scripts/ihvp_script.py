
# coding: utf-8

# Toy example을 제외하고, 코드 구현, 결과를 통합해서 visualization하기 위한 script
# 
# 다루고 있는 것으로는
# 
# # Code Implementation
# - HVP
# - IHVP-CG
# - IHVP-NCG
# - IHVP-SE
# 
# # Experimental Result
# 
# - TOP-k examples sorted by IF measure
# - t-sne result
# - interpretation of result
# - TOP-k examples sorted by IF measure on specific class
# - relabeling using IF measure
# 

# # Code Implementation

# ## HVP
# 
# Remark)
# 
# tensorflow에서는 gradient가 operator로 존재해서 이를 이용하면 hessian vector product를 automatic differentiation으로 구현할 수 있음. 그 결과 error 없이 값을 찾을 수 있음.
# 
# 반면 cntk에서는 gradient output이 값으로만 나오게 됨. 따라서 numerical differentiation을 이용해서 구현함. 그 결과 error가 조금 발생함.

# In[1]:

import cntk as C
from cntk.device import try_set_default_device, gpu
try_set_default_device(gpu(0))

import numpy as np
import time
import os

from torch.utils.data import DataLoader

from ipdb import set_trace

set_trace()
# go to 557

# In[2]:

# Hessian Vector Product

def grad_inner_product(grad1, grad2):
    # inner product for dictionary-format gradients (output scalar value)
    
    val = 0
    
    assert(len(grad1)==len(grad2))
    
    for ks in grad1.keys():
        val += np.sum(np.multiply(grad1[ks],grad2[ks]))
        
    return val

def weight_update(w, v, r):
    # w: weights of neural network (tuple)
    # v: value for delta w (dictionary, e.g., gradient value)
    # r: hyperparameter for a gradient (scalar)

    for p in w:
        p.value += r * v[p]

def HVP(y, x, v):
    # Calculate Hessian vector product 
    # y: scalar function to be differentiated (function, e.g. cross entropy loss)
    # x: feed_dict value for the network (dictionary, e.g. {model.X: image_batch, model.y: label_batch})
    # v: vector to be producted (by Hessian) (numeric dictionary, e.g., g(z_test))
    ## w: variables to differentiate (numeric, e.g. neural network weight)
    
    # hyperparameter r
    r = 1e-2
    
    assert type(x)==dict, "Input of HVP is wrong. 'x' should be dictionary(feed dict format)"
    assert type(v)==dict, "Input of HVP is wrong. 'v' should be dictionary(weight:value format)"

    w = v.keys()
    
    # gradient for plus
    weight_update(w, v, +r)
    g_plus = y.grad(x, wrt=w)
  
    # gradient for minus
    weight_update(w, v, -2*r)
    g_minus = y.grad(x, wrt=w)
    
    # weight reconstruction
    weight_update(w, v, +r)
    
    hvp = {ks: (g_plus[ks] - g_minus[ks])/(2*r) for ks in g_plus.keys()}
       
    return hvp


# ## IHVP
# 
# 논문에서 나온 conjugate gradient와 stochastic estimation 두 방법을 모두 구현함.
# 
# Remark)
# 
# 원작자 코드에서는 conjugate gradient를 구현할 때 scipy의 ncg (Newton's conjugate gradient)를 사용해서 구현함. 하지만 ncg의 경우 update를 담당하는 outer loop 안에 update 수치를 찾기 위한 작은 inner loop를 하나 더 돌게 되는데, 적절한 해를 찾지 못할 경우 이 inner loop를 벗어나지 못할 가능성이 있음.
# - 과거 버전에서는 while loop을 사용해서 진행되어 평생 벗어나지 못할 가능성이 있음.
# - 최신 버전에서는 for loop을 사용해서 진행되어 cg_maxiter를 넘기면 벗어날 가능성이 있으나 이 값은 내부에서만 존재하는 hyperparameter라서 바꿔줄 수 없음. 내부에서 지정된 값은 20 * len(x0)인데, 우리의 경우 len(x0)가 80000이 넘어가기 때문에 사실상 며칠을 돌려도 끝나지 않음. (몇 십분 돌렸을 때 겨우 400번 정도 돌았음.)
# 
# 때문에 scipy의 cg를 사용해서 구현함. 이 경우 maxiter를 이용하면 수렴하지 않더라도 학습을 중간에 끝낼 수 있음.

# In[3]:

# Newton-Conjugate Gradient

from scipy.optimize import fmin_ncg

def dic2vec(dic):
    # convert a dictionary with matrix values to a 1D vector
    # e.g. gradient of network -> 1D vector
    vec = np.concatenate([val.reshape(-1) for val in dic.values()])
    
    return vec

def vec2dic(vec, fmt):
    # convert a 1D vector to a dictionary of format fmt
    # fmt = {key: val.shape for (key,val) in dict}
    fmt_idx = [np.prod(val) for val in fmt.values()]
    #lambda ls, idx: [ls[sum(idx[:i]):sum(idx[:i+1])] for i in range(len(idx))]
    vec_split = [vec[sum(fmt_idx[:i]):sum(fmt_idx[:i+1])] for i in range(len(fmt_idx))]
    dic = {key: vec_split[i].reshape(shape) for (i,(key,shape)) in enumerate(fmt.items())}

    return dic

def get_inverse_hvp_ncg(model, y, v, data_set, **kwargs):
    # return x, which is the solution of QP, whose value is H^-1 v
    # kwargs: hyperparameters for conjugate gradient
    batch_size = kwargs.pop('batch_size', 50)
    damping = kwargs.pop('damping', 0.0)
    avextol = kwargs.pop('avextol', 1e-8)
    maxiter = kwargs.pop('maxiter', 1e1)
    
    dataloader = DataLoader(data_set, batch_size, shuffle=True, num_workers=6)

    def HVP_minibatch_val(y, v):
        # Calculate Hessian vector product w.r.t whole dataset
        # y: scalar function output of the neural network (e.g. model.loss)
        # v: vector to be producted by inverse hessian (i.e.H^-1 v) (numeric dictionary, e.g. v_test)
        
        ## model: neural network model (e.g. model)
        ## dataloader: training set dataloader
        ## damping: damp term to make hessian convex

        hvp_batch = {ks: [] for ks in v.keys()}

        for img, lb in dataloader:
            img = img.numpy(); lb = lb.numpy()
            x_feed = {model.X: img, model.y:lb}
            hvp = HVP(y,x_feed,v)
            # add hvp value
            [hvp_batch[ks].append(hvp[ks]/img.shape[0]) for ks in hvp.keys()]

        hvp_mean = {ks: np.mean(hvp_batch[ks], axis=0) + damping*v[ks] for ks in hvp_batch.keys()}

        return hvp_mean

    def get_fmin_loss(x):
        x_dic = vec2dic(x, {key: val.shape for (key, val) in v.items()})
        hvp_val = HVP_minibatch_val(y, x_dic)

        return 0.5 * grad_inner_product(hvp_val, x_dic) - grad_inner_product(v, x_dic)

    def get_fmin_grad(x):
        # x: 1D vector
        x_dic = vec2dic(x, {key: val.shape for (key, val) in v.items()})
        hvp_val = HVP_minibatch_val(y, x_dic)
        hvp_flat = dic2vec(hvp_val)
        v_flat = dic2vec(v)

        return hvp_flat - v_flat
    
    def get_fmin_hvp(x, p):
        p_dic = vec2dic(p, {key: val.shape for (key, val) in v.items()})
        hvp_val = HVP_minibatch_val(y, p_dic)
        hvp_flat = dic2vec(hvp_val)

        return hvp_flat
    
    fmin_loss_fn = get_fmin_loss
    fmin_grad_fn = get_fmin_grad
    fmin_hvp_fn = get_fmin_hvp
    
    fmin_results = fmin_ncg(            f = fmin_loss_fn, x0 = dic2vec(v), fprime = fmin_grad_fn,            fhess_p = fmin_hvp_fn, avextol = avextol, maxiter = maxiter)
    
    return vec2dic(fmin_results, {key: val.shape for (key, val) in v.items()})


# In[4]:

# Conjugate Gradient

from scipy.optimize import fmin_cg

def dic2vec(dic):
    # convert a dictionary with matrix values to a 1D vector
    # e.g. gradient of network -> 1D vector
    vec = np.concatenate([val.reshape(-1) for val in dic.values()])
    
    return vec

def vec2dic(vec, fmt):
    # convert a 1D vector to a dictionary of format fmt
    # fmt = {key: val.shape for (key,val) in dict}
    fmt_idx = [np.prod(val) for val in fmt.values()]
    #lambda ls, idx: [ls[sum(idx[:i]):sum(idx[:i+1])] for i in range(len(idx))]
    vec_split = [vec[sum(fmt_idx[:i]):sum(fmt_idx[:i+1])] for i in range(len(fmt_idx))]
    dic = {key: vec_split[i].reshape(shape) for (i,(key,shape)) in enumerate(fmt.items())}

    return dic

def get_inverse_hvp_cg(model, y, v, data_set, **kwargs):
    # return x, which is the solution of QP, whose value is H^-1 v
    # kwargs: hyperparameters for conjugate gradient
    batch_size = kwargs.pop('batch_size', 50)
    damping = kwargs.pop('damping', 0.0)
    maxiter = kwargs.pop('maxiter', 5e1)
    
    dataloader = DataLoader(data_set, batch_size, shuffle=True, num_workers=6)

    def HVP_minibatch_val(y, v):
        # Calculate Hessian vector product w.r.t whole dataset
        # y: scalar function output of the neural network (e.g. model.loss)
        # v: vector to be producted by inverse hessian (i.e.H^-1 v) (numeric dictionary, e.g. v_test)
        
        ## model: neural network model (e.g. model)
        ## dataloader: training set dataloader
        ## damping: damp term to make hessian convex

        hvp_batch = {ks: [] for ks in v.keys()}

        for img, lb in dataloader:
            img = img.numpy(); lb = lb.numpy()
            x_feed = {model.X: img, model.y:lb}
            hvp = HVP(y,x_feed,v)
            # add hvp value
            [hvp_batch[ks].append(hvp[ks]/img.shape[0]) for ks in hvp.keys()]

        hvp_mean = {ks: np.mean(hvp_batch[ks], axis=0) + damping*v[ks] for ks in hvp_batch.keys()}

        return hvp_mean

    def get_fmin_loss(x):
        x_dic = vec2dic(x, {key: val.shape for (key, val) in v.items()})
        hvp_val = HVP_minibatch_val(y, x_dic)

        return 0.5 * grad_inner_product(hvp_val, x_dic) - grad_inner_product(v, x_dic)

    def get_fmin_grad(x):
        # x: 1D vector
        x_dic = vec2dic(x, {key: val.shape for (key, val) in v.items()})
        hvp_val = HVP_minibatch_val(y, x_dic)
        hvp_flat = dic2vec(hvp_val)
        v_flat = dic2vec(v)

        return hvp_flat - v_flat

    fmin_loss_fn = get_fmin_loss
    fmin_grad_fn = get_fmin_grad
    
    fmin_results = fmin_cg(f=get_fmin_loss, x0=dic2vec(v), fprime=fmin_grad_fn, maxiter=maxiter)
    
    return vec2dic(fmin_results, {key: val.shape for (key, val) in v.items()})


# In[5]:

# Stochastic Estimation

def get_inverse_hvp_se(model, y, v, data_set, **kwargs):
    # Calculate inverse hessian vector product over the training set
    # model: neural network model (e.g. model)
    # y: scalar function output of the neural network (e.g. model.loss)
    # v: vector to be producted by inverse hessian (i.e.H^-1 v) (e.g. v_test)
    # data_set: training set to be summed in Hessian
    # kwargs: hyperparameters for stochastic estimation
    recursion_depth = kwargs.pop('recursion_depth', 50) # epoch
    scale = kwargs.pop('scale', 1e1) # similar to learning rate
    damping = kwargs.pop('damping', 0.0) # paper reference: 0.01
    batch_size = kwargs.pop('batch_size', 1)
    num_samples = kwargs.pop('num_samples', 1) # the number of samples(:stochatic estimation of IF) to be averaged
    verbose = kwargs.pop('verbose', False)
    
    dataloader = DataLoader(data_set, batch_size, shuffle=True, num_workers=6)
    
    inv_hvps = []
    
    params = v.keys()
    
    for i in range(num_samples):
        # obtain num_samples inverse hvps
        cur_estimate = v
        
        for depth in range(recursion_depth):
            # epoch-scale recursion depth
            t1 = time.time()
            for img, lb in dataloader:
                img = img.numpy(); lb = lb.numpy()
                x_feed = {model.X: img, model.y:lb}
                hvp = HVP(y,x_feed,cur_estimate)
                # cur_estimate = v + (1-damping)*cur_estimate + 1/scale*(hvp/batch_size)
                cur_estimate = {ks: v[ks] + (1-damping/scale)*cur_estimate[ks] - (1/scale)*hvp[ks]/batch_size for ks in cur_estimate.keys()}
            
            if verbose:
                print('#w: \n', list(map(lambda x: x.value, params)), '\n#hvp: \n', hvp, '\n#ihvp: \n', cur_estimate)
            
            cur_norm = np.sqrt(grad_inner_product(cur_estimate,cur_estimate))
            print('Recursion depth: {}, norm: {}, time: {} \n'.format(depth, cur_norm,time.time()-t1))
            
            if np.isnan(cur_norm):
                print('## The result has been diverged ##')
                break
        
        inv_hvp = {ks: (1/scale)*cur_estimate[ks] for ks in cur_estimate.keys()}
        inv_hvps.append(inv_hvp)
    
    inv_hvp_val = {ks: np.mean([inv_hvps[i][ks] for i in range(num_samples)], axis=0) for ks in inv_hvps[0].keys()}
    
    return inv_hvp_val


# ## Toy Example for IHVP (CG, NCG, SE)
# 
# 간단한 neural network를 사용해서 위 알고리즘이 잘 동작하는지 확인.
# 
# 사실 network의 Hessian은 w에 대해서 locally convex함. 따라서 수렴하지 않거나 발산할 가능성이 있음. 
# 
# 하지만 w를 고정시켜두고 이를 진행시켰을 때 만약 알고리즘이 locally convex한 경우에서도 잘 동작한다면, (1.25, -0.083) 값이 나와야 함.
# 
# 세 알고리즘 모두 원하는 값에 잘 수렴함을 확인함.
# (SE의 경우에는 scale에 따라서 발산할 때도 있음.)
# 
# 이에 대한 자세한 결과는 ihvp_toy.ipynb를 참고

# In[6]:

# toy example for inverse HVP (CG, NCG and SE)

class SimpleNet(object):
    def __init__(self):
        self.X = C.input_variable(shape=(1,))
        self.h = C.layers.Dense(1, activation=None, init=C.uniform(1), bias=False)(self.X)
        self.pred = C.layers.Dense(1, activation=None, init=C.uniform(1), bias=False)(self.h)
        self.y = C.input_variable(shape=(1,))
        self.loss = C.squared_error(self.pred, self.y)
        
class SimpleDataset(object):
    def __init__(self, images, labels):
        self._images, self._labels = images, labels
    
    def __getitem__(self, index):
        X = self._images[index]
        y = self._labels[index]
        
        return X, y
    
    def __len__(self):
        return len(self._images)


net = SimpleNet()

params = net.pred.parameters

x_feed = {net.X:np.array([[2.]],dtype=np.float32), net.y:np.array([[1.]],dtype=np.float32)}
v_feed = {p: np.ones_like(p.value) for p in params}

print('w1 = \n', params[0].value, '\nw2 = \n', params[1].value, '\nloss = \n', net.loss.eval(x_feed))
params[0].value = np.asarray([[1.]])
params[1].value = np.asarray([[1./3.]])
print('w1 = \n', params[0].value, '\nw2 = \n', params[1].value, '\nloss = \n', net.loss.eval(x_feed))

print('hvp', HVP(net.loss, x_feed, v_feed))

#images = np.asarray([[2.],[2.]], dtype=np.float32)
#labels = np.asarray([[1.],[1.]], dtype=np.float32)
images = np.asarray([[2.]], dtype=np.float32)
labels = np.asarray([[1.]], dtype=np.float32)

train_set = SimpleDataset(images,labels)

print('######## damping = 0.0, desired solution: [1.25, -0.08] ########'); t1 = time.time()
ihvp_ncg = get_inverse_hvp_ncg(net, net.loss, v_feed, train_set, **{'damping': 0.0}); t2 = time.time()
ihvp_cg = get_inverse_hvp_cg(net, net.loss, v_feed, train_set, **{'damping': 0.0}); t3 = time.time()
ihvp_se = get_inverse_hvp_se(net, net.loss, v_feed, train_set, **{'damping': 0.0, 'recursion_depth': 100}); t4 = time.time()
print('inverse hvp_ncg', ihvp_ncg, '\ntime: ', t2-t1)
print('inverse hvp_cg', ihvp_cg, '\ntime: ', t3-t2 )
print('inverse hvp_se', ihvp_se, '\ntime: ', t4-t3)

# print('inverse hvp_ncg', get_inverse_hvp_ncg(net, net.loss, v_feed, train_set, **{'damping': 0.1}))
# print('inverse hvp_cg', get_inverse_hvp_cg(net, net.loss, v_feed, train_set, **{'damping': 0.1}))
# print('inverse hvp_se', get_inverse_hvp_se(net, net.loss, v_feed, train_set, **{'scale':10, 'damping':0.1}))


# # Experimental Result
# 
# Noisy EMNIST dataset을 사용해서 실험을 진행함.
# 
# 이 데이터를 사용하는 이유는
# - EMNIST dataset은 일반적으로 사용하는 typo이기 때문에 직관적으로 해석할 수 있음.
# - noisy label 문제를 다루기 때문에 이와 연결지어 해석할 수 있음.
# - 과거 학습된 network를 가지고 있음. (suawiki/noisy label 참고)

# In[117]:

import os, sys
sys.path.append('../refer/boot_strapping')
import json

from datasets import dataset28 as dataset

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.misc
from torch.utils.data import DataLoader


# In[8]:

def show_image_from_data(img):
    # show image from dataset
    # img: (C,W,H) numpy array
    img_show = np.squeeze(np.transpose(img, [1,2,0]))
    imshow(img_show)
    plt.show()


# In[124]:

def IF_val(net, ihvp, data_set):
    # Calculate influence function w.r.t ihvp and data_set
    # This should be done in sample-wise, since the gradient operation will sum up over whole feed-dicted data
    
    # ihvp: inverse hessian vector product (dictionary)
    # data_set: data_set to be feed to the gradient operation (dataset)
    IF_list = []
    
    #params = net.logits.parameters
    params = ihvp.keys()
    
    dataloader = DataLoader(data_set, 1, shuffle=False, num_workers=6)
    
    t1 = time.time()
    for img, lb in dataloader:
        img = img.numpy(); lb = lb.numpy()
        gd = net.loss.grad({net.X:img, net.y:lb}, wrt=params)
        IF = -grad_inner_product(ihvp, gd) / len(dataloader)
        IF_list.append(IF)
    print('IF_val takes {} sec'.format(time.time()-t1))
        
    return IF_list

def visualize_topk_samples(measure, train_set, num_sample=5, verbose='ALL', save_path='./result'):
    # 'ALL': show DISADV / ADV / INF / NEG examples
    # 'ADV': show ADV only
    # 'DIS': show DIS only
    
    argsort = np.argsort(measure) 
    topk = argsort[-1:-num_sample-1:-1]
    botk = argsort[0:num_sample]

    if verbose == 'DIS' or verbose == 'ALL':
        dis = []
        true_label = ''; noisy_label = ''
        print('\n## SHOW {}-MOST DISADVANTAGEOUS EXAMPLES ##\n'.format(num_sample))
        for idx in topk:
            img, lb = train_set.__getitem__(idx)
            show_image_from_data(img)
            print('training set name: ', train_set.filename_list[idx])
            print('training set label: ', train_set.anno_dict['classes'][str(np.argmax(lb))])
            print('IF measure: ', measure[idx])
            print(trainval_list[idx])
            dis.append(img)
            true_label += train_set.filename_list[idx].split('_')[1]
            noisy_label += train_set.anno_dict['classes'][str(np.argmax(lb))]
        dis = np.squeeze(np.concatenate(dis, axis=2))
        scipy.misc.imsave(save_path+'/disadvantageous_true_{}_noisy_{}.png'.format(true_label, noisy_label), dis)

    if verbose == 'ADV' or verbose == 'ALL':
        adv = []
        true_label = ''; noisy_label = ''
        print('\n## SHOW {}-MOST ADVANTAGEOUS EXAMPLES ##\n'.format(num_sample))
        for idx in botk:
            img, lb = train_set.__getitem__(idx)
            show_image_from_data(img)
            print('training set name: ', train_set.filename_list[idx])
            print('training set label: ', train_set.anno_dict['classes'][str(np.argmax(lb))])
            print('IF measure: ', measure[idx])
            print(trainval_list[idx])
            adv.append(img)
            true_label += train_set.anno_dict['classes'][str(np.argmax(lb))]
            noisy_label += train_set.filename_list[idx].split('_')[1]
        adv = np.squeeze(np.concatenate(adv, axis=2))
        scipy.misc.imsave(save_path+'/advantageous_true_{}_noisy_{}.png'.format(true_label, noisy_label), adv)

    if verbose == 'ALL':
        argsort_abs = np.argsort(np.abs(measure))
        topk_abs = argsort_abs[-1:-num_sample-1:-1]
        botk_abs = argsort_abs[0:num_sample]

        print('\n## SHOW {}-MOST INFLUENTIAL EXAMPLES ##\n'.format(num_sample))
        for idx in topk_abs:
            img, lb = train_set.__getitem__(idx)
            show_image_from_data(img)
            print('training set name: ', train_set.filename_list[idx])
            print('training set label: ', train_set.anno_dict['classes'][str(np.argmax(lb))])
            print('IF measure: ', measure[idx])

        print('\n## SHOW {}-MOST NEGLIGIBLE EXAMPLES ##\n'.format(num_sample))
        for idx in botk_abs:
            img, lb = train_set.__getitem__(idx)
            show_image_from_data(img)
            print('training set name: ', train_set.filename_list[idx])
            print('training set label: ', train_set.anno_dict['classes'][str(np.argmax(lb))])
            print('IF measure: ', measure[idx])
        
    return 0


# In[26]:

# emnist dataset
root_dir = '/Data/emnist/balanced/original'

# sample size
#trainval_list, anno_dict = dataset.read_data_subset(root_dir, mode='train1', sample_size=1000)
trainval_list, anno_dict = dataset.read_data_subset(root_dir, mode='train1')
#test_list, _ = dataset.read_data_subset(root_dir, mode='validation1', sample_size=100)
test_list, _ = dataset.read_data_subset(root_dir, mode='validation1')

with open('/Data/emnist/balanced/original/annotation/annotation1_wp_0.3.json','r') as fid:
    noisy_anno_dict = json.load(fid)

train_set = dataset.LazyDataset(root_dir, trainval_list, noisy_anno_dict)
test_set = dataset.LazyDataset(root_dir, test_list, anno_dict)

# emnist dataset: SANITY CHECK
print(len(test_set), type(test_set))
print(len(test_list))


# In[27]:

# emnist network
from models.nn import VGG as ConvNet

hp_d = dict() # hyperparameters for a network
net = ConvNet(train_set.__getitem__(0)[0].shape, len(anno_dict['classes']), **hp_d)
#net.logits.restore('/Data/checkpts/noisy/model_fold_1_trainval_ratio_1.0.dnn')
net.logits.restore('/Data/checkpts/noisy/model_fold_1_trainval_ratio_0.3.dnn')

# emnist network: SANITY CHECK
start_time = time.time()
ys, y_preds, test_score, confusion_matrix = net.predict(test_set, **hp_d)
total_time = time.time() - start_time

print('Test error rate: {}'.format(test_score))
print('Total tack time(sec): {}'.format(total_time))
print('Tact time per image(sec): {}'.format(total_time / len(test_list)))
print('Confusion matrix: \n{}'.format(confusion_matrix))


## In[153]:
#
## emnist network
#from models.nn import VGG as ConvNet
#tt_list, _ = dataset.read_data_subset(root_dir, mode='validation1')
#tt_set = dataset.LazyDataset(root_dir, tt_list, anno_dict)
#
#hp_d = dict() # hyperparameters for a network
#net = ConvNet(train_set.__getitem__(0)[0].shape, len(anno_dict['classes']), **hp_d)
#net.logits.restore('/Data/checkpts/noisy/model_fold_1_trainval_ratio_1.0.dnn')
#
## emnist network: SANITY CHECK
#start_time = time.time()
#ys, y_preds, test_score, confusion_matrix = net.predict(tt_set, **hp_d)
#total_time = time.time() - start_time
#
#print('Test error rate: {}'.format(test_score))
#print('Total tack time(sec): {}'.format(total_time))
#print('Tact time per image(sec): {}'.format(total_time / len(tt_list)))
#print('Confusion matrix: \n{}'.format(confusion_matrix))
#
#
## In[160]:
#
## Set a single test image
#
## # Re-sample a test instance
## test_list, _ = dataset.read_data_subset(root_dir, mode='validation1', sample_size=100)
## test_set = dataset.LazyDataset(root_dir, test_list, anno_dict)
#
#params = net.logits.parameters
#
#idx_test = 0
#
#name_test = test_list[idx_test]
#img_test, lb_test = test_set.__getitem__(idx_test)
#show_image_from_data(img_test)
#v_test = net.loss.grad({net.X:img_test, net.y:lb_test}, wrt=params)
#
#print('testfile name: ', name_test)
#print('ground truth label: ', anno_dict['classes'][str(np.argmax(lb_test))])
#print('network prediction: ', anno_dict['classes'][str(np.argmax(net.logits.eval({net.X:img_test})))])
#
#save_path = os.path.join('./result', name_test.split('.')[0])
#if not os.path.exists(save_path):
#    # make folder
#    os.makedirs(save_path)
#
#np.save(save_path+'/trainval_list', trainval_list)
#np.save(save_path+'/test_list', test_list)
#
#
## In[164]:
#
## CALCULATE IF WITH FREEZED NETWORK
#
#params = net.loss.parameters
#p_ftex = net.d['dense1'].parameters
#p_logreg = tuple(set(params) - set(p_ftex)) # extract the weights of the last-layer (w,b)
#print(p_logreg)
#v_logreg = net.loss.grad({net.X:img_test, net.y:lb_test}, wrt=p_logreg)
#
## Calculate influence functions
#
## the solution which is converged properly can be found within 10 iterations, otherwise does not converge
#t1 = time.time()
##ihvp_cg_logreg = get_inverse_hvp_cg(net, net.loss, v_logreg, train_set,**{'damping':0.0, 'maxiter':10})
#IF_cg_logreg = np.load(save_path+'/if_cg_logreg.npy')
##IF_cg_logreg = IF_val(net, ihvp_cg_logreg, train_set)
#print('CG_logreg takes {} sec, and its max/min value {}'.format(time.time()-t1, [max(IF_cg_logreg),min(IF_cg_logreg)]))
#
##np.save(save_path+'/if_cg_logreg.npy', IF_cg_logreg)
#
## t1 = time.time()
## ihvp_ncg_logreg = get_inverse_hvp_ncg(net, net.loss, v_logreg, train_set,**{'damping':0.1, 'maxiter':3})
## IF_ncg_logreg = IF_val(net, ihvp_ncg_logreg, train_set)
## print('NCG_logreg takes {} sec, and its value {}'.format(time.time()-t1, IF_ncg_logreg))
#
## Restore weights
#net.logits.restore('/Data/checkpts/noisy/model_fold_1_trainval_ratio_1.0.dnn')
#
#
## In[159]:
#
#test_list[0]#'train_n_77228.png'
#
#
## # Visualization
## show_image_from_data(img_test)
## print('ground truth label: ', anno_dict['classes'][str(np.argmax(lb_test))])
## test_list[0]
## visualize_topk_samples(IF_cg_logreg, train_set, num_sample=5, save_path=save_path)
## # visualize_topk_samples(IF_ncg_logreg, num_sample=5)
#
## In[77]:
#
## influence function w.r.t. label mask
#
## mask_n = ['n' in exmp for exmp in trainval_list]
## train_n_77228.png
## anno_dict['classes'][str(np.argmax(lb_test))]
#
#mask_n = [anno_dict['images'][exmp]['class'][0] == np.argmax(lb_test) for exmp in trainval_list]
## mask_n = [noisy_anno_dict['images'][exmp]['class'][0] == np.argmax(lb_test) for exmp in trainval_list]
#print(np.sum(mask_n))
#
#IF_msk_n = [e[0]*e[1] for e in zip(mask_n, IF_cg_logreg)]
#IF_msk_not_n = [(1-e[0])*e[1] for e in zip(mask_n, IF_cg_logreg)]
#
#print('show n examples only')
#visualize_topk_samples(IF_msk_n, train_set, num_sample=5, verbose='ADV')
#print('show non-n examples only')
#visualize_topk_samples(IF_msk_not_n, train_set, num_sample=5, verbose='DIS')
#
#
## In[78]:
#
## influence function with training example
#
## Sample a train instance
#params = net.logits.parameters
#
#idx_train = 0
#
#name_train = trainval_list[idx_train]
#img_train, lb_train = train_set.__getitem__(idx_train)
#show_image_from_data(img_train)
#v_train = net.loss.grad({net.X:img_train, net.y:lb_train}, wrt=params)
#
#print('trainfile name: ', name_train)
#print('ground truth label: ', anno_dict['classes'][str(np.argmax(lb_train))])
#print('network prediction: ', anno_dict['classes'][str(np.argmax(net.logits.eval({net.X:img_train})))])
#
#save_path = os.path.join('./result', name_train.split('.')[0])
#if not os.path.exists(save_path):
#    # make folder
#    os.makedirs(save_path)
#
#np.save(save_path+'/trainval_list', trainval_list)
#np.save(save_path+'/test_list', test_list)
#
## CALCULATE IF WITH FREEZED NETWORK
#
#params = net.loss.parameters
#p_ftex = net.d['dense1'].parameters
#p_logreg = tuple(set(params) - set(p_ftex)) # extract the weights of the last-layer (w,b)
#print(p_logreg)
#v_logreg = net.loss.grad({net.X:img_train, net.y:lb_train}, wrt=p_logreg)
#
## Calculate influence functions
#
## the solution which is converged properly can be found within 10 iterations, otherwise does not converge
#t1 = time.time()
#ihvp_cg_logreg = get_inverse_hvp_cg(net, net.loss, v_logreg, train_set,**{'damping':0.0, 'maxiter':10})
#IF_cg_logreg = IF_val(net, ihvp_cg_logreg, train_set)
#print('CG_logreg takes {} sec, and its max/min value {}'.format(time.time()-t1, [max(IF_cg_logreg),min(IF_cg_logreg)]))
#
#np.save(save_path+'/if_cg_logreg.npy', IF_cg_logreg)
#
## t1 = time.time()
## ihvp_ncg_logreg = get_inverse_hvp_ncg(net, net.loss, v_logreg, train_set,**{'damping':0.1, 'maxiter':3})
## IF_ncg_logreg = IF_val(net, ihvp_ncg_logreg, train_set)
## print('NCG_logreg takes {} sec, and its value {}'.format(time.time()-t1, IF_ncg_logreg))
#
## Restore weights
#net.logits.restore('/Data/checkpts/noisy/model_fold_1_trainval_ratio_1.0.dnn')
#
#
## In[80]:
#
#visualize_topk_samples(IF_cg_logreg, train_set, num_sample=5)
#print('vector')
#IF_vec = IF_val(net, v_test, train_set)
#visualize_topk_samples(IF_vec, train_set, num_sample=5)
#
#
## In[151]:
#
## DO THIS FOR SEVERAL EXAMPLES
#
#for idx_test in range(86, 156):
#    # Set a single test image
#
#    # # Re-sample a test instance
#    # test_list, _ = dataset.read_data_subset(root_dir, mode='validation1', sample_size=100)
#    # test_set = dataset.LazyDataset(root_dir, test_list, anno_dict)
#
#    params = net.logits.parameters
#
#    #idx_test = 0
#
#    name_test = test_list[idx_test]
#    img_test, lb_test = test_set.__getitem__(idx_test)
#    show_image_from_data(img_test)
#    v_test = net.loss.grad({net.X:img_test, net.y:lb_test}, wrt=params)
#    
#    lb_true = anno_dict['classes'][str(np.argmax(lb_test))]
#    lb_pred = anno_dict['classes'][str(np.argmax(net.logits.eval({net.X:img_test})))]
#    print('testfile name: ', name_test)
#    print('ground truth label: ', lb_true)
#    print('network prediction: ', lb_pred)
#
#    save_path = os.path.join('./result', name_test.split('.')[0])
#    if not os.path.exists(save_path):
#        # make folder
#        os.makedirs(save_path)
#        
#    scipy.misc.imsave(save_path+'/test_reference_true_{}_pred_{}.png'.format(lb_true,lb_pred), np.squeeze(img_test))
#
#    np.save(save_path+'/trainval_list', trainval_list)
#    np.save(save_path+'/test_list', test_list)
#
#    # CALCULATE IF WITH FREEZED NETWORK
#
#    params = net.loss.parameters
#    p_ftex = net.d['dense1'].parameters
#    p_logreg = tuple(set(params) - set(p_ftex)) # extract the weights of the last-layer (w,b)
#    print(p_logreg)
#    v_logreg = net.loss.grad({net.X:img_test, net.y:lb_test}, wrt=p_logreg)
#
#    # Calculate influence functions
#
#    # the solution which is converged properly can be found within 10 iterations, otherwise does not converge
#    t1 = time.time()
#    ihvp_cg_logreg = get_inverse_hvp_cg(net, net.loss, v_logreg, train_set,**{'damping':0.0, 'maxiter':10})
#    IF_cg_logreg = IF_val(net, ihvp_cg_logreg, train_set)
#    print('CG_logreg takes {} sec, and its max/min value {}'.format(time.time()-t1, [max(IF_cg_logreg),min(IF_cg_logreg)]))
#
#    np.save(save_path+'/if_cg_logreg.npy', IF_cg_logreg)
#
#    # otherwise, load
#    IF_cg_logreg = np.load(save_path+'/if_cg_logreg.npy')
#
#    # t1 = time.time()
#    # ihvp_ncg_logreg = get_inverse_hvp_ncg(net, net.loss, v_logreg, train_set,**{'damping':0.1, 'maxiter':3})
#    # IF_ncg_logreg = IF_val(net, ihvp_ncg_logreg, train_set)
#    # print('NCG_logreg takes {} sec, and its value {}'.format(time.time()-t1, IF_ncg_logreg))
#
#    # Restore weights
#    net.logits.restore('/Data/checkpts/noisy/model_fold_1_trainval_ratio_1.0.dnn')
#    
#    visualize_topk_samples(IF_cg_logreg, train_set, num_sample=5, save_path=save_path)
#
#
## In[147]:
#
# RELABELING

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

import time

def review(ratios, method):
    # ratios: the list of ratio which is the proportion of the data considered to be reviewed.  
    # reviewing is done by oracle. the label may or may not be changed. 
    #(i.e. if a single reviewed data has correct label, the label won't be changed, and vice versa)
    # method: the methodology of selecting data torch be reviewed. 
    # this can be 'random', 'influence', loss', 'entropy'
    t1 = time.time()

    # FIXME
    anno_dir = '/Data/emnist/balanced/original/annotation/'
    root_dir = '/Data/emnist/balanced/original/'
    #checkpt_dir = '/Data/github/interview/save/dropout_0.5_noaugmentation/model_fold_1_trainval_ratio_0.3.dnn'
    #checkpt_dir = '/Data/checkpts/noisy/model_fold_1_trainval_ratio_1.0.dnn'
    checkpt_dir = '/Data/checkpts/noisy/model_fold_1_trainval_ratio_0.3.dnn'

    with open(anno_dir + 'annotation1.json', 'r') as fid:
        annotation = json.load(fid)

    with open(anno_dir + 'annotation1_wp_0.3.json', 'r') as fid:
        noisy_annotation = json.load(fid)

    #image_list = list(noisy_annotation['images'].keys())
    #image_list = trainval_list
    image_list = list(np.load('./result/train_n_77228/trainval_list.npy'))
    num_image = len(image_list)

    ## sorting
    if method == 'random':
        # random policy
        image_list_random = image_list[:]
        random.shuffle(image_list_random)
        review_list = image_list_random
        #review_list_random = np.random.choice(image_list, int(num_image * ratio), replace=False)
        
    elif method == 'influence':
        # influence function
        save_path = './result/train_n_77228'
        IF_measure = np.load(save_path+'/if_cg_logreg.npy')
        argsort_abs = np.argsort(np.abs(IF_measure))[::-1]
        #review_list = image_list[argsort_abs]
        review_list = [image_list[idx] for idx in argsort_abs]
#         noisy_list = [noisy_annotation['images'][fname]['class'] for fname in review_list]
#         print(review_list[0:int(num_image * ratios[0])])
#         print(noisy_list[0:int(num_image * ratios[0])])

    elif method == 'influence-sum':
        # summation of influence function among several samples
        save_path = glob.glob('./result/*')
        IF_measures = []
        for pth in save_path:
            IF_measure = np.load(pth+'/if_cg_logreg.npy')
            IF_measures.append(np.abs(IF_measure))
        IF_measures = np.mean(IF_measures, axis=0)
        argsort_abs = np.argsort(IF_measures)[::-1]
        review_list = [image_list[idx] for idx in argsort_abs]
        
    else:
        # loss
        image_set = dataset.LazyDataset(root_dir, image_list, noisy_annotation)
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
            review_list = image_list_loss
        
        elif method == 'entropy':
            # entropy ascending policy
            idx_entropy = np.argsort(entropy)[::-1]
            image_list_entropy = [image_list[i] for i in idx_entropy]
            review_list = image_list_entropy

    ## correcting
    corrected_list = []
    for ratio in ratios:
        print(ratio)
        num_corrected = 0
        review_list_ratio = review_list[0:int(num_image * ratio)]
        print(len(review_list_ratio))
        for fname in review_list_ratio:
            correct_class = annotation['images'][fname]['class']
            noisy_class = noisy_annotation['images'][fname]['class']
            if noisy_class != correct_class:
                num_corrected += 1
        #corrected_list.append(num_corrected/int(0.3*len(image_list)))
        corrected_list.append(num_corrected)

    return corrected_list
    #return [cr/int(0.3*len(image_list)) for cr in corrected_list]

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

# main code

#x = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
#x = [0.2]
#x = [0.1, 0.5, 0.9]
x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

rnd = review(x, 'random')
ls = review(x, 'loss')
etp = review(x, 'entropy')
inf1 = review(x, 'influence')
inf2 = review(x, 'influence-sum')

print('random:', rnd)
print('loss:', ls)
print('entropy:', etp)
print('infleunce:', inf1)
print('influence_sum:', inf2)

# draw a graph
fig, ax = plt.subplots(1,1, figsize=(9,9))
_ = ax.plot(x, rnd, color='b', label='random')
_ = ax.plot(x, ls, color='g', label='loss')
_ = ax.plot(x, etp, color='r', label='entropy') 
_ = ax.plot(x, inf1, color='y', label='influence') 
_ = ax.plot(x, inf2, color='c', label='influence-sum') 
_ = ax.set_title('Recovery results')
_ = ax.set_ylabel('Ratio of corrected labeled: $Num_{corrected}/Num_{mislabeled}$')
_ = ax.set_xlabel('Ratio of reviewed data: $Num_{reviewed}/Num_{total}$')
_ = ax.set_xticks(x)
_ = plt.legend()
plt.savefig('./images/recovery_results.png', bbox_inches='tight')
plt.show()
#
#
## # tsne
## tsne.py 참고
#
## In[ ]:
#
#
#
