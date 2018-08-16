
# coding: utf-8

# # EMNIST DATASET
# 
# load network
# check IF
# compare btw TF code
# check retrain & IF
# sorting & relabeling

# ## IHVP
# 
# Algorithms for Inverse of Hessian Vector Product (CG & SE)

# In[1]:


import cntk as C
from cntk.device import try_set_default_device, gpu
try_set_default_device(gpu(0))

import numpy as np
import time

from torch.utils.data import DataLoader


# In[2]:


# Hessian Vector Product

def grad_inner_product(grad1, grad2):
    # inner product for dictionary-format gradients (output scalar value)
    
    val = 0
    
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
    
    assert type(x)==dict, "Input of HVP is wrong. this should be dictionary"
     
    w = y.parameters
    
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


# In[3]:


# def HVP_minibatch_val(model, y, v, dataloader):
#     # Calculate Hessian vector product w.r.t whole dataset
#     # model: neural network model (e.g. model)
#     # y: scalar function output of the neural network (e.g. model.loss)
#     # v: vector to be producted by inverse hessian (i.e.H^-1 v) (numeric dictionary, e.g. v_test)
#     # dataloader: training set dataloader
    
#     # hyperparameters
#     damping = 0.0 # convexity term; paper ref:0.01
    
#     hvp_batch = {ks: [] for ks in v.keys()}
    
#     for img, lb in dataloader:
#         img = img.numpy(); lb = lb.numpy()
#         x_feed = {model.X: img, model.y:lb}
#         hvp = HVP(y,x_feed,v)
#         # add hvp value
#         [hvp_batch[ks].append(hvp[ks]/img.shape[0]) for ks in hvp.keys()]
        
#     hvp_mean = {ks: np.mean(hvp_batch[ks], axis=0) + damping*v[ks] for ks in hvp_batch.keys()}
    
#     return hvp_mean

# # x: solution vector for conjugate gradient, whose shape is same as flattened gradient. NOT feed dict value

# def dic2vec(dic):
#     # convert a dictionary with matrix values to a 1D vector
#     # e.g. gradient of network -> 1D vector
#     vec = np.concatenate([val.reshape(-1) for val in dic.values()])
    
#     return vec

# def vec2dic(vec, fmt):
#     # convert a 1D vector to a dictionary of format fmt
#     # fmt = {key: val.shape for (key,val) in dict}
#     fmt_idx = [np.prod(val) for val in fmt.values()]
#     #lambda ls, idx: [ls[sum(idx[:i]):sum(idx[:i+1])] for i in range(len(idx))]
#     vec_split = [vec[sum(fmt_idx[:i]):sum(fmt_idx[:i+1])] for i in range(len(fmt_idx))]
#     dic = {key: vec_split[i].reshape(shape) for (i,(key,shape)) in enumerate(fmt.items())}

#     return dic


# def get_fmin_loss_fn(model, y, v, dataloader):
    
#     def get_fmin_loss(x):
#         x_dic = vec2dic(x, {key: val.shape for (key, val) in v.items()})
#         hvp_val = HVP_minibatch_val(model, y, x_dic, dataloader)
        
#         return 0.5 * grad_inner_product(hvp_val, x_dic) - grad_inner_product(v, x_dic)
    
#     return get_fmin_loss

# def get_fmin_grad_fn(model, y, v, dataloader):
    
#     def get_fmin_grad(x):
#         # x: 1D vector
#         x_dic = vec2dic(x, {key: val.shape for (key, val) in v.items()})
#         hvp_val = HVP_minibatch_val(model, y, x_dic, dataloader)
#         hvp_flat = dic2vec(hvp_val)
#         v_flat = dic2vec(v)
        
#         return hvp_flat - v_flat
    
#     return get_fmin_grad

# def get_fmin_hvp_fn(model, y, v, dataloader):

#     def get_fmin_hvp(x, p):
#         p_dic = vec2dic(p, {key: val.shape for (key, val) in v.items()})
#         hvp_val = HVP_minibatch_val(model, y, p_dic, dataloader)
#         hvp_flat = dic2vec(hvp_val)

#         return hvp_flat
    
#     return get_fmin_hvp

# def get_inverse_hvp_cg(model, y, v, data_set):
#     # return x, which is the solution of QP, whose value is H^-1 v

#     # hyperparameters
#     batch_size = 50
    
#     dataloader = DataLoader(data_set, batch_size, shuffle=True, num_workers=6)
    
#     fmin_loss_fn = get_fmin_loss_fn(model, y, v, dataloader)
#     fmin_grad_fn = get_fmin_grad_fn(model, y, v, dataloader)
#     fmin_hvp_fn = get_fmin_hvp_fn(model, y, v, dataloader)
    
#     fmin_results = fmin_ncg(\
#             f = fmin_loss_fn, x0 = dic2vec(v), fprime = fmin_grad_fn,\
#             fhess_p = fmin_hvp_fn, avextol = 1e-8, maxiter = 1e2)
    
#     #return fmin_results
#     return vec2dic(fmin_results, {key: val.shape for (key, val) in v.items()})


# ## 속도 문제
# 
# - jupyter notebook에서 돌리면 전체적으로 더 오래 걸림
# - 때문에 python code로 convert해서 돌릴 것을 추천
# 
# ## 무한 loop 문제 (fmin_ncg & fmin_cg)
# 
# - 원저자의 구현코드에서는 일반적인 conjugate gradient를 사용한 fmin_cg 대신 newton conjugate gradient를 사용한 fmin_ncg를 통해 구현함.
# - ncg의 경우 iteration이 크게 2번 일어나게 되는데 가장 바깥 loop는 xk를 업데이트 하는 것인 반면 안쪽 loop는 얼마나 update를 할 지 newton 방법으로 찾는 것.
#     - 이 때 gradient 대신 newton method의 해를 찾기 때문에 (scipy에서 제공하는) 예전 코드에서는 while을 사용해서 error의 크기만을 가지고 terminate condition보다 작으면 벗어나게 만들어서 무한 loop가 걸릴 가능성이 있고, 최근 코드에서는 추가적으로 cg_maxiter를 정해 내부 loop가 20 * len(x0)번 이상 돌아가지 않게 만들었으나 우리의 경우 len(x0)가 80000이 넘어가기 때문에 사실상 며칠을 돌려도 돌아가지 않음. (몇 십분 돌렸을 때 겨우 400번 돌았음.)
#     - cg_maxiter는 함수 내부 변수이기 때문에 우리가 조정할 수 없음...
#     - 대신 iteration 1번만 돌아도 왠만하면 잘 동작하는 해를 찾을 수 있음. (i.e. maxiter hyperparameter에 robust해짐)
# - cg의 경우에는 hyperparameter에 좀 더 의존적일 수 있겠지만 iteration loop 하나만 있어서 수렴을 했던 하지 않았던 중간에 멈출 수 있음.
# 

# In[4]:


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


# In[5]:


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


# In[6]:


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
    
    params = y.parameters
    
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


# In[7]:


# toy example for inverse HVP (CG and SE)

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
ihvp_se = get_inverse_hvp_se(net, net.loss, v_feed, train_set, **{'damping': 0.0}); t4 = time.time()
print('inverse hvp_ncg', ihvp_ncg, '\ntime: ', t2-t1)
print('inverse hvp_cg', ihvp_cg, '\ntime: ', t3-t2 )
print('inverse hvp_se', ihvp_se, '\ntime: ', t4-t3)

# print('inverse hvp_ncg', get_inverse_hvp_ncg(net, net.loss, v_feed, train_set, **{'damping': 0.1}))
# print('inverse hvp_cg', get_inverse_hvp_cg(net, net.loss, v_feed, train_set, **{'damping': 0.1}))
# print('inverse hvp_se', get_inverse_hvp_se(net, net.loss, v_feed, train_set, **{'scale':10, 'damping':0.1}))


# ## EMNIST dataset

# In[8]:


import os, sys
sys.path.append('../refer/boot_strapping')
import json

from datasets import dataset28 as dataset

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from torch.utils.data import DataLoader


# In[9]:


def show_image_from_data(img):
    # show image from dataset
    # img: (C,W,H) numpy array
    img_show = np.squeeze(np.transpose(img, [1,2,0]))
    imshow(img_show)
    plt.show()


## In[10]:
#
#
## emnist dataset
#root_dir = '/Data/emnist/balanced/original'
#
## sample size
#trainval_list, anno_dict = dataset.read_data_subset(root_dir, mode='train1', sample_size=100)
#test_list, _ = dataset.read_data_subset(root_dir, mode='validation1', sample_size=100)
#
#with open('/Data/emnist/balanced/original/annotation/annotation1_wp_0.3.json','r') as fid:
#    noisy_anno_dict = json.load(fid)
#
#train_set = dataset.LazyDataset(root_dir, trainval_list, noisy_anno_dict)
#test_set = dataset.LazyDataset(root_dir, test_list, anno_dict)
#
## emnist dataset: SANITY CHECK
#print(len(test_set), type(test_set))
#print(len(test_list))
#
#
## In[11]:
#
#
## emnist network
#from models.nn import VGG as ConvNet
#
#hp_d = dict() # hyperparameters for a network
#net = ConvNet(train_set.__getitem__(0)[0].shape, len(anno_dict['classes']), **hp_d)
#net.logits.restore('/Data/checkpts/noisy/model_fold_1_trainval_ratio_1.0.dnn')
#
## emnist network: SANITY CHECK
#start_time = time.time()
#ys, y_preds, test_score, confusion_matrix = net.predict(test_set, **hp_d)
#total_time = time.time() - start_time
#
#print('Test error rate: {}'.format(test_score))
#print('Total tack time(sec): {}'.format(total_time))
#print('Tact time per image(sec): {}'.format(total_time / len(test_list)))
#print('Confusion matrix: \n{}'.format(confusion_matrix))
#
#
## In[12]:
#
#
## re-initialize network parameters since these can be NaN due to previous execution
#net.logits.restore('/Data/checkpts/noisy/model_fold_1_trainval_ratio_1.0.dnn')
#params = net.logits.parameters
#
#img_test, lb_test = test_set.__getitem__(1)
#show_image_from_data(img_test)
#v_test = net.loss.grad({net.X:img_test, net.y:lb_test}, wrt=params)
#
#print('ground truth label: ', anno_dict['classes'][str(np.argmax(lb_test))])
#print('network prediction: ', anno_dict['classes'][str(np.argmax(net.logits.eval({net.X:img_test})))])
#
#img, lb = train_set.__getitem__(1)
#show_image_from_data(img)
#v_train = net.loss.grad({net.X:img, net.y:lb}, wrt=params)
#
## influence functions
#
#print(grad_inner_product(v_train, v_test))
#
## the solution which is converged properly can be found within 10 iterations, otherwise does not converge
#t1 = time.time()
#ihvp_cg = get_inverse_hvp_cg(net, net.loss, v_test, train_set,**{'damping':0.1, 'maxiter':10})
#IF_cg = grad_inner_product(ihvp_cg, v_train)/1000 # loss difference = -1/num_sample * influence function
#print('CG takes {} sec, and its value {}'.format(time.time()-t1, IF_cg))
#
## t1 = time.time()
## ihvp_ncg = get_inverse_hvp_ncg(net, net.loss, v_test, train_set,**{'damping':0.1, 'maxiter':3})
## IF_ncg = grad_inner_product(ihvp_ncg, v_train)/1000 # loss difference = -1/num_sample * influence function
## print('NCG takes {} sec, and its value {}'.format(time.time()-t1, IF_cg))
#
#t1 = time.time()
#ihvp_se = get_inverse_hvp_se(net, net.loss, v_test, train_set,**{'scale':1e5, 'damping':0.1, 'verbose':False})
#IF_se = grad_inner_product(ihvp_se, v_train)/1000 # loss difference = -1/num_sample * influence function
#print('SE takes {} sec'.format(time.time()-t1))

#print(IF_cg, IF_se)


# In[26]:


def IF_val(net, ihvp, data_set):
    # Calculate influence function w.r.t ihvp and data_set
    # This should be done in sample-wise, since the gradient operation will sum up over whole feed-dicted data
    
    # ihvp: inverse hessian vector product (dictionary)
    # data_set: data_set to be feed to the gradient operation (dataset)
    IF_list = []
    
    #params = net.logits.parameters
    params = ihvp.keys()
    
    dataloader = DataLoader(data_set, 1, shuffle=False, num_workers=6)
    
    for img, lb in dataloader:
        img = img.numpy(); lb = lb.numpy()
        gd = net.loss.grad({net.X:img, net.y:lb}, wrt=params)
        IF = grad_inner_product(ihvp, gd) / len(dataloader)
        IF_list.append(IF)
        
    return IF_list


# In[17]:


def visualize_topk_samples(measure, num_sample=5):
    argsort = np.argsort(measure) 
    topk = argsort[-1:-num_sample-1:-1]
    botk = argsort[0:num_sample]

    print('\n## SHOW {}-MOST DISADVANTAGEOUS EXAMPLES ##\n'.format(num_sample))
    for idx in topk:
        img, lb = train_set.__getitem__(idx)
        show_image_from_data(img)
        print('training set label (noisy): ', anno_dict['classes'][str(np.argmax(lb))])

    print('\n## SHOW {}-MOST ADVANTAGEOUS EXAMPLES ##\n'.format(num_sample))
    for idx in botk:
        img, lb = train_set.__getitem__(idx)
        show_image_from_data(img)
        print('training set label (noisy): ', anno_dict['classes'][str(np.argmax(lb))])

    argsort_abs = np.argsort(np.abs(measure))
    topk_abs = argsort_abs[-1:-num_sample-1:-1]
    botk_abs = argsort_abs[0:num_sample]

    print('\n## SHOW {}-MOST INFLUENTIAL EXAMPLES ##\n'.format(num_sample))
    for idx in topk_abs:
        img, lb = train_set.__getitem__(idx)
        show_image_from_data(img)
        print('training set label (noisy): ', anno_dict['classes'][str(np.argmax(lb))])

    print('\n## SHOW {}-MOST NEGLIGIBLE EXAMPLES ##\n'.format(num_sample))
    for idx in botk_abs:
        img, lb = train_set.__getitem__(idx)
        show_image_from_data(img)
        print('training set label (noisy): ', anno_dict['classes'][str(np.argmax(lb))])
        
    return 0


# In[19]:


# sort training examples by IF measure and check result

# 1: v_test^T * v_train
# 2: v_cg^T * v_train
# 3: v_se^T * v_train

# emnist dataset
root_dir = '/Data/emnist/balanced/original'

# sample size
trainval_list, anno_dict = dataset.read_data_subset(root_dir, mode='train1', sample_size=100)
test_list, _ = dataset.read_data_subset(root_dir, mode='validation1', sample_size=100)

with open('/Data/emnist/balanced/original/annotation/annotation1_wp_0.3.json','r') as fid:
    noisy_anno_dict = json.load(fid)

train_set = dataset.LazyDataset(root_dir, trainval_list, noisy_anno_dict)
test_set = dataset.LazyDataset(root_dir, test_list, anno_dict)

# emnist network
from models.nn import VGG as ConvNet

hp_d = dict() # hyperparameters for a network
net = ConvNet(train_set.__getitem__(0)[0].shape, len(anno_dict['classes']), **hp_d)
net.logits.restore('/Data/checkpts/noisy/model_fold_1_trainval_ratio_1.0.dnn')
params = net.logits.parameters

img_test, lb_test = test_set.__getitem__(1)
show_image_from_data(img_test)
v_test = net.loss.grad({net.X:img_test, net.y:lb_test}, wrt=params)

print('ground truth label: ', anno_dict['classes'][str(np.argmax(lb_test))])
print('network prediction: ', anno_dict['classes'][str(np.argmax(net.logits.eval({net.X:img_test})))])

## influence functions
#
## the solution which is converged properly can be found within 10 iterations, otherwise does not converge
#t1 = time.time()
#ihvp_cg = get_inverse_hvp_cg(net, net.loss, v_test, train_set,**{'damping':0.01, 'maxiter':10})
#IF_cg = IF_val(net, ihvp_cg, train_set)
#print('CG takes {} sec, and its value {}'.format(time.time()-t1, IF_cg))
#
#IF_vec = IF_val(net, v_test, train_set)
#
## t1 = time.time()
## ihvp_ncg = get_inverse_hvp_ncg(net, net.loss, v_test, train_set,**{'damping':0.1, 'maxiter':3})
## IF_ncg = grad_inner_product(ihvp_ncg, v_train)/1000 # loss difference = -1/num_sample * influence function
## print('NCG takes {} sec, and its value {}'.format(time.time()-t1, IF_cg))
#
## t1 = time.time()
## ihvp_se = get_inverse_hvp_se(net, net.loss, v_test, train_set,**{'scale':1e5, 'damping':0.1, 'verbose':False})
## IF_cg = IF_val(net, ihvp_se, train_set)
## print('SE takes {} sec'.format(time.time()-t1))
#
##print(IF_cg, IF_se)
#
#
## In[20]:
#
#
#visualize_topk_samples(IF_cg, num_sample=5)
## visualize_topk_samples(IF_vec, num_sample=5)
## visualize_topk_samples(IF_se, num_sample=5)


# In[22]:





# In[28]:
from ipdb import set_trace

set_trace()

img_test, lb_test = test_set.__getitem__(1)
show_image_from_data(img_test)

params = net.loss.parameters
p_ftex = net.d['dense1'].parameters
p_logreg = tuple(set(params) - set(p_ftex))
print(p_logreg)
v_logreg = net.loss.grad({net.X:img_test, net.y:lb_test}, wrt=p_logreg)

print('ground truth label: ', anno_dict['classes'][str(np.argmax(lb_test))])
print('network prediction: ', anno_dict['classes'][str(np.argmax(net.logits.eval({net.X:img_test})))])

# influence functions

# the solution which is converged properly can be found within 10 iterations, otherwise does not converge
t1 = time.time()
ihvp_cg_logreg = get_inverse_hvp_cg(net, net.loss, v_logreg, train_set,**{'damping':0.01, 'maxiter':10})
IF_cg_logreg = IF_val(net, ihvp_cg_logreg, train_set)
print('CG_logreg takes {} sec, and its value {}'.format(time.time()-t1, IF_cg_logreg))

visualize_topk_samples(IF_cg_logreg, num_sample=5)


# In[25]:


# sorting with IF

# 1: v_test^T * v_train
# 2: v_cg^T * v_train
# 3: v_se^T * v_train


# In[ ]:


# freezing network


# In[ ]:


# logistic regression: retraining vs influence function


# 후자 cg 택 (후자는 hyperparameter 연동시킴)
# 
# 수정: cg callback 추가
# 
# 발산 시 강제 종료하는 법
# 
# train sample size 
