
# coding: utf-8

# ## HVP
# 
# implement HVP using cntk
# refer: https://cntk.ai/pythondocs/cntk.ops.functions.html#cntk.ops.functions.Function.forward

# In[1]:


import cntk as C
from cntk.device import try_set_default_device, gpu
try_set_default_device(gpu(0))

import numpy as np
import time

from torch.utils.data import DataLoader

## Conjugate gradient

# In[171]:


def grad_inner_product(grad1, grad2):
    # inner product for dictionary-format gradients (output scalar value)
    
    val = 0
    
    for ks in grad1.keys():
        #C.inner(grad1[ks], grad2[ks])
        val += np.sum(np.multiply(grad1[ks],grad2[ks]))
        #val += np.dot(grad1[ks], grad2[ks])
        
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
    g_plus = y.grad(x, wrt=params)
  
    # gradient for minus
    weight_update(w, v, -2*r)
    g_minus = y.grad(x, wrt=params)
    
    # weight reconstruction
    weight_update(w, v, +r)
    
    hvp = {ks: (g_plus[ks] - g_minus[ks])/(2*r) for ks in g_plus.keys()}
       
    return hvp

def HVP_minibatch_val(model, y, v, data_set):
    # Calculate Hessian vector product w.r.t whole dataset
    # model: neural network model (e.g. model)
    # y: scalar function output of the neural network (e.g. model.loss)
    # v: vector to be producted by inverse hessian (i.e.H^-1 v) (numeric dictionary, e.g. v_test)
    # data_set: training set to be summed in Hessian
    
    # hyperparameters
    damping = 0.0 # convexity term; paper ref:0.01
    batch_size = 1
    
    hvp_batch = {ks: [] for ks in v.keys()}
    
    dataloader = DataLoader(data_set, batch_size, shuffle=True, num_workers=6)
    
    for img, lb in dataloader:
        img = img.numpy(); lb = lb.numpy()
        x_feed = {model.X: img, model.y:lb}
        hvp = HVP(y,x_feed,v)
        # add hvp value
        [hvp_batch[ks].append(hvp[ks]/img.shape[0]) for ks in hvp.keys()]
        
    hvp_mean = {ks: np.mean(hvp_batch[ks], axis=0) + damping*v[ks] for ks in hvp_batch.keys()}
    
    return hvp_mean

from scipy.optimize import fmin_ncg

# 정리정리!! 함수 인풋도 정리!
# x: solution vector for conjugate gradient, whose shape is same as flattened gradient. NOT feed dict value

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
    dic = {key: vec[i].reshape(shape) for (i,(key,shape)) in enumerate(fmt.items())}

    return dic


def get_fmin_loss_fn(model, y, data_set, v):
    
    def get_fmin_loss(x):
        x_dic = vec2dic(x, {key: val.shape for (key, val) in v.items()})
        hvp_val = HVP_minibatch_val(model, y, x_dic, data_set)
        
        return 0.5 * grad_inner_product(hvp_val, x_dic) - grad_inner_product(v, x_dic)
    
    return get_fmin_loss

def get_fmin_grad_fn(model, y, data_set, v):
    
    def get_fmin_grad(x):
        # x: 1D vector
        x_dic = vec2dic(x, {key: val.shape for (key, val) in v.items()})
        hvp_val = HVP_minibatch_val(model, y, x_dic, data_set)
        hvp_flat = dic2vec(hvp_val)
        v_flat = dic2vec(v)
        
        return hvp_flat - v_flat
    
    return get_fmin_grad

def get_fmin_hvp_fn(model, y, data_set, v):

    def get_fmin_hvp(x, p):
        p_dic = vec2dic(p, {key: val.shape for (key, val) in v.items()})
        hvp_val = HVP_minibatch_val(model, y, p_dic, data_set)
        hvp_flat = dic2vec(hvp_val)

        return hvp_flat
    
    return get_fmin_hvp

def get_inverse_hvp_cg(model, y, data_set, v):
    # return x, which is the solution whose value is H^-1 v
    
    fmin_loss_fn = get_fmin_loss_fn(model, y, data_set, v)
    fmin_grad_fn = get_fmin_grad_fn(model, y, data_set, v)
    fmin_hvp_fn = get_fmin_hvp_fn(model, y, data_set, v)
    
    fmin_results = fmin_ncg(\
            f = fmin_loss_fn,\
            x0 = dic2vec(v),\
            fprime = fmin_grad_fn,\
            fhess_p = fmin_hvp_fn,\
            avextol = 1e-8,\
            maxiter = 1e2)
    
    return fmin_results
    #return vec2dic(fmin_results, {key: val.shape for (key, val) in v.items()}))


# In[172]:


# toy example for IHVP: 1D example
class SimpleNet(object):
    def __init__(self):
        self.X = C.input_variable(shape=(1,))
        self.h = C.layers.Dense(1, activation=None, init=C.uniform(1), bias=False)(self.X)
        self.pred = C.layers.Dense(1, activation=None, init=C.uniform(1), bias=False)(self.h)
        self.y = C.input_variable(shape=(1,))
        #self.loss = C.reduce_l2(self.pred-self.y)
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
    
from ipdb import set_trace

set_trace()
      
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

print('hvp_batch', HVP_minibatch_val(net, net.loss, v_feed, train_set))

print('inverse hvp', get_inverse_hvp_cg(net, net.loss, train_set, v_feed))


## In[ ]:
#
#
#def HVP(y, x, v):
#    # Calculate Hessian vector product 
#    # y: scalar function to be differentiated (function, e.g. cross entropy loss)
#    # x: feed_dict value for the network (dictionary, e.g. {model.X: image_batch, model.y: label_batch})
#    # v: vector to be producted (by Hessian) (numeric dictionary, e.g., g(z_test))
#    ## w: variables to differentiate (numeric, e.g. neural network weight)
#    
#    # hyperparameter r
#    r = 1e-2
#    
#    assert type(x)==dict, "Input of HVP is wrong. this should be dictionary"
#     
#    w = y.parameters
#    
#    # gradient for plus
#    weight_update(w, v, +r)
#    g_plus = y.grad(x, wrt=params)
#  
#    # gradient for minus
#    weight_update(w, v, -2*r)
#    g_minus = y.grad(x, wrt=params)
#    
#    # weight reconstruction
#    weight_update(w, v, +r)
#    
#    hvp = {ks: (g_plus[ks] - g_minus[ks])/(2*r) for ks in g_plus.keys()}
#       
#    return hvp
#
# stochastic estimation
def IHVP(model, y, v, data_set, verbose=False): # data, network, etc. as we did in GradCAM
    # Calculate inverse hessian vector product over the training set
    # model: neural network model (e.g. model)
    # y: scalar function output of the neural network (e.g. model.loss)
    # v: vector to be producted by inverse hessian (i.e.H^-1 v) (e.g. v_test)
    # data_set: training set to be summed in Hessian
    
    # hyperparameters (hp_d)
    recursion_depth = 10
    scale = 1
    damping = 0.0 # paper ref:0.01
    batch_size = 1
    num_samples = 5 # the number of samples(:stochatic estimation of IF) to be averaged
    
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
                cur_estimate = {ks: v[ks] + (1-damping)*cur_estimate[ks] - (1/scale)*hvp[ks]/batch_size for ks in cur_estimate.keys()}
                if verbose:
                    print('#w: \n', list(map(lambda x: x.value, params)), '\n#hvp: \n', hvp, '\n#ihvp: \n', cur_estimate)
            print("Recursion depth: {}, norm: {}, time: {} \n".format(depth, np.sqrt(grad_inner_product(cur_estimate,cur_estimate)),time.time()-t1))
        
        inv_hvp = {ks: (1/scale)*cur_estimate[ks] for ks in cur_estimate.keys()}
        inv_hvps.append(inv_hvp)
    
    return inv_hvps

ihvp = IHVP(net, net.loss, v_feed, train_set, verbose=True)
print(ihvp)
#
## In[13]:
#
#
#gd1 = y.grad({x:np.array([[1.]])}, wrt=y.parameters)
#gd2 = y.grad({x:np.array([[1.],[1.]])}, wrt=y.parameters)
#print(gd1,gd2)
#
#
## remark) gradient with minibatch whose size is greater than 1
## 여러 sample에 대해서 gradient을 구하는 경우 average 대신 summation된 값을 내보냄.
## 따라서 hvp를 sample 수에 대해서 normalize해줘야 함.
