
# coding: utf-8

# ## HVP
# 
# implement HVP using cntk
# refer: https://cntk.ai/pythondocs/cntk.ops.functions.html#cntk.ops.functions.Function.forward

# In[44]:


import cntk as C
from cntk.device import try_set_default_device, gpu
try_set_default_device(gpu(0))

import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from torch.utils.data import DataLoader

from ipdb import set_trace

# Calculate HVP using $\frac{g(x+rv)-g(x-rv)}{2r}$.

# In[2]:


def weight_update(w, v, r):
    # w: weights of neural network (tuple)
    # v: value for delta w (dictionary, e.g., gradient value)
    # r: hyperparameter for a gradient (scalar)
    
    for p in w:
        p.value += r * v[p]


# In[3]:


def HVP(y, x, v):
    # y: function to be differentiated (function, e.g. neural network logit)
    # w: variables to differentiate (numeric, e.g. neural network weight)
    # x: feed_dict value for the network y (numpy array, e.g., image)
    # v: vector to be producted (by Hessian) (numeric dictionary, e.g., g(z_test))
    
    # hyperparameter r
    r = 1e-4
    
    #feed = y.inputs[-1] # input of the neural network
    feed = list(filter(lambda x: x.is_input, y.inputs))[0]
    
    x = np.asarray(x, dtype=feed.dtype)
    assert(type(v)==dict)
    
    w = y.parameters
    
    # gradient for plus
    weight_update(w, v, +r)
    g_plus = y.grad({feed:x}, wrt=params)
    weight_update(w, v, -r)
    
    # gradient for minus
    weight_update(w, v, -r)
    g_minus = y.grad({feed:x}, wrt=params)
    weight_update(w, v, +r)
    
    # hvp = (g({feed:x+np.dot(r,v_stop)}, wrt=params) - g({feed:x-np.dot(r,v_stop)}, wrt=params))/(2*r) # dict implemented
    
    hvp = {ks: (g_plus[ks] - g_minus[ks])/(2*r) for ks in g_plus.keys()}
       
    return hvp


# In[17]:


def grad_inner_product(grad1, grad2):
    # inner product for dictionary-format gradietns
    
    val = 0
    
    for ks in grad1.keys():
        #C.inner(grad1[ks], grad2[ks])
        val += np.sum(np.multiply(grad1[ks],grad2[ks]))
        #val += np.dot(grad1[ks], grad2[ks])
        
    return val


# In[32]:


# toy example

x = C.input_variable(shape=(1,))
h = C.layers.Dense(1, activation=None, init=C.uniform(1), bias=False)(x)
y = C.layers.Dense(1, activation=None, init=C.uniform(1), bias=False)(h)

y_plus = y.clone('clone')
y_minus = y.clone('clone')
x_feed = [[1.]]
params = y.parameters
v_feed = {p: np.ones_like(p.value) for p in params}

HVP(y, x_feed, v_feed)

# output should be 1, 1


# 원래 답은 1.0, 1.0이 나와야 함. 이런 차이는 hyperparameter r과 관계가 있을 것. r이 작이질 수록 오차도 적어질 것으로 예상되지만, 지나치게 r이 적게 되면 precision number보다 적은 값이 나와서 문제가 생길 수 있음. 특히 gradient 값이 작은 부분에선 큰 문제가 됨.

# In[5]:


# emnist dataset
import os, sys
sys.path.append('../refer/boot_strapping')
import json

from datasets import dataset28 as dataset

root_dir = '/Data/emnist/balanced/original'

# sample sized
trainval_list, anno_dict = dataset.read_data_subset(root_dir, mode='train1', sample_size=1000)
test_list, _ = dataset.read_data_subset(root_dir, mode='validation1', sample_size=1000)

with open('/Data/emnist/balanced/original/annotation/annotation1_wp_0.3.json','r') as fid:
    noisy_anno_dict = json.load(fid)

train_set = dataset.LazyDataset(root_dir, trainval_list, noisy_anno_dict)
test_set = dataset.LazyDataset(root_dir, test_list, anno_dict)

# emnist dataset: SANITY CHECK
print(len(test_set), type(test_set))
print(len(test_list))


# In[6]:


# emnist network
from models.nn import VGG as ConvNet

hp_d = dict() # hyperparameters for a network
model = ConvNet(train_set.__getitem__(0)[0].shape, len(anno_dict['classes']), **hp_d)
model.logits.restore('/Data/checkpts/noisy/model_fold_1_trainval_ratio_1.0.dnn')

# emnist network: SANITY CHECK
start_time = time.time()
ys, y_preds, test_score, confusion_matrix = model.predict(test_set, **hp_d)
total_time = time.time() - start_time

print('Test error rate: {}'.format(test_score))
print('Total tack time(sec): {}'.format(total_time))
print('Tact time per image(sec): {}'.format(total_time / len(test_list)))
print('Confusion matrix: \n{}'.format(confusion_matrix))


# In[7]:


# neural network architecture

# neural network load

# calculate HVP for an example

# check tack time


# In[29]:


def show_image_from_data(img):
    # show image from dataset
    # img: (C,W,H) numpy array
    img_show = np.squeeze(np.transpose(img, [1,2,0]))
    imshow(img_show)
    plt.show()


# In[31]:


# hessian vector product w.r.t. test image

params = model.logits.parameters
img_test, lb_test = test_set.__getitem__(1)
#img2 = np.squeeze(np.transpose(img_test, [1,2,0]))
#imshow(img2)
#plt.show()
show_image_from_data(img_test)
print('ground truth label: ', anno_dict['classes'][str(np.argmax(lb_test))])
print('network prediction: ', anno_dict['classes'][str(np.argmax(model.logits.eval({model.X:img_test})))])

v_test = model.logits.grad({model.X:img_test}, wrt=params)

# HVP(y,x,v)
img = train_set.__getitem__(1)[0]
img.shape
show_image_from_data(img)

hvp = HVP(model.logits, img, v_test)

#grad_inner_product(v_test, v_test)
#grad_inner_product(hvp, hvp)
grad_inner_product(hvp, v_test)


# In[57]:


# stochastic estimation
def IHVP(y, v, data_set): # data, network, etc. as we did in GradCAM
    # Calculate inverse hessian vector product over the training set
    # y: output of the neural network (e.g. model.logits)
    # v: vector to be producted by inverse hessian (i.e.H^-1 v) (e.g. v_test)
    # data_set: training set to be summed in Hessian
    
    # hyperparameters (hp_d)
    recursion_depth = 2
    scale = 10
    damping = 0.0
    batch_size = 1
    num_samples = 1 # the number of samples(:stochatic estimation of IF) to be averaged
    
    dataloader = DataLoader(data_set, batch_size, shuffle=True, num_workers=6)
    
    inv_hvps = []
    
    params = y.parameters
    
    for i in range(num_samples):
        # obtain num_samples inverse hvps
        cur_estimate = v
        
        for depth in range(recursion_depth):
            # epoch-scale recursion depth
            
            for X, _ in dataloader:
                set_trace()
                print(cur_estimate[list(cur_estimate.keys())[2]][0])
                # X = X.numpy()
                hvp = HVP(y,X,cur_estimate)
                # cur_estimate = v + (1-damping)*cur_estimate + 1/scale*(hvp/batch_size)
                #cur_estimate = {ks: a + (1-damping)*b + 1/scale*c/batch_size for (ks, a,b,c) in zip(cur_estimate.keys(), cur_estimate, v, hvp)}
                cur_estimate = {ks: cur_estimate[ks] + (1-damping)*v[ks] + 1/scale*hvp[ks]/batch_size for ks in cur_estimate.keys()}
            print("Recursion depth: {}, norm: {} \n".format(depth, grad_inner_product(cur_estimate,cur_estimate)))
        
        #inv_hvp = {ks: 1/scale*a/batch_size for (ks,a) in zip(cur_estimate.keys(), cur_estimate)}
        inv_hvp = {ks: 1/scale*cur_estimate[ks]/batch_size for ks in cur_estimate.keys()}
        inv_hvps.append(inv_hvp)
    
    return inv_hvps
    
    # average among samples
    #ihvp = average(inv_hvps)
    #return ihvp


# In[56]:


set_trace()
IHVP(model.logits, v_test, train_set)


# In[39]:


gd1 = y.grad({x:np.array([[1.]])}, wrt=y.parameters)
gd2 = y.grad({x:np.array([[1.],[1.]])}, wrt=y.parameters)
print(gd1,gd2)


# cf) gradient with minibatch whose size is greater than 1
# 여러 sample에 대해서 gradient을 구하는 경우 average 대신 summation된 값을 내보냄.
# 따라서 hvp를 sample 수에 대해서 normalize해줘야 함.
