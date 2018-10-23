import cntk as C
from cntk.device import try_set_default_device, gpu
try_set_default_device(gpu(0))
import numpy as np
import time

from hvp import HVP
from influence import get_inverse_hvp_cg, get_inverse_hvp_se

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
set_trace()

print('######## damping = 0.0, desired solution: [1.25, -0.08] ########'); t1 = time.time()
ihvp_ncg = get_inverse_hvp_cg(net, net.loss, v_feed, train_set, method='Newton', **{'damping': 0.0}); t2 = time.time()
ihvp_cg = get_inverse_hvp_cg(net, net.loss, v_feed, train_set, **{'damping': 0.0}); t3 = time.time()
ihvp_se = get_inverse_hvp_se(net, net.loss, v_feed, train_set, **{'damping': 0.0, 'recursion_depth': 100}); t4 = time.time()
print('inverse hvp_ncg', ihvp_ncg, '\ntime: ', t2-t1)
print('inverse hvp_cg', ihvp_cg, '\ntime: ', t3-t2 )
print('inverse hvp_se', ihvp_se, '\ntime: ', t4-t3)

# check divergence when scale is low, and check the weight parameter recovery
#ihvp_se = get_inverse_hvp_se(net, net.loss, v_feed, train_set, **{'scale': 1, 'num_samples': 3, 'damping': 0.0, 'recursion_depth': 100}); t4 = time.time()

# print('inverse hvp_ncg', get_inverse_hvp_cg(net, net.loss, v_feed, train_set, method='Newton', **{'damping': 0.1}))
# print('inverse hvp_cg', get_inverse_hvp_cg(net, net.loss, v_feed, train_set, **{'damping': 0.1}))
# print('inverse hvp_se', get_inverse_hvp_se(net, net.loss, v_feed, train_set, **{'scale':10, 'damping':0.1}))
