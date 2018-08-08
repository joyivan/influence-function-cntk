import cntk as C
import time
import matplotlib.pyplot as plt
import numpy as np

from ipdb import set_trace

def weight_update(w, v, r):
    # w: weights of neural network (tuple)
    # v: value for delta w (dictionary, e.g., gradient value)
    # r: hyperparameter for a gradient (scalar)
                        
    for p in w:
        p.value += r * v[p]
    
    return

def HVP(y, y_plus, y_minus, x, v):
    # y: function to be differentiated (function, e.g. neural network logit)
    # y_plus & y_minus: clones of y used to calculate Hessian
    # w: variables to differentiate (numeric, e.g. neural network weight)
    # x: feed_dict value for the network y (numpy array, e.g., image)
    # v: vector to be producted (by Hessian) (numeric dictionary, e.g., g(z_test))
    
    # hyperparameter r
    r = 1e-4

    feed = y.inputs[-1] # input of the neural network

    x = np.asarray(x, dtype=feed.dtype)
    assert(type(v)==dict)

    w = y.parameters
    w_plus = y_plus.parameters
    w_minus = y_minus.parameters

    # update paramter
    weight_update(w_plus, v, +r)
    weight_update(w_minus, v, -r)

    # hvp = (g({feed:x+np.dot(r,v_stop)}, wrt=params) -
    # g({feed:x-np.dot(r,v_stop)}, wrt=params))/(2*r) # dict implemented
    g_plus = y_plus.grad({feed:x}, wrt=params)
    g_minus = y_minus.grad({feed:x}, wrt=params)
    hvp = {ks: (g_plus[ks] - g_minus[ks])/(2*r) for ks in g_plus.keys()}

    # recover parameter
    weight_update(w_plus, v, -r)
    weight_update(w_minus, v, +r)

    return hvp

set_trace()

# toy example
x = C.input_variable(shape=(1,))
h = C.layers.Dense(1, activation=None, init=C.uniform(1), bias=False)(x)
y = C.layers.Dense(1, activation=None, init=C.uniform(1), bias=False)(h)

y_plus = y.clone('clone')
y_minus = y.clone('clone')
x_feed = [[1.]]
params = y.parameters
v_feed = {p: np.ones_like(p.value) for p in params}
# check v_feed

HVP(y, y_plus, y_minus, x_feed, v_feed)

set_trace()

print('done')
