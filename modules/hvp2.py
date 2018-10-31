# Hessian Vector Product (Using numerical differentiation)
import numpy as np

# FIXME
import cntk as C
from ipdb import set_trace

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
        #p.value += r * v[p]
        C.assign(p, p.value+r*v[p])
    return 0

def HVP(y, x, v):
    # Calculate Hessian vector product 
    # y: scalar function to be differentiated (function, e.g. cross entropy loss)
    # x: feed_dict value for the network (dictionary, e.g. {model.X: image_batch, model.y: label_batch})
    # v: vector to be producted (by Hessian) (numeric dictionary, e.g., g(z_test))
    ## w: variables to differentiate (numeric, e.g. neural network weight)

    # hyperparameter r
    r = 1e-4
    
    assert type(x)==dict, "Input of HVP is wrong. 'x' should be dictionary(feed dict format)"
    assert type(v)==dict, "Input of HVP is wrong. 'v' should be dictionary(weight:value format)"
    assert set(v.keys())-set(y.parameters)==set(), "Keys of 'v' should be a subset of weights of 'y'"

    w = v.keys()

    # gradient for plus
    weight_update(w, v, +r)
    g_plus = y.grad(x, wrt=w)

    # gradient for minus
    weight_update(w, v, -2*r)
    g_minus = y.grad(x, wrt=w)

    # weight reconstruction
    weight_update(w, v, +r) # FIXME : this does not reconstruct the weights well

    hvp = {ks: (g_plus[ks] - g_minus[ks])/(2*r) for ks in g_plus.keys()}

    return hvp
