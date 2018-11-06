# Hessian Vector Product (Using numerical differentiation)
import numpy as np

def weight_update(w, v, r):
    # w: weights of neural network (tuple)
    # v: value for delta w (dictionary, e.g., gradient value)
    # r: hyperparameter for a gradient (scalar)

    for p in w:
        p.value += r * v[p]

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

    # weight reconstruction
    # intermediate reconstruction may increase extra computational cost, but decrease the precision loss
    weight_update(w, v, -r)

    # gradient for minus
    weight_update(w, v, -r)
    g_minus = y.grad(x, wrt=w)

    # weight reconstruction
    weight_update(w, v, +r)

    hvp = {ks: (g_plus[ks] - g_minus[ks])/(2*r) for ks in g_plus.keys()}

    return hvp
