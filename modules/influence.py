# implementation of inverse HVP using conjugate gradient and stochastic estimation
from ipdb import set_trace

import cntk as C
import numpy as np
import time

from torch.utils.data import DataLoader

from scipy.optimize import fmin_ncg, fmin_cg

from modules.hvp import HVP

# INSTEAD OF USING THIS, SET 'IS_TRAIN=FALSE'
#def disable_dropout(y):
#    # clone and make dropout_rate which is used in 'C.layers.Dropout' to 0
#    # this should be done if you use y.grad in 'test phase'
#
#    # input share
#    subs = {ip: ip for ip in y.inputs if ip.kind()==0} # add to substitutions if input_variable
#
#    # dropout substitution
#    list_layers = y.find_all_with_name('')
#    list_dropout = [l for l in list_layers if l.as_string().split(':')[0]=='Dropout']
#    for do in list_dropout:
#        assert len(do.inputs)==1, 'Input node of Dropout layers could be multiple?'
#        subs[do] = C.layers.Dropout(dropout_rate=0.0)(do.inputs[0])
#
#    return y.clone('share', substitutions=subs)

# Conjugate Gradient
def get_inverse_hvp_cg(model, y, v, data_set, method='Basic', **kwargs):
    # Calculate inverse hessian vector product over the training set using CG method
    # return x, which is the solution of QP, whose value is H^-1 v
    # model: neural network model (e.g. model)
    # y: scalar function output of the neural network (e.g. model.loss)
    # v: vector to be producted by inverse hessian (i.e.H^-1 v) (e.g. v_test)
    # data_set: training set to be summed in Hessian
    # method: Basic-> Conjugate Gradient, Newton -> Newton-Conjugate Gradient
    # kwargs: hyperparameters for conjugate gradient

    # hyperparameters
    batch_size = kwargs.pop('batch_size', 128)
    #batch_size = kwargs.pop('batch_size', 1) 
    # remark) changing the size of batch can induce randomness of output 
    # due to precision loss and parallel computing
    damping = kwargs.pop('damping', 0.0)
    avextol = kwargs.pop('avextol', 1e-8)
    maxiter = kwargs.pop('maxiter', 1e2)
    num_workers = kwargs.pop('num_workers', 6)

    get_inverse_hvp_cg.dl = DataLoader(data_set, batch_size, shuffle=False, num_workers=num_workers)
    get_inverse_hvp_cg.damp = damping
    get_inverse_hvp_cg.cnt = 0
    get_inverse_hvp_cg.fmt = {key: val.shape for (key, val) in v.items()}
    get_inverse_hvp_cg.temp_hvp = dic2vec(v) # temporal hvp for callback

    t0 = time.time()

    def HVP_minibatch_val(y, v):
        # Calculate Hessian vector product w.r.t whole dataset
        # y: scalar function output of the neural network (e.g. model.loss)
        # v: vector to be producted by inverse hessian (i.e.H^-1 v) (numeric dictionary, e.g. v_test)

        ## model: neural network model (e.g. model)
        ## dataloader: dataloader for the training set
        ## damping: damp term to make hessian convex

        num_data = data_set.__len__()

        hvp_batch = {key: np.zeros_like(value) for key,value in v.items()}

        for img, lb in get_inverse_hvp_cg.dl:
            img = img.numpy(); lb = lb.numpy()
            x_feed = {model.X: img, model.y:lb}
            hvp = HVP(y,x_feed,v)
            # add hvp value
            for ks in hvp.keys():
                hvp_batch[ks] += hvp[ks] # gradient will do batch-wise summation

        # normalize after the summation to reduce precision loss
        hvp_batch = {key: val/num_data for (key,val) in hvp_batch.items()}

        # damping term
        for ks in hvp.keys():
            hvp_batch[ks] += get_inverse_hvp_cg.damp * v[ks]

        # update after evaluation
        get_inverse_hvp_cg.temp_hvp = dic2vec(hvp_batch)

        return hvp_batch

    def get_fmin_loss_fn(y, v):
        def fmin_loss_fn(x):
            x_dic = vec2dic(x, get_inverse_hvp_cg.fmt)
            hvp_val = HVP_minibatch_val(y, x_dic)

            return 0.5 * grad_inner_product(hvp_val, x_dic) - grad_inner_product(v, x_dic)
        return fmin_loss_fn

    def get_fmin_grad_fn(y, v):
        def fmin_grad_fn(x):
            # x: 1D vector
            x_dic = vec2dic(x, get_inverse_hvp_cg.fmt)
            hvp_val = HVP_minibatch_val(y, x_dic)
            hvp_flat = dic2vec(hvp_val)
            v_flat = dic2vec(v)

            return hvp_flat - v_flat
        return fmin_grad_fn

    def get_fmin_hvp_fn(y, v):
        def fmin_hvp_fn(x, p):
            p_dic = vec2dic(p, get_inverse_hvp_cg.fmt)
            hvp_val = HVP_minibatch_val(y, p_dic)
            hvp_flat = dic2vec(hvp_val)

            return hvp_flat
        return fmin_hvp_fn

    def get_cg_callback(v, t0):
        def cg_callback(x):
            print('iteration: {}'.format(get_inverse_hvp_cg.cnt), ', ', time.time()-t0, '(sec) elapsed')
            print('vector element-wise square: ', np.inner(x, x))
            grad_prev = get_inverse_hvp_cg.temp_hvp-dic2vec(v) # previous gradient value which should be 0
            print('temporal gradient value: ', np.inner(grad_prev,grad_prev))
            ambiguous_loss = 1/2* np.inner(get_inverse_hvp_cg.temp_hvp,x) - np.inner(x, dic2vec(v))
            print('temporal function value(ambiguous): ', ambiguous_loss)
            get_inverse_hvp_cg.cnt += 1

            return 0
        return cg_callback

    fmin_loss_fn = get_fmin_loss_fn(y, v)
    fmin_grad_fn = get_fmin_grad_fn(y, v)
    fmin_hvp_fn = get_fmin_hvp_fn(y, v)
    cg_callback = get_cg_callback(v, t0)

    if method == 'Newton':
        fmin_results = fmin_ncg(\
                f = fmin_loss_fn, x0 = dic2vec(v), fprime = fmin_grad_fn,\
                fhess_p = fmin_hvp_fn, avextol = avextol, maxiter = maxiter, callback=cg_callback)
    else:
        fmin_results = fmin_cg(\
                f = fmin_loss_fn, x0 = dic2vec(v), fprime = fmin_grad_fn,\
                maxiter = maxiter, callback = cg_callback)

    return vec2dic(fmin_results, get_inverse_hvp_cg.fmt)

def dic2vec(dic, order=None):
    # convert a dictionary with matrix values to a 1D vector
    # e.g. gradient of network -> 1D vector
    # order: list of keys of dic to specify order
    if order == None:
        order = dic.keys()
    vec = np.concatenate([dic[key].reshape(-1) for key in order])

    return vec

def vec2dic(vec, fmt, order=None):
    # convert a 1D vector to a dictionary of format fmt
    # fmt: {key: val.shape for (key,val) in dict}
    # order: list of keys of dic to specify order
    if order == None:
        order = fmt.keys()
    fmt_idx = [np.prod(fmt[key]) for key in order]
    vec_split = [vec[sum(fmt_idx[:i]):sum(fmt_idx[:i+1])] for i in range(len(fmt_idx))]
    dic = {key: vec_split[i].reshape(fmt[key]) for (i,key) in enumerate(order)}

    return dic

# Stochastic Estimation (or LISSA)
def get_inverse_hvp_se(model, y, v, data_set, **kwargs):
    # Calculate inverse hessian vector product over the training set using SE method
    # model: neural network model (e.g. model)
    # y: scalar function output of the neural network (e.g. model.loss)
    # v: vector to be producted by inverse hessian (i.e.H^-1 v) (e.g. v_test)
    # data_set: training set to be summed in Hessian
    # kwargs: hyperparameters for stochastic estimation

    # hyperparameters
    recursion_depth = kwargs.pop('recursion_depth', 50) # epoch
    scale = kwargs.pop('scale', 1e1) # similar to learning rate
    damping = kwargs.pop('damping', 0.0) # paper reference: 0.01
    batch_size = kwargs.pop('batch_size', 1)
    num_samples = kwargs.pop('num_samples', 1) # the number of samples(:stochatic estimation of IF) to be averaged
    tolerance = kwargs.pop('tolerance', 1e-2) # the difference btw l2 norms of current and previous vector used for early stopping
    num_workers = kwargs.pop('num_workers', 6)
    verbose = kwargs.pop('verbose', False)

    dataloader = DataLoader(data_set, batch_size, shuffle=True, num_workers=num_workers)

    inv_hvps = []

    params = list(v.keys())
    params0 = [p.value for p in params]

    for i in range(num_samples):
        # obtain num_samples inverse hvps
        cur_estimate = v
        prev_norm = 0

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
                print('#w: \n', [p.value for p in params], '\n#hvp: \n', hvp, '\n#ihvp: \n', cur_estimate)

            cur_norm = np.sqrt(grad_inner_product(cur_estimate,cur_estimate))
            print('Recursion depth: {}, norm: {}, time: {} \n'.format(depth, cur_norm,time.time()-t1))

            # divergence check
            if np.isnan(cur_norm):
                print('## The result has been diverged ##')
                # recover the params from NaN
                for p, pv0 in zip(params, params0):
                    p.value = pv0
                break

            # convergence check
            if np.abs(cur_norm - prev_norm) <= tolerance:
                # change this to more precise one (<- scipy.fmin_cg also use gnorm)
                print('## Early stopped due to small change')
                break
            prev_norm = cur_norm

        inv_hvp = {ks: (1/scale)*cur_estimate[ks] for ks in cur_estimate.keys()}
        inv_hvps.append(inv_hvp)

    inv_hvp_val = {ks: np.mean([inv_hvps[i][ks] for i in range(num_samples)], axis=0) for ks in inv_hvps[0].keys()}

    return inv_hvp_val

def get_influence_val(model, y, ihvp, data_set, normalization=False, **kwargs):
    # Calculate influence function value when H^-1 v_test is given w.r.t. data_set
    # cf) this will be calculated sample-wisely due to memory issue

    # y: scalar function output of the neural network (e.g. model.loss)
    # ihvp: inverse of Hessian Vector Product (dictionary) (e.g. H^-1 v_test)
    # data_set: data set to be fed (dataset class) (e.g. train_set)
    # kwargs: hyperparameters

    num_workers = kwargs.pop('num_workers', 6)

    if_list = []

    params = list(ihvp.keys()) # not (model.logits.parameters) due to freezing

    num_data = data_set.__len__()
    dataloader = DataLoader(data_set, 1, shuffle=False, num_workers=num_workers)

    t1 = time.time()
    for img, lb in dataloader:
        img = img.numpy(); lb = lb.numpy()
        gd = y.grad({model.X:img, model.y:lb}, wrt=params)
        if normalization == 'cosine':
            # cosine normalization
            nrm = np.sqrt(grad_inner_product(gd,gd))
            gd = {k: v/nrm for k,v in gd.items()}
        if normalization == 'loss':
            # divided by loss
            loss = y.loss({model.X:img, model.y:lb})
            gd = {k: v/loss for k,v in gd.items()}
        if_val = -grad_inner_product(ihvp, gd) / num_data
        if_list.append(if_val)
    print('get_influence_val takes {} sec'.format(time.time()-t1))

    return if_list

def grad_inner_product(grad1, grad2):
    # inner product for dictionary-format gradients (output scalar value)

    val = 0

    assert(len(grad1)==len(grad2))

    for ks in grad1.keys():
        val += np.sum(np.multiply(grad1[ks],grad2[ks]))

    return val
