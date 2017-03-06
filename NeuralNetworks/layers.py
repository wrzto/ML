#-*-coding:utf-8-*-

import numpy as np


def affine_forward(x, w, b):
    """
    compute the result of affine forward propagation.
    """
    out, cache = None, None
    N = x.shape[0]
    D = np.prod(x.shape[1:])
    M = w.shape[-1]
    x_v = x.reshape(N, -1)
    out = x_v.dot(w) + b
    cache = (x, w, b)

    return out, cache


def affine_backward(dout, cache):
    """
    compute the result of affine backward propagation.
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    N = x.shape[0]
    M = w.shape[-1]
    D = np.prod(x.shape[1:])
    x_v = x.reshape(N, -1)
    dw = x_v.T.dot(dout)
    dx = dout.dot(w.T).reshape(x.shape)
    db = np.ones(N).dot(dout)

    return dx, dw, db


def relu_forward(x):
    """
    compute the result of relu forward propagation.
    """
    out = None
    out = np.maximum(0, x)
    cache = x

    return out, cache


def relu_backward(dout, cache):
    """
    compute the result of relu backwrd propagation.
    """
    dx, x = None, cache
    dx = dout
    dx[x <= 0] = 0

    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    compute the result of batch normalization forward propagation.
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        mu = np.mean(x, axis=0)
        xcorrected = x - mu
        xsquarred = np.square(xcorrected)
        var = np.mean(xsquarred, axis=0)
        std = np.sqrt(var + eps)
        isstd = 1.0 / std
        xhat = xcorrected * isstd
        out = gamma * xhat + beta
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var
        cache = (gamma, xhat, xcorrected, isstd, std)
    elif mode == 'test':
        std = np.sqrt(running_var + eps)
        out = (x - running_mean) / std
        out = gamma * out + beta
    else:
        raise ValueError("Invalid forward batchnorm mode %s" % mode)
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    compute the result of batch normalization backward propagation.
    """
    dx, dgamma, dbeta = None, None, None
    gamma, xhat, _, isstd, _ = cache
    N, _ = dout.shape
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(xhat * dout, axis=0)
    dx = (gamma * isstd / N) * (N * dout - xhat * dgamma - dbeta)

    # Slow version
    # dx, dgamma, dbeta = None, None, None
    # gamma, xhat, xcorrected, isstd, std = cache
    # N, D = dout.shape

    # dbeta = np.sum(dout, axis=0)
    # dgamma = np.sum(xhat * dout, axis=0)
    # dxhat = dout * gamma
    # disstd = np.sum(dxhat * xcorrected, axis=0)
    # dstd = disstd * -1 / (std**2)
    # dvar = dstd * 1 / (2 * std)
    # dxsquarred = dvar * np.ones((N, D)) / N
    # dxcorrected = dxhat * isstd + dxsquarred * 2 * xcorrected
    # dmu = np.sum(dxcorrected, axis=0) * -1
    # dx = dxcorrected * 1 + dmu * np.ones((N, D)) / N

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    compute the result of dropout forward propagation.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])
    mask = None
    out = None
    if mode == 'train':
        mask = (np.random.rand(*x.shape) < p) / p
        out = mask * x
    elif mode == 'test':
        out = x
    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    compute the result of dropout backward propagation.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        dx = mask * dout
    elif mode == 'test':
        dx = dout

    return dx


def svm_loss(x, y, C=1.0):
    """
    compute data loss and dloss/dscore by svm.
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores.reshape(N, 1) + C)
    margins[np.arange(N), y] = 0
    data_loss = np.sum(margins) / N
    num_loss = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_loss
    dx /= N

    return data_loss, dx


def softmax_loss(x, y):
    """
    compute data loss and dloss/dscore by softmax. 
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))  # 减去最大的score，防止指数爆炸
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    data_loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    return data_loss, dx
