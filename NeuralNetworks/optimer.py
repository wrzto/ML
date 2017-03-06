#-*-coding:utf-8-*-

import numpy as np


def sgd(w, dw, config=None):
    """
    stochastic gradient descent.
    """
    if config is None:
        config = dict()
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    gradient descent combine with momentum.
    Related papers:
    [1]https://arxiv.org/pdf/1212.0901v2.pdf
    [2]http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf
    """
    if config is None:
        config = dict()
    config.setdefault('learing_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    next_w = None
    # Momentum
    # momentum, learning_rate = config['momentum'], config['learning_rate']
    # v = momentum * v - learning_rate * dw
    # w += v
    # next_w = w
    momentum, learning_rate = config['momentum'], config['learning_rate']
    v_pred = v
    v = momentum * v - learning_rate * dw
    w += - momentum * v_pred + (1 + momentum) * v
    next_w = w
    config['velocity'] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    A very efficient, but not publicly published, adaptive learning rate approach.
    Related papers:
    [1]http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """
    if config is None:
        config = dict()
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(w))

    next_w = None
    learning_rate = config['learning_rate']
    decay_rate = config['decay_rate']
    epsilon = config['epsilon']
    cache = config['cache']

    # Adagrad
    # cache += np.square(dx)
    # x += -learning_rate * dx / (np.sqrt(cache) + epsilon)

    cache = decay_rate * cache + (1 - decay_rate) * np.square(dw)
    w += -learning_rate * dw / (np.sqrt(cache) + epsilon)

    next_w = w
    config['cache'] = cache

    return next_w, config


def adam(w, dw, config=None):
    """
    rmsprop combine with momentum. suggest using this method. 
    Related papers:
    [1]https://arxiv.org/abs/1412.6980
    """
    if config is None:
        config = dict()
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(w))
    config.setdefault('v', np.zeros_like(w))
    config.setdefault('t', 0)

    next_w = None
    learning_rate = config['learning_rate']
    beta1 = config['beta1']
    beta2 = config['beta2']
    epsilon = config['epsilon']
    m, v, t = config['m'], config['v'], config['t'] + 1
    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * np.square(dw)
    mb = m / (1 - beta1**t)
    vb = v / (1 - beta2**t)
    w += -learning_rate * mb / (np.sqrt(vb) + epsilon)

    next_w = w
    config['m'] = m
    config['v'] = v
    config['t'] = t

    return next_w, config
