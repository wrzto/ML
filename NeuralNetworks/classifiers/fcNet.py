#-*-coding:utf-8-*-
import numpy as np
from ..layers import *


class FullyConnectedNet(object):
    """
    FullyConnected version of neural network.
    params:
        - hidden_dims: Number of hidden layer neurons.
            ->type: tuple
        - input_dim: Number of input layer features.
            ->type: int
        - num_classes: The output layer dimension is the number of categories.
            ->type: int
        - dropout: According to a certain probability to retain the data, less or equal zero means unuse dropout.
            ->type: float
            ->range: [0, 1]
        - use_batchnorm: use batch normalization or not.
            ->type: bool
        - reg: use regularization or not. zero means unuse.
            ->type: float
        - weight_scale: control weight scale. Suppose that if your network is deeper, it should be smaller. default 1e-2
            ->type: float or int
        - dtype: control weights and biases data type, they must be float.
        - seed: a seed service for random dropout, default None
            ->type: int
        - loss_func: choose a loss function to compute loss. only two options , svm or softmax
            -type: str
    """
    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0, weight_scale=1e-2,
                 dtype=np.float32, seed=None, loss_func='softmax'):
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = dict()
        self.loss_func = None
        if loss_func not in ['softmax', 'svm']:
            raise ValueError('%s function not support!' % loss_func)
        if loss_func == 'softmax':
            self.loss_func = softmax_loss
        elif loss_func == 'svm':
            self.loss_func = svm_loss
        
        #network params initialization.
        for layer in range(self.num_layers):
            if layer == 0:
                N = input_dim
                M = hidden_dims[layer]
            elif layer == len(hidden_dims):
                N = hidden_dims[-1]
                M = num_classes
            else:
                N = hidden_dims[layer - 1]
                M = hidden_dims[layer]
            self.params['W%d' % (layer + 1)] = weight_scale * \
                np.random.randn(N, M)
            self.params['b%d' % (layer + 1)] = np.zeros(M)
            if self.use_batchnorm and layer < len(hidden_dims):
                self.params['gamma%d' % (layer + 1)] = np.ones(M)
                self.params['beta%d' % (layer + 1)] = np.zeros(M)
        
        #set dropout params.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed
        
        #set batch normalization params.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for i in range(self.num_layers - 1)]

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    #compute loss and gradient. 
    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        #test model only compute the scores, train mode compute loss and gradient.
        mode = 'test' if y is None else 'train'
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode
                
        scores = None
        #store intermediate data to compute gradient.
        cache = dict()
        inputs = X
        #Forward propagation.
        for layer in range(self.num_layers):
            inputs, cache['affine_cache%d' % (layer + 1)] = affine_forward(
                inputs, self.params['W%d' % (layer + 1)], self.params['b%d' % (layer + 1)])
            if layer != self.num_layers - 1:
                if self.use_batchnorm:
                    inputs, cache['bn_cache%d' % (layer + 1)] = batchnorm_forward(inputs, self.params[
                        'gamma%d' % (layer + 1)], self.params['beta%d' % (layer + 1)], self.bn_params[layer])
                inputs, cache['relu_cache%d' %
                              (layer + 1)] = relu_forward(inputs)
                if self.use_dropout:
                    inputs, cache['dropout_cache%d' % (
                        layer + 1)] = dropout_forward(inputs, self.dropout_param)
        scores = inputs

        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        loss, dloss_dscore = self.loss_func(scores, y)
        for layer in range(self.num_layers):
            loss += 0.5 * self.reg * \
                np.sum(np.square(self.params['W%d' % (layer + 1)]))
            loss += 0.5 * self.reg * \
                np.sum(np.square(self.params['b%d' % (layer + 1)]))

        dout = dloss_dscore
        #Backward propagation
        for layer in range(self.num_layers):
            layer = self.num_layers - layer
            if layer != self.num_layers:
                if self.use_dropout:
                    dout = dropout_backward(
                        dout, cache['dropout_cache%d' % layer])
                dout = relu_backward(dout, cache['relu_cache%d' % layer])
                if self.use_batchnorm:
                    dout, grads['gamma%d' % layer], grads['beta%d' % layer] = batchnorm_backward(
                        dout, cache['bn_cache%d' % (layer)])
            dout, grads['W%d' % layer], grads['b%d' % layer] = affine_backward(
                dout, cache['affine_cache%d' % layer])
            grads['W%d' % layer] += self.reg * self.params['W%d' % layer]
            grads['b%d' % layer] += self.reg * self.params['b%d' % layer]

        return loss, grads
