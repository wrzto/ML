#-*-coding:utf-8-*-

import numpy as np
from . import optimer
import matplotlib.pyplot as plt


class Solver(object):
    """
    params:
        - model: network model object.
        - data: it contains X_train, y_train, X_val, y_val
            ->type: dict
        - update_relu: 'sgd', 'sgd_momentum', 'rmsprop', 'adam', default 'sgd'
            ->type: str
        - optim_config: optimer config. default empty dictionary.
            ->type: dcit
        - lr_decay: Learning Rate Decay. default 1.0
            ->range: (0, 1.0]
            ->type: float
        - batch_size: default 100. the size for train.
            ->type: int
        - num_epochs: epochs 
            ->type: int
        - print_every: print frequency 
            ->type: int
        - verbose: print info or not
            ->type: bool
    """

    def __init__(self, model, data, **kwargs):
        self.model = model
        self.X_train = data.get('X_train', None)
        self.y_train = data.get('y_train', None)
        self.X_val = data.get('X_val', None)
        self.y_val = data.get('y_val', None)

        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)

        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in kwargs.keys())
            raise ValueError('Unrecognized arguments %s' % extra)

        if not hasattr(optimer, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optimer, self.update_rule)

        self._reset()

    def _reset(self):
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = dict()
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        self.optim_configs = dict()
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.iteritems()}
            self.optim_configs[p] = d

    def _step(self):
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        for p, w in self.model.params.iteritems():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        num_batches = N / batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc

    def train(self):
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train / self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            if self.verbose and t % self.print_every == 0:
                print('(iteration %d / %d) loss: %f' %
                      (t + 1, num_iterations, self.loss_history[-1]))

            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay
            first_it = (t == 0)
            last_it = (t == num_iterations - 1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(
                    self.X_train, self.y_train, num_samples=1000)
                val_acc = self.check_accuracy(self.X_val, self.y_val)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)

                if self.verbose:
                    print '(Epoch %d / %d) train acc: %f; val_acc: %f' % (self.epoch, self.num_epochs, train_acc, val_acc)
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.iteritems():
                        self.best_params[k] = v.copy()
        self.model.params = self.best_params
    
    def predict_prob(self, X):
        scores = self.model.loss(X)
        prob = scores - np.max(scores, axis=1, keepdims=True)
        prob = np.exp(prob) / np.sum(np.exp(prob), axis=1, keepdims=True)

        return prob
    
    def predict(self, X):
        prob = self.preidct_prob(X)
        return np.argmax(prob)
    
    

