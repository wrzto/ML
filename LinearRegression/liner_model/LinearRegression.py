#-*-coding:utf-8-*-

import numpy as np


class LinearRegression(object):
    def __init__(self, slover='standRegres'):
        self.cof = None
        self.slover = slover
        self.alpha = 0.01
        self.max_iter = 3000
        self.tol = 1e-7
        self.batch_size = 20
    
    def standRegres(self, X, y):
        '''
        X --> [m,n]
        y --> [m]
        '''
        X = np.mat(X)
        y = np.mat(y).T
        xtx = X.T * X
        if np.linalg.det(xtx) == 0.0:
            print("输入矩阵不可逆")
            return
        self.cof = xtx.I * (X.T*y)
    
    def compute_loss(self, X, y):
        X = np.mat(X)
        y = np.mat(y)
        return np.sqrt(np.sum(np.square((X * self.cof).T - y)) / y.shape[1])
    
    def gradDescent(self, X, y):
        X = np.mat(X)
        y = np.mat(y)
        try:
            n = X.shape[1]
        except Exception as err:
            n = 1
        self.cof = np.random.randn(n,1)
        for i in range(self.max_iter):
            next_cof = self.cof - self.alpha * (((X * self.cof).T-y)*X).T
            if np.sqrt(np.sum(np.square(next_cof-self.cof))) < self.tol:
                break
            print('loss: {}'.format(self.compute_loss(X,y)))
            self.cof = next_cof
    
    def StocGradientDescent(self, X, y):
        X = np.mat(X)
        y = np.mat(y)
        try:
            n = X.shape[1]
        except Exception as err:
            n = 1
        self.cof = np.random.randn(n,1)
        flag = False
        for _ in range(self.max_iter):
            for i,sample in enumerate(X):
                next_cof = self.cof - self.alpha * (((sample * self.cof).T-y[0,i])*sample).T
                if np.sqrt(np.sum(np.square(next_cof-self.cof))) < self.tol:
                    break
                    flag = True
                self.cof = next_cof
            if flag:
                break
                
    def minibatchGradDescent(self, X, y):
        X = np.mat(X)
        y = np.mat(y)
        m = X.shape[0]
        try:
            n = X.shape[1]
        except Exception as err:
            n = len(X)
            m = 1
        if m < self.batch_size:
            print('样本数小于batch size。')
            return
        self.cof = np.random.randn(n,1)
        flag = False
        for _ in range(self.max_iter):
            batch_mask = np.random.choice(m, self.batch_size)
            next_cof = self.cof - self.alpha * (((X[batch_mask,:] * self.cof).T-y[0,batch_mask])*X[batch_mask,:]).T
            if np.sqrt(np.sum(np.square(next_cof-self.cof))) < self.tol:
                break
            self.cof = next_cof 

    def fit(self, X, y):
        getattr(self, self.slover)(X, y)
    
    def predict(self, X):
        X = np.mat(X)
        return np.array(self.cof.T * X.T).reshape(-1)




    


        

