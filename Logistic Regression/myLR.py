#-*-coding:utf-8-*-

import numpy as np


def loadDataset():
    dataset = []
    labels = []
    with open('testSet.txt') as f:
        for line in f:
            line = line.strip().split()
            dataset.append([1.0, float(line[0]), float(line[1])])
            labels.append(int(line[2]))
    return dataset, labels

def sigmoid(x):
    return 1/(1 + np.exp(-x))


####梯度上升算法
def gradAscent(dataset, labels):
    if not isinstance(dataset, np.ndarray):
        dataset = np.array(dataset, dtype=np.float32)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels, dtype=np.float32)
        labels.shape = (labels.shape[0], 1)
    m, n = dataset.shape
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    x = [[1],[1],[1]]
    y = 0
    for i in range(maxCycles):
        h = sigmoid(np.dot(dataset, weights))
        error = labels - h
        weights = weights + alpha * np.dot(dataset.transpose(), error)
        x[0].append(weights[0])
        x[1].append(weights[1])
        x[2].append(weights[2])
        y += 1
    return weights, x, y

####随机梯度上升算法version-1
def stocGradAscent0(dataset, labels):
    if not isinstance(dataset, np.ndarray):
        dataset = np.array(dataset, dtype=np.float32)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels, dtype=np.float32)
    m, n = dataset.shape
    alpha = 0.001
    weights = np.ones(n)
    x = [[1],[1],[1]]
    y = 0
    for i in range(m):
        h = sigmoid(np.sum(dataset[i] * weights))
        error = labels[i] - h
        weights = weights + alpha * error * dataset[i]
        x[0].append(weights[0])
        x[1].append(weights[1])
        x[2].append(weights[2])
        y += 1
    return weights, x, y

####随机梯度上升算法version-2
def stocGradAscent1(dataset, labels, numIter = 150):
    from random import sample
    if not isinstance(dataset, np.ndarray):
        dataset = np.array(dataset, dtype=np.float32)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels, dtype=np.float32)
    m, n = dataset.shape
    weights = np.ones(n)
    x = [[1],[1],[1]]
    y = 0
    for j in range(numIter):
        index = sample(range(m), m)
        for i in range(m):
            alpha = 4/(1.0+i+j) + 0.01
            h = sigmoid(np.sum(dataset[index[i]] * weights))
            error = labels[index[i]] - h
            weights = weights + alpha * error * dataset[index[i]]
            x[0].append(weights[0])
            x[1].append(weights[1])
            x[2].append(weights[2])
            y += 1
    return weights, x, y

def plotBestFit(dataset=None, labels=None):
    import matplotlib.pyplot as plt
    if dataset is None:
        dataset, labels = loadDataset()
    dataset = np.array(dataset, dtype=np.float32)
    num = dataset.shape[0]
    # weights, wx, wy = gradAscent(dataset, labels)
    # weights, wx, wy = stocGradAscent0(dataset, labels)
    weights, wx, wy = stocGradAscent1(dataset, labels)
    xcord1 = []
    ycord1 = []
    xcord0 = []
    ycord0 = []
    for i in range(num):
        if labels[i] == 1:
            xcord1.append(dataset[i][1])
            ycord1.append(dataset[i][2])
        else:
            xcord0.append(dataset[i][1])
            ycord0.append(dataset[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord0, ycord0, s=30, c='red', marker='s')
    ax.scatter(xcord1, ycord1, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1]*x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    plotW(wx, wy)       

def plotW(x, y):
    import matplotlib.pyplot as plt
    plt.figure(1)
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)
    ax1.plot(range(y+1), x[0])
    ax2.plot(range(y+1), x[1])
    ax3.plot(range(y+1), x[2])
    plt.show()


def classify(inx, weights):
    prob = sigmoid(sum(inx*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

####实例:从疝气病症预测病马的死亡率
def testing():
    train_dataset = []
    train_labels = []
    test_dataset = []
    test_labels = []
    with open('horseColicTraining.txt') as f:
        for line in f:
            line = line.strip().split('\t')
            train_dataset.append(line[:-1])
            train_labels.append(line[-1])
    with open('horseColicTest.txt') as f:
        for line in f:
            line = line.strip().split('\t')
            test_dataset.append(line[:-1])
            test_labels.append(line[-1])
    weights, _, _ = stocGradAscent1(train_dataset, train_labels)
    test_dataset = np.array(test_dataset, dtype=np.float32)
    test_labels = [float(label) for label in test_labels]
    result_labels = []
    for data in test_dataset:
        result_labels.append(classify(data, weights))
    error = 0
    for res in zip(result_labels, test_labels):
        if res[0] != res[1]:
            error += 1
    print 'error accuary: %f' % (error/float(len(test_labels)))


































