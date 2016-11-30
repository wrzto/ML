#!/usr/bin python
#-*-coding:utf-8-*-

import numpy as np
from collections import Counter

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inx, dataset, labels, k=1):
    ##预处理(此处的输入labels是带有具体分类内容的list),inx和dataset都numpy对象
    if k <= 0:
        k = 1
    try:
        y = inx.shape[1]
    except:
        inx.shape=(-1, inx.shape[0])
    ##计算欧氏距离
    num_test = inx.shape[0]
    num_train = dataset.shape[0]
    dists = np.zeros((num_test, num_train))
    dists = np.multiply(np.dot(inx, dataset.T), -2)
    inx_sq = np.sum(np.square(inx), axis=1, keepdims=True)
    dataset_sq = np.sum(np.square(dataset), axis=1)
    dists = np.add(dists, inx_sq)
    dists = np.add(dists, dataset_sq)
    dists = np.sqrt(dists)
    ###获取标签
    result = []
    per_line_labels=[]
    sort_arg = dists.argsort()[:,:k]
    for line in sort_arg:
        per_line_labels = [labels[index] for index in line]
        result.append(Counter(per_line_labels).most_common(1)[0][0])
    return result


# group,labels = createDataSet()
# print classify0([[0,0],[0,0]],group,labels,3)

#########################################################约会网站############################################################################            

def file1matrix(filename):
    ###从文件中读取数据并转为可计算的numpy对象
    dataset = []
    labels = []
    with open(filename,'r') as f:
        for line in f:
            line = line.strip().split('\t')
            labels.append(line.pop())
            dataset.append(line)
    dataset = np.array(dataset, dtype=np.float32)
    return dataset, labels

def convert(labels):
    label_names = list(set(labels))
    labels = [label_names.index(label) for label in labels]
    return label_names,labels 

# filename = 'datingTestSet.txt'
# dataset, labels = file1matrix(filename)
# label_names, labels = convert(labels)

def draw(dataset, labels, label_names):
    labels = [ i+1 for i in labels]  ###下表加1，绘色
    from matplotlib import pyplot as plt
    from matplotlib import font_manager
    zhfont = font_manager.FontProperties(fname='C:\\Windows\\Fonts\\msyh.ttc')
    plt.figure(figsize=(8, 5), dpi=80)
    ax = plt.subplot(111)
    # ax.scatter(dataset[:,1], dataset[:,2], 15.0*np.array(labels), 15.0*np.array(labels))
    # plt.show()
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    for i in xrange(len(labels)):
        if labels[i] == 1:
            type1_x.append(dataset[i][0])
            type1_y.append(dataset[i][1])
        if labels[i] == 2:
            type2_x.append(dataset[i][0])
            type2_y.append(dataset[i][1])
        if labels[i] == 3:
            type3_x.append(dataset[i][0])
            type3_y.append(dataset[i][1])
    ax.scatter(type1_x, type1_y, color = 'red', s = 20)
    ax.scatter(type2_x, type2_y, color = 'green', s = 20)    
    ax.scatter(type3_x, type3_y, color = 'blue', s = 20)    
    plt.xlabel(u'飞行里程数', fontproperties=zhfont)
    plt.ylabel(u'视频游戏消耗时间', fontproperties=zhfont)
    ax.legend((label_names[0], label_names[1], label_names[2]), loc=2, prop=zhfont)
    plt.show()

# draw(dataset, labels, label_names)

def autoNorm0(dataset):
    if not isinstance(dataset, np.ndarray):
        dataset = np.array(dataset,dtype=np.float32)
    ###归一化特征值 newvalue = (oldvalue - min) / (max - min)
    minVals = dataset.min(0)
    maxVals = dataset.max(0)
    ranges = maxVals - minVals
    dataset = dataset - minVals
    dataset = dataset / ranges
    return dataset

def autoNorm1(dataset):
    ###归一化特征值 newvalue = (oldvalue - 均值) / 标准差    ----->推荐使用这种方法
    if not isinstance(dataset, np.ndarray):
        dataset = np.array(dataset,dtype=np.float32)
    mean = dataset.mean(0)
    std = dataset.std(0)
    dataset = dataset - mean
    dataset = dataset / std
    return dataset
       

def datingTest():
    ##随机选取测试集和训练集
    filename = 'datingTestSet.txt'
    dataset, labels = file1matrix(filename)
    dataset = autoNorm1(dataset)
    train_length = int(dataset.shape[0] * 0.9)
    test_length = dataset.shape[0] - train_length
    from random import sample
    all_index = sample(range(dataset.shape[0]), dataset.shape[0])
    train_index = all_index[:train_length]
    test_index = all_index[-test_length:]
    train_dataset = dataset[train_index, :]
    train_labels = []
    test_dataset = dataset[test_index, :]
    test_labels = []
    for index in train_index:
        train_labels.append(labels[index])
    for index in test_index:
        test_labels.append(labels[index])
    ##训练并计算错误率
    test_result = classify0(test_dataset, train_dataset, train_labels, k=10)
    error = 0
    for res in zip(test_result, test_labels):
        if res[0] != res[1]:
            error += 1
    print 'error accaury:%f' % (float(error) / len(test_labels))


def classifyPerson(inx, k=1):
    filename = 'datingTestSet.txt'
    dataset, labels = file1matrix(filename)
    dataset = autoNorm1(dataset)
    inx = autoNorm1(inx)
    print inx
    print classify0(inx, dataset, labels, k)

#########################################################数字识别#################################################################


import os

def imgVector(filename):
    vect = []
    with open(filename,'r') as f:
        for line in f:
            line = line.strip()
            vect += [float(n) for n in line]
    number = os.path.split(filename)[-1].split('_')[0]
    return np.array(vect, dtype=np.float32), number

def all_imgVector(directory):
    filelist = os.listdir(directory)
    vects = []
    labels = []
    for filename in filelist:
        vect, label= imgVector(os.path.join(directory, filename))
        vects.append(vect)
        labels.append(label)
    return np.array(vects, dtype=np.float32), labels

# test_dir = 'digits\\testDigits'
# train_dir = 'digits\\trainingDigits'
# train_dataset, train_labels = all_imgVector(train_dir)
# test_dataset, test_labels = all_imgVector(test_dir)


def handwritingClassTest():
    test_dir = 'digits\\testDigits'
    train_dir = 'digits\\trainingDigits'
    train_dataset, train_labels = all_imgVector(train_dir)
    test_dataset, test_labels = all_imgVector(test_dir)
    result_labels = classify0(test_dataset, train_dataset, train_labels, k=3)

    error = 0 
    for res in zip(result_labels, test_labels):
        if res[0] != res[1]:
            error += 1
    print 'error accaury:%f' % (float(error) / len(test_labels))     


def classifyImg(inx, k=1):
    train_dir = 'digits\\trainingDigits'
    train_dataset, train_labels = all_imgVector(train_dir)
    result_labels = classify0(inx, train_dataset, train_labels, k)
    print result_labels



def handwriting1():
    import pandas as pd
    import numpy as np
    trainDataset = pd.read_csv('./dataset/train.csv')
    testDataset = pd.read_csv('./dataset/test.csv')
    X_train = trainDataset.drop(['label'], axis=1).values
    Y_train = trainDataset['label'].values.reshape(-1)
    print Y_train.dtype
    # X_test = testDataset.values
    num = int(X_train.shape[0]) * 0.8
    result_labels = classify0(X_train[num:], X_train[:num], Y_train[:num], k=3)
    error = 0 
    for res in zip(result_labels, Y_train[num:]):
        if int(res[0]) != int(res[1]):
            error += 1
    print 'error accaury:%f' % (float(error) / len(test_labels)) 













