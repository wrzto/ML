#!/usr/bin python
#-*-coding:utf-8-*-

from math import log
from collections import defaultdict , Counter
import operator

def createDataset():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

# dataset, labels = createDataSet()

def calcShannonEnt(dataset):
    ###计算数据集的香农熵，用于衡量该数据集的复杂度，如果该数据集越复杂，香农熵值越大，反之越小
    numEntries = len(dataset)
    labels = defaultdict(int) 
    for featVec in dataset:
        label = featVec[-1]
        labels[label] += 1
    shannonEnt= 0.0
    for v in labels.values():
        prob = float(v) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt
 
def splitDataset(dataset, axis, value):
    ###根据特征值划分数据集 axis：特征下标 value:特征值
    retDataset=[]
    for featVec in dataset:
        if featVec[axis] == value:
            reducedFeatvec = featVec[:axis]
            reducedFeatvec.extend(featVec[axis+1:])
            retDataset.append(reducedFeatvec)
    return retDataset

def chooseBestFeatureToSplit(dataset):
    ###在当前数据集寻找最适合划分数据集的特征，通过计算根据每种特征划分数据集的信息熵之和,寻找熵增最大的特征
    numFeats = len(dataset[0]) - 1
    baseEntropy = calcShannonEnt(dataset)
    bestInfoGain = 0.0
    bestFeat = -1
    for i in range(numFeats):
        featValues = set([data[i] for data in dataset])
        newEntropy = 0.0
        for value in featValues:
            resDataset = splitDataset(dataset, i, value)
            prop = len(resDataset) / float(len(dataset))
            newEntropy += prop * calcShannonEnt(resDataset)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeat = i
    return bestFeat ##返回特征下标

def majorityCnt(classList):
    ###寻找当前类标签出现最多的标签 classList:标签值得集合
    # # classCount = defaultdict(int)
    # # for vote in classList:
    # #     classCount[vote] += 1
    # # sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # return sortedClassCount[0][0]
    return Counter(classList).most_common()[0][0]

def createTree(dataset, inputLabels):
    ####inputLabels： 特征标签
    labels = inputLabels[:] ##防止输入标签被更改
    classList = [data[-1] for data in dataset]
    ###当前数据集的所有标签值相同，分类结束，返回标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    ###当前已经根据所有特征划分数据集，返回最多的标签值
    if len(dataset[0]) == 1:
        return majorityCnt(classList)
    bestFeatIndex = chooseBestFeatureToSplit(dataset)
    bestFeatValues = set([data[bestFeatIndex] for data in dataset])
    bestFeatLabel = labels[bestFeatIndex]
    del labels[bestFeatIndex] ###划分数据集会移除该特征，对应特征标签也要移除
    trees = {bestFeatLabel:{}}
    for value in bestFeatValues:
        new_labels = labels[:] ###这里必须创建新的list对象，传参后引用会影响当前labels的值
        trees[bestFeatLabel][value] = createTree(splitDataset(dataset, bestFeatIndex, value), new_labels)
    return trees

def classify(tree, featLabels, testVec):
### {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    firstStr = tree.keys()[0]
    secondDict = tree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if key == testVec[featIndex]:
            if isinstance(secondDict[key], dict):
                classLabels = classify(secondDict[key], featLabels, testVec)
            else:
                classLabels = secondDict[key]
    return classLabels

def dumpTree(tree,filename='tree.pkl'):
    import pickle
    with open(filename, 'w') as f:
        pickle.dump(tree, f)

def loadTree(filename='tree.pkl'):
    import pickle
    with open(filename, 'r') as f:
        tree = pickle.load(f)
    return tree

###实例一:隐形眼镜
def loadLensesData():
    dataset = []
    with open('lenses.txt', 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            dataset.append(line)
    labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    return dataset, labels
# ###实例二:识别手写图片
# import os

# def loadImgData(filename):
#     vect = []
#     with open(filename, 'r') as f:
#         for line in f:
#             line = line.strip()
#             vect += list(line)
#     number = os.path.split(filename)[-1].split('_')[0]
#     return vect,number

# def loadAllImgData(directory):
#     filelist = os.listdir(directory)
#     vects = []
#     labels = []
#     for filename in filelist:
#         vect, label= loadImgData(os.path.join(directory, filename))
#         vects.append(vect)
#         labels.append(label)
#     return vects,labels

# def loadTrainData():
#     train_dir = '..\\knn\\digits\\trainingDigits'
#     vects, labels = loadAllImgData(train_dir)
#     for i in xrange(len(labels)):
#         vects[i].append(labels[i])
#     return vects

# def loadTestData():
#     test_dir = '..\\knn\\digits\\testDigits'
#     return loadAllImgData(test_dir)

# def classifyImg():
#     train_dataset = loadTrainData()
#     test_dataset, test_labels = loadTestData()
#     tree = createTree(train_dataset, range(1024))
#     print u'tree已经建立.'
#     dumpTree(tree, filename='img.pkl')
#     result_labels = []
#     for data in test_dataset:
#         result_labels.append(classify(tree, range(1024), data))
#     error = 0
#     for res in zip(result_labels, test_labels):
#         if res[0] != res[1]:
#             error += 1
#     print 'error accuary :%f' % (float(error) / len(test_labels))

#####################绘图区######################
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeText, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeText, xy=parentPt, xycoords='axes fraction', \
        xytext=centerPt, textcoords='axes fraction', va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)

def createPlot(tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(tree))
    plotTree.totalD = float(getTreeDepth(tree))
    plotTree.x0ff = -0.5 / plotTree.totalW
    plotTree.y0ff = 1.0
    # plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    # plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plotTree(tree, (0.5, 1.0), '')
    plt.show()

def getNumLeafs(trees):
    numLeafs = 0
    firstStr = trees.keys()[0]
    secondDict = trees[firstStr]
    for key in secondDict.keys():
        if isinstance(secondDict[key],dict):
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(trees):
    maxDepth = 0
    firstStr = trees.keys()[0]
    secondDict = trees[firstStr]
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]
    cntrPt = (plotTree.x0ff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.y0ff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.y0ff = plotTree.y0ff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.x0ff = plotTree.x0ff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.x0ff, plotTree.y0ff), cntrPt, leafNode)
            plotMidText((plotTree.x0ff, plotTree.y0ff), cntrPt, str(key))
    plotTree.y0ff = plotTree.y0ff + 1.0 / plotTree.totalD






