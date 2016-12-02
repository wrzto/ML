#-*-coding:utf-8-*-

import numpy as np

'''
boosting是通过集中关注被已有分类器错分的数据来获取新的分类器
p(wrong) = nums of wrong / total_nums
alpha = 1/2 * ln((1-p(wrong))/p(wrong))
'''
def loadSimpData():
    datMat = np.matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

##迭代所有特征和其对应阈值，寻找使得error最小的特征及其阈值
def buildStump(dataArr,classLabels,D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m,n = dataMatrix.shape
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m,1)))
    minError = np.inf 
    for i in range(n):
        rangeMin = np.min(dataMatrix[:,i])
        rangeMax = np.max(dataMatrix[:,i])
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt', 'gt']: 
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr 
                print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

# dataset,label = loadSimpData()
# # D = np.mat(np.ones((5,1))/5)
# # print buildStump(dataset,label,D)

#迭代训练单层决策树
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = np.mat(dataArr).shape[0]
    D = np.mat(np.ones((m,1))/m)
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print "D:",D.T
        alpha = float(0.5 * np.log((1.0 - error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print "classEst: ",classEst.T
        ##由于样本是由-1和1分类的，这里expon的结果是负的表明样本正确分类，正表示错误分类
        expon = np.multiply(-1.0 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        print 'aggClassEst: ', aggClassEst.T
        ##计算错误率，即分错的样本数/总样本数
        ##符号不同表示分类错误，标记为1
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m,1)))
        errorRate = aggErrors.sum()/m
        print "total error:", errorRate, "\n"
        if errorRate == 0.0:
            break
    return weakClassArr

# classifierArr = adaBoostTrainDS(dataset, label, 30)

##分类函数
def adaClassify(dataset, classifierArr):
    dataMat = np.mat(dataset)
    m = dataMat.shape[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMat, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print aggClassEst
    return np.sign(aggClassEst)


##实例一
def loadData(filename):
    with open(filename, 'r') as f:
        datasets = []
        labels = []
        for line in f.readlines():
            line = line.strip().split('\t')
            line = map(lambda x: float(x), line)
            labels.append(line.pop())
            datasets.append(line)
    return datasets,labels

train_dataset, train_labels = loadData('horseColicTraining2.txt')
classifierArr = adaBoostTrainDS(train_dataset, train_labels, numIt=50)
test_dataset, test_labels = loadData('horseColicTest2.txt')
predict_labels = adaClassify(test_dataset, classifierArr)

error = 0
for res in zip(test_labels, predict_labels):
    if res[0] != res[1]:
        error += 1
print 'error accuary: ',float(error) / len(test_labels)








