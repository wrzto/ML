#-*-coding:utf-8-*-

import numpy as np

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataset):
    vocabSet = set([])
    for doc in dataset:
        vocabSet = vocabSet | set(doc)
    return list(vocabSet)

####词集模型
def setOfWords2Vec(vocabList, inputSet):
    ####为每篇文章生成词集模型
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec

####词袋模型
def bagOfWords2Vec(vocabList, inputSet):
        ####为每篇文章生成词袋模型
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def trainNB0(dataset, labels):
    ###dataset :每一行类似词汇表的矩阵，由setOfWords2Vec(),bagOfWords2Vec()生成，判断文件中的词在词汇表是否出现
    if not isinstance(dataset, np.ndarray):
        dataset = np.array(dataset)
    numTrainDocs = len(dataset)
    numWords = len(dataset[0])
    pC1 = sum(labels) / float(len(labels))
    # pC0WordNum = np.zeros(numWords)
    # pC1WordNum = np.zeros(numWords)
    # pC0WordCount = 0.0
    # pC1WordCount = 0.0
    ###防止某个概率值为0导致所有概率相乘为0
    pC0WordNum = np.ones(numWords)
    pC1WordNum = np.ones(numWords)
    pC0WordCount = 2.0
    pC1WordCount = 2.0
    for i in range(numTrainDocs):
        if labels[i] == 1:
            pC1WordNum += dataset[i]
            pC1WordCount += np.sum(dataset[i])
        else:
            pC0WordNum += dataset[i]
            pC0WordCount += np.sum(dataset[i])
    ####预防下溢出,由于太多过小的数相乘导致得不到正确答案
    ####ln(ab) = ln(a) + ln(b) 通过求对数可以避免 ， f(x) 与 ln(f(x)) 图像走势是一致的 
    pC0Vect = np.log(pC0WordNum / pC0WordCount)
    pC1Vect = np.log(pC1WordNum / pC1WordCount)
    return pC0Vect, pC1Vect, pC1

def classifyNB(docVec, pC0Vect, pC1Vect, pC1):
    ###docVec 由setOfWords2Vec(),bagOfWords2Vec()生成
    if not isinstance(docVec, np.ndarray):
        docVec = np.array(docVec)
    ###p = p(w0|ci) * p(w1|ci) * ... * p(wn|ci) * p(ci) / p(w0,w1,...wn)
    ###ln(p(w0|ci) * p(w1|ci) * ... * p(wn|ci) * p(ci)) = ln(p(w0|ci)) + ln(p(w1|ci)) + ... + ln(p(wn|ci)) + ln(p(ci))
    p1 = np.sum(docVec * pC1Vect) + np.log(pC1)
    p0 = np.sum(docVec * pC0Vect) + np.log(1 - pC1)
    if p1 > p0:
        return 1
    return 0

def testingNB():
    dataset, labels = loadDataSet()
    vocabList = createVocabList(dataset)
    docVects = []
    for data in dataset:
        docVects.append(setOfWords2Vec(vocabList, data))
    pC0Vect, pC1Vect, pC1 = trainNB0(docVects, labels)

    testDocs = [['love', 'my', 'dalmation'],
                ['stupid', 'garbage']]
    for testDoc in testDocs:
        print testDoc,'classified as: ',classifyNB(setOfWords2Vec(vocabList, testDoc), pC0Vect, pC1Vect, pC1)

def textParse(text):
    import re
    listOfTokens = re.split(r'\W*', text)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]
    # import jieba
    # listOfTokens = jieba.cut(text)
    # return [tok.lower() for tok in listOfTokens if len(tok) > 2]


###实例一:判断垃圾邮件
def testingEmail():
    import os
    from random import sample
    spamDir = 'email\\spam'
    hamDir = 'email\\ham'
    ####从文件导入数据
    spamDocList = []
    hamDocList = []
    spamFileList = os.listdir(spamDir)
    hamFileList = os.listdir(hamDir)
    for filename in spamFileList:
        spamDocList.append(textParse(open(os.path.join(spamDir,filename)).read()))
    for filename in hamFileList:
        hamDocList.append(textParse(open(os.path.join(hamDir, filename)).read()))
    error_accuary = 0.0
    ###随机选取10次数据做训练
    for i in range(10):
        ###选择70%的数据构建训练集
        trainDocList = []
        trainDocLabel = []
        testDocList = []
        testDocLabel = []
        index = sample(range(len(spamDocList)), int(len(spamDocList) * 0.8))
        for i in range(len(spamDocList)):
            if i in index:
                trainDocList.append(spamDocList[i])
                trainDocLabel.append(1)
            else:
                testDocList.append(spamDocList[i])
                testDocLabel.append(1)
        index = sample(range(len(hamDocList)), int(len(hamDocList) * 0.8))
        for i in range(len(hamDocList)):
            if i in index:
                trainDocList.append(hamDocList[i])
                trainDocLabel.append(0)
            else:
                testDocList.append(hamDocList[i])
                testDocLabel.append(0)
        vocabList = createVocabList(trainDocList)
        trainDocVec = [setOfWords2Vec(vocabList, doc) for doc in trainDocList]
        pC0Vect, pC1Vect, pC1 = trainNB0(trainDocVec, trainDocLabel)
        testDocVec = [setOfWords2Vec(vocabList, doc) for doc in testDocList]
        result_labels = []
        for docVec in testDocVec:
            result_labels.append(classifyNB(docVec, pC0Vect, pC1Vect, pC1))
        error = 0
        for res in zip(result_labels, testDocLabel):
            if res[0] != res[1]:
                error += 1
        error_accuary += error / float(len(result_labels))
    print 'error accuary: %f' % (error_accuary / 10)
    
    
###实例2 
def delFreqWords(vocabList, fullText, num=30):
    from collections import Counter
    wordCounts = Counter(fullText)
    removeCount = 0
    for wordCount in wordCounts.most_common():
        if wordCount[0] in vocabList and removeCount < num:
            vocabList.remove(wordCount[0])
            removeCount += 1
    return vocabList

def localWords(feed1, feed0):
    from random import sample
    num = len(feed0['entries']) if len(feed1['entries']) > len(feed0['entries']) else len(feed1['entries'])
    trainDocList = []
    trainDocLabel = []
    testDocList = []
    testDocLabel = []
    fulltext = []

    index = sample(range(num), int(num * 0.7))
    for i in range(num):
        if i in index:
            trainDocList.append(textParse(feed1['entries'][i]['summary']))
            trainDocLabel.append(1)
        else:
            testDocList.append(textParse(feed1['entries'][i]['summary']))
            testDocLabel.append(1)
        fulltext.extend(textParse(feed1['entries'][i]['summary']))

    index = sample(range(num), int(num * 0.7))
    for i in range(num):
        if i in index:
            trainDocList.append(textParse(feed0['entries'][i]['summary']))
            trainDocLabel.append(0)
        else:
            testDocList.append(textParse(feed0['entries'][i]['summary']))
            testDocLabel.append(0)
        fulltext.extend(textParse(feed0['entries'][i]['summary']))
    
    vocabList = createVocabList(trainDocList)
    print 'vocabList---0---:%d ' % (len(vocabList))
    vocabList = delFreqWords(vocabList, fulltext)
    print 'vocabList---1---:%d ' % (len(vocabList))

    trainDocVec = [bagOfWords2Vec(vocabList, doc) for doc in trainDocList]
    testDocVec = [bagOfWords2Vec(vocabList, doc) for doc in testDocList]

    pC0Vect, pC1Vect, pC1 = trainNB0(trainDocVec, trainDocLabel)

    result_labels = []
    for docVec in testDocVec:
        result_labels.append(classifyNB(docVec, pC0Vect, pC1Vect, pC1))

    error = 0
    for res in zip(result_labels, testDocLabel):
        if res[0] != res[1]:
            error += 1
    print 'error accuary : %f' % (error / float(len(result_labels)))
    return vocabList, pC0Vect, pC1Vect

def testing():
    url1 = 'http://newyork.craigslist.org/stp/index.rss'
    url2 = 'http://sfbay.craigslist.org/stp/index.rss'
    import feedparser
    feed1 = feedparser.parse(url1)
    feed0 = feedparser.parse(url2)
    vocabList, pC0Vect, pC1Vect = localWords(feed1, feed1)
    import operator
    topNY = []
    topSF = []
    for i in range(len(pC0Vect)):
        if pC0Vect[i] > -5.0:
            topSF.append((vocabList[i], pC0Vect[i]))
        if pC1Vect[i] > -5.0:
            topNY.append((vocabList[i], pC1Vect[i]))
    sortedSF = sorted(topSF, key=lambda pair:pair[1] , reverse=True)
    sortedNY = sorted(topNY, key=lambda pair:pair[1] , reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF"
    for item in sortedSF:
        print item[0]
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NF"
    for item in sortedNY:
        print item[0] 
    print 'SF : %d' % len(sortedSF)
    print 'NY : %d' % len(sortedNY)
    print 'all: %d' % len(vocabList)        











