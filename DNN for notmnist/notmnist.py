
# coding: utf-8

# In[67]:

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle


# In[68]:

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


# In[69]:

image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# In[70]:

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


# In[63]:

batch_size = 128

graph = tf.Graph()
with graph.as_default():
    ###数据初始化
  tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  hidden_node_count = 784
 # 变量初始化
  weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_node_count]))
  biases1 = tf.Variable(tf.zeros([hidden_node_count]))

  weights2 = tf.Variable(tf.truncated_normal([hidden_node_count, num_labels]))
  biases2 = tf.Variable(tf.zeros([num_labels]))
  
  # 训练
  ys = tf.matmul(tf_train_dataset, weights1) + biases1
  hidden = tf.nn.relu(ys)
  #   hidden = tf.nn.dropout(hidden, 0.2)
  logits = tf.matmul(hidden, weights2) + biases2
  #计算损失函数
  beta = 0.002
  l2_loss = tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2) + tf.nn.l2_loss(biases1) + tf.nn.l2_loss(biases2) 
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))  + beta * l2_loss
   # 优化
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # 预测训练集，测试集，验证集
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1), weights2) + biases2)
  test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1), weights2) + biases2)


# In[64]:

num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


# In[100]:

def train_dnn(dropout = False, regular = False, lrd = False, num_steps = 3001):
    batch_size = 128

    graph = tf.Graph()
    with graph.as_default():
        # 数据初始化
        tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
        hidden_node_count = 1024
        # 变量初始化
        weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_node_count]))
        biases1 = tf.Variable(tf.zeros([hidden_node_count]))

        weights2 = tf.Variable(tf.truncated_normal([hidden_node_count, num_labels]))
        biases2 = tf.Variable(tf.zeros([num_labels]))

        # 训练
        ys = tf.matmul(tf_train_dataset, weights1) + biases1
        hidden = tf.nn.relu(ys)
        hidden_drop = hidden
        ##开启drop out
        keep_prob = 0.5
        if dropout:
            hidden_drop = tf.nn.dropout(hidden, keep_prob)

        logits = tf.matmul(hidden_drop, weights2) + biases2
        # 计算损失函数
        beta = 0
        l2_loss = tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2) + tf.nn.l2_loss(biases1) + tf.nn.l2_loss(biases2)
        # l2加入正则化
        if regular:
            beta = 0.002
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))  + beta * l2_loss
        # 优化
        if lrd:
            cur_step = tf.Variable(0)
            starter_learning_rate = 0.1
            learning_rate = tf.train.exponential_decay(starter_learning_rate, cur_step, 10000, 0.96, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=cur_step)
        else:
            optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)


        # 预测训练集，测试集，验证集
        train_prediction = tf.nn.softmax(tf.matmul(hidden, weights2) + biases2)
        valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1), weights2) + biases2)
        test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1), weights2) + biases2)

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels))
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


# In[101]:

train_dnn(dropout = True, regular=True, lrd= True, num_steps = 2751)


# In[116]:

def train_dnns(dropout=False, regular=False, lrd=False,layer = 2):
    #设置每批训练数据
    batch_size = 128
    num_labels = 10
    image_size = 28
    with graph.as_default():
        ###数据集初始化
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size*image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
        ###神经元个数初始化
        hidden_node_count = image_size * image_size
        ###权重初始化
        w0 = tf.Variable(tf.truncated_normal([image_size*image_size, hidden_node_count]))
        b0 = tf.Variable(tf.zeros([hidden_node_count]))
        w_last= tf.Variable(tf.truncated_normal([hidden_node_count, num_labels]))
        b_last = tf.Variable(tf.zeros([num_labels]))
        weights = []
        biases = []
        ###初始化隐层权重
        for i in range(layer - 2):
            weights.append(tf.Variable(tf.truncated_normal([hidden_node_count, hidden_node_count])))
            biases.append(tf.Variable(tf.zeros([hidden_node_count])))
        ###训练集计算第一层
        y0 = tf.matmul(tf_train_dataset, w0) + b0
        hidden = tf.nn.relu(y0)
        hidden_drop = hidden
        keep_prob = 0.5
        if dropout:
            hidden_drop = tf.nn.dropout(hidden, keep_prob)
        ###验证集计算第一层
        valid_y0 = tf.matmul(tf_valid_dataset,w0) + b0
        valid_hidden = tf.nn.relu(valid_y0)
        ###测试集计算第一层
        test_y0 = tf.matmul(tf_test_dataset, w0) + b0
        test_hidden = tf.nn.relu(test_y0)
        ###计算中间层
        if len(weights) > 0:
            for i in range(len(weights)):
                hidden = tf.nn.relu(tf.matmul(hidden, weights[i]) + biases[i])
                valid_hidden = tf.nn.relu(tf.matmul(valid_hidden, weights[i]) + biases[i])
                test_hidden = tf.nn.relu(tf.matmul(test_hidden, weights[i]) + biases[i])
                hidden_drop = tf.nn.relu(tf.matmul(hidden_drop, weights[i]) + biases[i])
                if dropout:
                    keep_prob += 0.5 * i / (layer + 1)
                    hidden_drop = tf.nn.dropout(hidden_drop, keep_prob)
        ###计算最后一层
        logits = tf.matmul(hidden_drop, w_last) + b_last
        ###正则化
        beta = 0
        l2_loss = 0
        if regular:
            beta = 0.002
            l2_loss += tf.nn.l2_loss(w0) + tf.nn.l2_loss(w_last) + tf.nn.l2_loss(b0) + tf.nn.l2_loss(b_last)
            if len(weights) > 0:
                for i in range(len(weights)):
                    l2_loss += tf.nn.l2_loss(weights[i])
                    l2_loss += tf.nn.l2_loss(biases[i])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + beta * l2_loss

        ###添加学习率衰减
        if lrd:
            cur_step = tf.Variable(0, trainable=True)
            starter_learning_rate = 0.4
            learning_rate = tf.train.exponential_decay(starter_learning_rate, cur_step, 100000, 0.96, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=cur_step)
        else:
            optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        train_prediction = tf.nn.softmax(tf.matmul(hidden, w_last) + b_last)
        valid_prediction = tf.nn.softmax(tf.matmul(valid_hidden, w_last) + b_last)
        test_prediction = tf.nn.softmax(tf.matmul(test_hidden, w_last) + b_last)
    num_steps = 3001

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels))
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


# In[119]:

train_dnns(dropout=False,regular=False,lrd=False,layer=2)

