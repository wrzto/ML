线性回归: 

![](file:img/linear.png)

参数:[n,1]

X:[n,m]

y:[m,1]

m为样本数

![](file:img/linear1.png)

误差ε独立同分布，服从均值为0，方差为某定值σ^2的高斯分布。随机现象可以看作众多因素的独立影响的综合反应，往往近似服从正太分布。

![](file:img/linear2.png)

其极大似然函数:

![](file:img/linear3.png)

由于该函数是累乘的，所以取对数转化为加法。

![](file:img/linear4.png)

我们的目标是求其极大似然函数的最大值，根据以上公式转化为求![](file:img/linear5.png)的极小值，从而推导出最小二乘法。

![](file:img/linear6.png)

对其求梯度:

![](file:img/linear7.png)

求解出参数:
![](file:img/linear8.png)

通过上述方法可快速求出参数的解，前提是![](file:img/linear9.png)可逆。如果其不可逆可以加入扰动因子使其可逆(同时可以防止过拟合):

![](file:img/linear10.png)

注:上述梯度计算涉及的矩阵求导知识

![](file:img/linear11.png)


为目标函数增加复杂度惩罚因子(抑制过拟合)

L1-norm

![](file:img/linear12.png)

L2-norm

![](file:img/linear13.png)
本质假定参数服从高斯分布。

梯度下降算法求解参数

![](file:img/linear14.png)

![](file:img/linear15.png)
m为样本数，n为参数的个数

写成向量的形式:

![](file:img/linear16.png)

梯度下降方法:

1.批量梯度下降(每次更新使用所有样本)，该方法可收敛至全局最小值(更新速率不能太大),目标函数必须是凸函数。但是当样本数量较大时，计算较慢。

2.随机梯度下降(每次更新使用一个样本)，该方法计算速度快，但是较难收敛到极小值，收敛至极小值附近，可跳出局部极小值,适合在线学习。

3.mini-batch梯度下降(每次更新选取固定数量的样本的平均梯度)

LogisticRegression(用于解决分类问题)

![](file:img/linear19.png)

![](file:img/linear17.png)

其Sigmoid函数图像为:

![sigmoid函数](file:img/linear18.png)

与线性回归不同，它将输出值压缩在[0,1]作为概率输出。

Sigmoid函数求导:

![](file:img/linear20.png)

Logistic回归参数估计

![](file:img/linear21.png)

乘性公式取对数转化为加性公式,则其对数似然函数:

![](file:img/linear22.png)

其对数似然函数就是我们的loss函数

对其求导得:

![](file:img/linear23.png)












