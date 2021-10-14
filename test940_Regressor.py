'''逻辑回归(音译)： Logistic Regression 对数几率回归
    优点：
     1. 逻辑回归对线性关系的拟合效果丧心病狂
     2. 逻辑回归计算快
     3. 逻辑回归返回的分类结果不是固定的0或1， 而是以小数形式呈现的类概率数字。
        因此可以将逻辑回归返回的结果当成连续型数据来利用。 评分卡/信用分等。
     4. 强大的抗噪能力
    在线性数据上表现优异的分类器，逻辑回归主要被应用在金融领域。其数学目的是求解能够让模型对数据拟合程度最高的参数⍬的值，以此构建预测函数 ，
    然后将特征矩阵输入预测函数来计算出逻辑回归的结果y。注意，虽然我们熟悉的逻辑回归通常被用于处理二分类问题

    逻辑回归相关类                             说明
    linear_model.LogisticRegression          逻辑回归分类器
    linear_model.LogisticRegressionCV        带交叉验证的逻辑回归分类器
    linear_model.logistic_regression_path    (调整专用)计算logistic回归模型以获得正则化参数的列表
    linear_model.SGDClassifier                利用梯度下降求解的线性分类器（SVM,逻辑回归等等）
    linear_model.SGDRegressor                 利用梯度下降最小化正则化后的损失函数的纯属回归模型
    metrics.log_loss                          对数损失，又称逻辑损失或交叉熵损失

    metrics.confusion_matrix                  混淆矩阵，模型评估指标之一
    metrics.roc_auc_score                     ROC曲线，模型评估指标之一
    metrics.accuracy_score                    精确性，模型评估指标之一

'''
'''2. linear_model.LogisticRegression
class sklearn.linear_model.LogisticRegression (
            penalty=’l2’, 
            dual=False, 
            tol=0.0001, 
            C=1.0, 
            fit_intercept=True, 
            intercept_scaling=1, 
            class_weight=None, 
            random_state=None, 
            solver=’warn’, 
            max_iter=100, multi_class=’warn’, verbose=0, warm_start=False, n_jobs=None)
'''
'''############################'''
'''2.1 二元逻辑回归的损失函数'''
'''2.1.1 损失函数的概念与解惑'''
'''在决策树和随机森林的时候，我们曾经提到过2种模型表现：在训练集上的表现，和在测试集上的表现。我们建模是追求测试集上的表现最优
但逻辑回归中，是基于训练数据求解参数⍬的需求，并且希望训练出来的模型能够尽可能地拟合训练数据，即模型在训练集上的预测准确越靠近100%越好。ß

我们使用”损失函数“这个评估指标，来衡量参数为 的模型拟合训练集时产生的信息损失的大小，并以此衡量参数⍬的优劣。越小越好。

逻辑回归的损失函数是由极大似然估计推导出来的。
J(⍬) = ℁
'''
'''############################'''
'''2.2 重要参数 penalty & C'''
'''2.2.1 正则化
正则化是用来防止模型过拟合的过程，常用的有L1和L2正则化2种选项，分别通过在损失函数后加上参数向量theta的L1和L2范式的倍数来实现。
这个增加的范式，称为正则项，也叫惩罚项

L1正则化和L2正则化虽然都可以控制过拟合，但它们的效果并不相同。
当正则化强度逐渐增大(即C逐渐变小)， 参数theta的取值会逐渐变小，
但L1正则化会将参数压缩为0（即控制稀疏性，0越多越稀疏），L2正则化只会让参数尽量小，不会取到0（L2正则化则会比较稠密，相较于L1）。

参数
penalty    默认是l2,如果选择l1,则另一个参数solver只能选择'liblinear'或'saga'， l2可用全部solver
C:         C是正则化强度的倒数，>0,默认为1.0，即默认正则项与损失函数的比值是1：1.
           C越小，损失函数会越小，模型对损失函数(theta)惩罚越重，正则化的效力越强，参数theta会逐渐被压缩得越来越小。
'''


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


data = load_breast_cancer()
x = data.data
y = data.target

lrl1 = LR(penalty='l1', solver='liblinear', C=0.5, max_iter=1000)
lrl2 = LR(penalty='l2', solver='liblinear', C=0.5, max_iter=1000)

#逻辑回归的重要属性coef_, 查看每个特征所对应的参数
lrl1 = lrl1.fit(x,y)
##l1 正则，很多参数都被设置为0
print(lrl1.coef_)

lrl2 = lrl2.fit(x,y)
##l2正则，是对所有参数都给出了值
print(lrl2.coef_)

##那究竟哪个正则效果最好？还是都是差不多？
## 下面图像可以看出 训练集上的表现优于测试集，L2总体效果优于L1
## 测试集在c取0.8后，就会呈下降趋势，因此我们应取8左右的值
l1 = []
l2 = []
l1test = []
l2test = []
# xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3, random_state=420)
# for i in np.linspace(0.05, 1, 19):
#     lrl1 = LR(penalty='l1', solver='liblinear', C=i, max_iter=1000)
#     lrl2 = LR(penalty='l2', solver='liblinear', C=i, max_iter=1000)
#     lrl1 = lrl1.fit(xtrain, ytrain)
#     l1.append(accuracy_score(lrl1.predict(xtrain),ytrain))
#     l1test.append(accuracy_score(lrl1.predict(xtest),ytest))
#
#     lrl2 = lrl2.fit(xtrain, ytrain)
#     l2.append(accuracy_score(lrl2.predict(xtrain),ytrain))
#     l2test.append(accuracy_score(lrl2.predict(xtest),ytest))
# #graph = [l1,l2,]
# graph = [l1,l2,l1test,l2test]
# color = ["green","black","red","purple"]
# label = ["L1","L2","L1test","L2test"]
# plt.figure(figsize=(6,6))
# for i in range(len(graph)):
#     plt.plot(np.linspace(0.05,1,19),graph[i],color[i],label=label[i])
# plt.legend(loc=4) #图例的位置在哪里?4表示，右下角
# plt.show()

'''2.2.2 逻辑回归中的特征工程'''
'''当特征的数量很多的时候，出于业务考虑，希望对逻辑回归进行特征选择，来降维从而降低特征个数
    （1） PCA和SVD一般不用
    （2） 统计方法可以使用，但不推荐（方差，卡方，互信息等）
    （3） 高效的嵌入法embedded
    （4） 比较麻烦的系数累加法
    （5） 简单快速的包装法
'''
'''（3）高效的嵌入法embedded'''
from sklearn.feature_selection import SelectFromModel
LR_ = LR(solver='liblinear', C=0.9, random_state=420)
x = data.data
y = data.target
print(cross_val_score(LR_, x, y, cv=10).mean())  #0.95

x_embedded = SelectFromModel(LR_, norm_order=1).fit_transform(x, y)
print(x_embedded.shape)
print(cross_val_score(LR_, x_embedded, y, cv=10).mean()) #0.93
##看看结果，特征数量被减小到个位数，并且模型的效果却没有下降太多，
##如果我们要求不高，在这里其实就可以停了。但是，能否让模型的拟合效果更好呢?在这里，我们有两种调整方式:
## 1）调节SelectFromModel这个类的参数threshold，这是嵌入法的阈值

#### 但，下面的图像告诉我们，这种方法是比较无效的，当threshold越来越大，被删除的特征越来越多，模型效果越来越差
# fullx = []
# fsx = []
# threshold = np.linspace(0, abs((LR_.fit(x, y).coef_)).max(), 20)
# k=0
# for i in threshold:
#     x_embedded = SelectFromModel(LR_, threshold=i).fit_transform(x, y)
#     fullx.append(cross_val_score(LR_, x, y,cv=5).mean())
#     fsx.append(cross_val_score(LR_, x_embedded, y,cv=5).mean())
#     print((threshold[k],x_embedded.shape[1]))
#     k += 1
# plt.figure(figsize=(20,5))
# plt.plot(threshold,fullx,label="full")
# plt.plot(threshold,fsx,label="feature selection")
# plt.xticks(threshold)
# plt.legend()
# plt.show()

## 2）调逻辑回归的类LR_, 通过画C的学习曲线来实现：
### 由下面的图像得出：当C=【4.01，6.01】时，效果最好。
fullx = []
fsx = []
#C = np.arange(0.01, 10.01, 0.5) ##用这行代码可以得出：C=【5.01，6.01】时，效果最好。
C = np.arange(5.0, 6.0, 0.005) ##根据上面一行代码的结果，再细化，，得出C=5.73499最好
# for i in C:
#     LR_ = LR(solver='liblinear', C=i, random_state=420)
#     fullx.append(cross_val_score(LR_, x, y, cv=10).mean())
#     x_embedded = SelectFromModel(LR_, norm_order=1).fit_transform(x,y)
#     fsx.append(cross_val_score(LR_, x_embedded, y, cv=10).mean())
# print(max(fsx),C[fsx.index(max(fsx))])
# plt.figure(figsize=(20,5))
# plt.plot(C,fullx,label="full")
# plt.plot(C,fsx,label="feature selection")
# plt.xticks(C)
# plt.legend()
# plt.show()


#验证模型效果，降维之前：
LR_ = LR(solver="liblinear",C=5.734999,random_state=420)
print(cross_val_score(LR_,x,y,cv=10).mean())
#降维之后：
LR_ = LR(solver="liblinear",C=5.734999,random_state=420)
x_embedded = SelectFromModel(LR_,norm_order=1).fit_transform(x,y)
print(cross_val_score(LR_,x_embedded,y,cv=10).mean())

'''############################'''
'''2.3 梯度下降:重要参数max_iter
逻辑回归的数学目的是求解能够让模型最优化，拟合程度最好的参数的值，即求解能够让损失函数J(⍬)最小化的⍬值。
对于二元逻辑回归来说，有多种方法可以用来求解参数，
最常见的有梯度下降法(Gradient Descent)，坐标下 降法(Coordinate Descent)，牛顿法(Newton-Raphson method)等，其中又以梯度下降法最为著名。
每种方法都 涉及复杂的数学原理，但这些计算在执行的任务其实是类似的。

2.3.1 梯度下降求解逻辑回归
小球其实就是一组组的坐标点 ;小球每次滚动的方向就是那一个坐标点的梯度向量的方 向，
因为每滚动一步，小球所在的位置都发生变化，坐标点和坐标点对应的梯度向量都发生了变化，
所以每次滚动 的方向也都不一样;人为设置的100次滚动限制，
就是sklearn中逻辑回归的参数max_iter，代表着能走的最大步 数，即最大迭代次数


2.3.2 梯度下降的概念与解惑
2.3.3 步长的概念与解惑
     步长是什么？ 

'''

'''来看看乳腺癌数据集下，max_iter（步长）的学习曲线， max_iter越大，代表步长越小'''
'''下面图像可以看出max_iter最好的是21'''
# l2 = []
# l2test = []
#
# xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3, random_state=420)
# for i in np.arange(1, 201, 10):
#     lrl2 = LR(penalty='l2', solver='liblinear', C=0.9, max_iter=i)
#     lrl2 = lrl2.fit(xtrain, ytrain)
#     l2.append(accuracy_score(lrl2.predict(xtrain), ytrain))
#     l2test.append(accuracy_score(lrl2.predict(xtest), ytest))
# graph = [l2,l2test]
# color = ['black', 'gray']
# label = ['L2', 'L2test']
# plt.figure(figsize=(20,5))
#
# for i in range(len(graph)):
#     plt.plot(np.arange(1, 201, 10), graph[i], color[i], label=label[i])
# plt.legend(loc=4)
# plt.xticks(np.arange(1, 201, 10))
# plt.show()

'''############################'''
'''2.4 二元回归与多元回归： 重要参数 solver & multi_class'''
'''之前我们讨论的逻辑回归，都是针对于二元回归，但其实sklearn也支持多元回归的选项。
比如 ：  
     (1) 我们可以把某种分类类型都看作1，其余的分类类型都为0值，和'数据预处理'中的二值化思维类似，这种方法被称为One-vs-rest(1对多，ovr, sklearn = 'ovr' 0.22中，ovr改为auto)
     (2) 我们可以把好几个分类类型划分为1， 剩下的几个分类类型划分为0值，这种方法被称为Many-vs-Many(多对多， MvM, sklearn='multinomial')
     
在sklearn中， 我们需要使用参数multi_class来告诉模型，我们的预测标签是什么样的类型
    multi_class: 'ovr', 'multinomial', 'auto' 来告知模型，我们要处理的分类问题的类型，默认是'auto'(ovr)
    
                         'ovr': 表示分类问题是二分类，或让模型使用'一对多'的形式来处理多分类总是
                'multinominal': 表示处理多分类问题，这个选项在参数solver='liblinear'时不可用
                        'auto': 表示会根据分类情况和其他参数来确定模型要处理的分类问题的类型。
                                比如，如果数据是二分类，或solver='liblinear'， 'auto'默认会选择'ovr'，反之则会选择'multinominal'
                                
                                
    我们之提到的梯度下降法，只是求解逻辑回归参数⍬的一种方法，并且我们只讲解了求解二分类变量的参数时的各种原理。
    sklearn为我们提供了多种选择，让我们可以使用不同的求解器(solver)来计算逻辑回归。
    
    solver          'liblinear'     'lbfgs'                                   'newton-cg'                              'sag'                                      'saga'
                    坐标下降法       拟牛顿法的一种，利用损失函数的                牛顿法的一种，利用损失函数的                  随机平均梯度下降，与普通梯度下降法的区别是       随机平均梯度下降的进化
                                    二阶导数矩阵（海森矩阵）来迭代优化损失函数     二阶导数矩阵（海森矩阵）来迭代优化损失函数      每次迭代仅仅用一部分的样本来计算梯度             稀疏多项逻辑回归的首选
支持的回归类型：
     MvM              否(不利)         是                                       是                                         是                                          是
     OvR              是              是                                       是                                         是                                           是
     二分类            是              是                                       是                                         是                                           是
solver的效果
惩罚截距(不要惩罚截距比较好) 是(不利)      否                                      否                                         否                                           否
在大型数据集             否(不利)        否(不利)                                 否(不利)                                    是                                          是
对未标准化的数据集很有用   是             是                                       是                                         否(不利)                                     否(不利)
（数据是非正太分布）
'''

'''来看看 iris数据上，multinominal和ovr的区别'''
# from sklearn.datasets import load_iris
# iris = load_iris() #iris.target标签数据中，可以看到是3分类数据集
# for multi_class in ('multinomial', 'ovr'):
#     clf = LR(solver='sag', max_iter=100, random_state=42,
#              multi_class=multi_class).fit(iris.data, iris.target)
#     print('training score: %.3f (%s)' % (clf.score(iris.data, iris.target),multi_class))



'''############################'''
'''2.3 样本不平衡与参数class_weight'''
'''样本不平衡是指在一组数据集中，标签的一类天生占有很大比例，或分类的代价很高，即我们想要捕捉某种特定的分类的时候的状况。
什么情况下误分类的代价很高：  例如，分类潜在犯罪者和普通群众，如果没有能识别出来潜在犯罪者，那这些人可能会去危害社会。
                                             （但但如果，我们将普通人错误地识别成了潜在犯罪者，代价却相对较小。
                                             所以我们宁愿将普通人分类为潜在犯罪者后再人工甄别，但是却不愿将潜在犯罪者分类为普通人，
                                             有种"宁愿错杀不能放过"的感觉。）
                            或，银行判断一个新客户是否会违约，通常不违约的人和违约的人会是99：1
因此sklearn用参数class_weight对样本标签进行一定的均衡： 给少量的标签更多的权重，让模型更偏向少数类。
        默认为none, 即自动1：1.
        当误分类的代价很高时，使用'balanced'模式。

但是class_weigth参数变幻莫测，很难去找出这个参数引导的模型趋势，或画出学习曲线来评估参数效果，因此非常难用。
我们有着处理样本不均衡的各种方法，其中主流的是采样法，是通过重复样本的方式来平衡标签 ，可以进行上采样（增加少数类的样本），如SMOTE
或者下采样（减少多数类的样本）。

对于逻辑回归来说，上采样是最好的方法。
示例如下：
'''


'''############################'''
'''3 案例：用逻辑回归制作评分卡'''
























