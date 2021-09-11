'''欠拟合，过拟合与交叉验证
在机器学习中，经常会出现模型在训练集数据上的得分很高，但在新的数据上表现很差的情况，这称之为过拟合over fitting,又叫高方差high variance = > 解决方向：使用正则方法，降低模型复杂度
而如果在训练数据上得分就很低，这称之为欠拟合under fitting,又叫高偏差 high bias => 解决方法：增加训练集模型数据，增加模型复杂度
验证是否出现拟合情况，大致有以下2种方法：
1.留出法，在数据集，使用 train_test_split,除了留出test_size,还要留出validation_size
        缺点：数据量可能就不够了
2.交叉验证：可以验证是否过拟合／欠拟合
    在数据量有限时，按留出法将数据分成3部分将会严重影响到模型训练的效果。为了有效利用有限的数据，可以采用交叉验证cross_validation方法
    其基本思想是：以不同的方式多次将数据集划分为训练集和测试集，分别训练和测试。
    简单的2折交叉验证：把数据划分成A,B组，先用A组训练B组测试，再用B组训练A组测试。所以叫交叉验证

常用的交叉验证方法：KFold(k折），LeaveOneOut,LOO(留1交叉验证），LeavePOut,LPO(留P交叉验证），RepeatKFold(重复K折交叉验证），ShuffleSp1it(随机排列交叉验证）

此外，为了保证训练集中每种标签类别数据的分布和完整数据集中的分布一致，还可以采用交叉验证方法（StratifiedKFold,StratifiedShuff1e5plit)
    当数据集的来源有不同的分组时，独立同分布假设（independent identical distributed:1.1.d)得被打破，
    可以使用分组交叉验证方法来确保测试集合中的所有样本来自训练样本中没有表示过的新的分组
    （GroupKFold,LeavePGroupsOut,LeavePGroupOut,GroupShuffleSplit)

    对于时间序列数据，一个非常重要的特点是时间相信的观测之间的相关性（自相关性），因此用过去的数据训练而用未来的数据测试非常重要。TimeSeriesSplit可以实现这样的分割。
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn import metrics
# ShuffleSplit(随机排列交叉验证）
from sklearn.model_selection import ShuffleSplit
x = np.arange(5)
ss = ShuffleSplit(n_splits=4,test_size=0.4,random_state=0)
# for train_index,test_index in ss.split(x):
#       print('%s,%s' % (train_index,test_index))
#StratifiedKFold(分层K折交叉验证）
from sklearn.model_selection import StratifiedKFold
x=np.ones(10)
y=[0,0,0,0,1,1,1,1,1,1]
skf=StratifiedKFold(n_splits=3,shuffle=False)
# for train index, test_index in skf.split(x,y):
#   print('%s,xs'％（train_index,test_index))
#LeavePGroupsOut 留P分组交叉验证
from sklearn.model_selection import LeavePGroupsOut

x=np.arange(6)
y=[1,1,1,2,2,2]
groups=[1,1,2,2,3,3]
lpgo=LeavePGroupsOut(n_groups=1)
# for train_index,test_index in 1pgo.split(x,y,groups=groups):
# print("%s,%s'%(train_index,test_index))
# 时间序列分割
# 保证测试集是在训练集之后的序列，即如果训练集index为【3,4,5],那测试集就在［6,7,8]
from sklearn.model_selection import TimeSeriesSplit
x=np.array([[1,2],[3,4],[1,2],[3,4],[1,2],[3,4],[2,2],[4,6]])
y=np.array([1,2,3,4,5,6,7,8])
tscv=TimeSeriesSplit(n_splits=3,max_train_size=3)
# for train_index, test_index in tscv.split(x,y):
# print('%s,%s' % (train_index,test_index))
'''交叉验证综合评分
调用cross_val_score函数可以计算模型在各交叉验证数据集上的得分。
可以指定metrics中的打分函数，也可以指定交叉验证迭代器
'''
from sklearn.model_selection import cross_val_score
from sklearn import svm
iris=datasets.load_iris()
clf=svm.SVC(kernel='linear',C=1)
scores=cross_val_score(clf,iris.data,iris.target,cv=5)##采用5折交叉验证
#print(scores)
#平均得分 和95%的置信区间
#print('Accuracy:%0.2f (+/=%0.2f)'%(scores.mean(),scores.std()*2))
#默认情况下，每个CV迭代计算的分数是估计器的score方法
#可以通过scoring参数来改变计算方式：
scores=cross_val_score(clf,iris.data,iris.target,cv=5,scoring='f1_macro')
#print(scores)
#通过传入一个交叉验证迭代器来指定其他交叉验证策略
from sklearn.model_selection import ShuffleSplit
n_samples=iris.data.shape[0]
ss=ShuffleSplit(n_splits=3, test_size=0.3,random_state=0)
cross_val_score(clf,iris.data,iris.target,cv=ss)
##cross_validate函数和cross_val_score函数类似，但功能更为强大，它允许指定多个指标进行评估
##并且返回指定的指标外，还会返回一个fit_time 和score_time即训练时间和评分时间
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
clf=svm.SVC(kernel='linear',C=1,random_state=0)
scores=cross_validate(clf,iris.data,iris.target,scoring=['f1_macro','f1_micro'],
                    cv=10,return_train_score=False)
#print(sorted(scores.keys()))
#print(scores['fit_time'])
#print(scores['score_time'])
#print(scores['test_f1_macro'])
#print(scores['test_f1_micro'])


