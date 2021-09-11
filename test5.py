'''模型的评估
评估模型的好坏：除了使用Estimator的score函数简单粗略地评估模型的质量之外，
在sklearn.metrics模块针对不同的问题类型提供了各种评估指标并且可以创建用户自定义的评估指标。
使用sklearn.model_selection模块中交叉验证相关方法可以评估模型的泛化能力，能够有效避免过度拟合。
一、metrics评估指标概述
sklean.metrics中的评估指标有两类：以＿score结尾的为某种得分，越大越好／以＿error或＿1oss結尾的是某种偏差，越小越好
常用的分类模型评估指标：accuracy_score,f1_score,precision_score,recall_score等
常用的回归模型评估指标：r2_score,explained_variance_score等
常用的聚类模型评估指标：adjusted_rand_score,adjusted_mutual_info_score等
'''
'''1.1分类模型评估指标'''
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
y_pred = [0,0,0,1,1,1,1,1]
y_true = [0,1,0,1,1,0,0,1]
#print(metrics.confusion_matrix(y_true,y_pred))
#print('accuracy:',metrics.accuracy_score(y_true,y_pred))
#print('precision:',metrics.precision_score(y_true,y_pred,average-None))
'''1.2回归模型评估指标
回归模型最常用的评估指标有：
1.r2_score(R方，拟合优度，可决系数）
2.explained_variance_score(解释方差得分）
'''
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
y_true=[3,-0.5,2,7] ##真实数据值
y_pred=[2.5,0.0,2,8]##预测值
#print('explained_variance_score:',explained_variance_score(y_true,y_pred))
#print('r2_score:',r2_score(y_true,y_pred))
'''·使用虚拟估计器产生基准得分
对于监督学习（分类和回归），可以用一些基于经验的简单估计策略（虚拟估计）的得分作为参照基准值
DummyClassifier实现了几种简单的分类策略
（1)stratified 通过在训练集类分布方面来生成随机预测
（2)most_frequent 总是预测训练集中最常见的标签
（3)prior 类似most_frequent,但具有precit_proba方法
（4)uniform 随机产生预测
（5)constant 总是预测用户提供的常量标签
DummyRegressor实现了4个简单的经验法则来进行回归：
（1)mean 总是预测训练目标的平均值
（2)median 总是预测训练目标的中位数
 (3) quantile 总是预测用户提供的训练目标的quntile(分位数）
  (4) constant 总是预测用户提供的常数值
  '''
from sklearn.model_selection import train_test_split
#创建一个不平衡的数据集
iris = datasets.load_iris()
x,y=iris.data,iris.target
y[y!=1]--1 #将0,2类别合并成－1类别，数据集就只有1/-1
x_train,x_test,y_train,y_test= train_test_split(x,y,random_state=0)
from sklearn.dummy import DummyClassifier
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVC
svc = SVC(kernel='linear',C=1).fit(x_train,y_train)
#print('linear svc classifier score:',svc.score(x_test,y_test))
dummy = DummyClassifier(strategy = 'most_frequent',random_state = 0)
dummy.fit(x_train,y_train)
#print('dummy classifier score:',dummy.score(x_test,y_test))
svc_rbf = SVC(kernel = 'rbf',C = 1).fit(x_train,y_train)
#print('rbf kernel svc classifier score:',svc_rbf.score(x_test,y_test))


