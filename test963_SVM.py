'''2.3 硬间隔与软间隔:重要参数C

    2.3.1 SVM在软间隔数据上的推广

    2.3.2 重要参数C

    参数C用于权衡”训练样本的正确分类“与”决策函数的边际最大化“两个不可同时完成的目标，希望找出一个平衡点来让模型的效果最佳

    参 数    含义
      C      浮点数，默认1，必须大于等于0，可不填
             松弛系数的惩罚项系数。
                如果C值设定比较大，那SVC可能会选择边际较小的，能够更好地分类所有训 练点的决策边界，不过模型的训练时间也会更长。
                如果C的设定值较小，那SVC会尽量最大化边界，决 策功能会更简单，但代价是训练的准确度。换句话说，C在SVM中的影响就像正则化参数对逻辑回归的影响。
    在实际使用中，C和核函数的相关参数(gamma，degree等等)们搭配，往往是SVM调参的重点。
    与gamma不同，C没有在对偶函数中出现，并且是明确了调参目标的，所以我们可以明确我们究竟是否需要训练集上的高精确度来调整C的方向。
    默认情况下C为1，通常来说这都是一个合理的参数。
    如果我们的数据很嘈杂，那我们往往减小 C。当然，我们也可以使用网格搜索或者学习曲线来调整C的值。
'''
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn.datasets import make_circles,make_blobs,make_classification,make_moons,load_breast_cancer
from sklearn.model_selection import train_test_split
from time import time
import datetime
from sklearn.preprocessing import StandardScaler

#调线性核函数
data = load_breast_cancer()
x = data.data
y = data.target
x = StandardScaler().fit_transform(x)
Xtrain,Xtest,Ytrain,Ytest = train_test_split(x,y, test_size=0.3, random_state=420)
score = []
C_range = np.linspace(0.01,30,50)
for i in C_range:
    clf = SVC(kernel="linear",C=i,cache_size=5000).fit(Xtrain,Ytrain)
    score.append(clf.score(Xtest,Ytest))
print(max(score), C_range[score.index(max(score))])
plt.plot(C_range,score)
plt.show()
#换rbf
# score = []
# C_range = np.linspace(0.01,30,50) for i in C_range:
#     clf = SVC(kernel="rbf",C=i,gamma =
# 0.012742749857031322,cache_size=5000).fit(Xtrain,Ytrain)
#     score.append(clf.score(Xtest,Ytest))
# print(max(score), C_range[score.index(max(score))])
# plt.plot(C_range,score)
# plt.show()
# #进一步细化
# score = []
# C_range = np.linspace(5,7,50) for i in C_range:
#     clf = SVC(kernel="rbf",C=i,gamma =
# 0.012742749857031322,cache_size=5000).fit(Xtrain,Ytrain)
#     score.append(clf.score(Xtest,Ytest))
# print(max(score), C_range[score.index(max(score))])
# plt.plot(C_range,score)
# plt.show()