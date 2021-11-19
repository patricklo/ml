'''多重共线性 Multillinearity
   指的是矩阵行与行之间存在精确线性关系 或 高度相关关系
    矩阵A第1行 * 2 = 第3行  存在精确线性关系
    矩阵B第1行 * 2 ~ 第3行  差异在0.0002，存在高度相关关系


   A    1   1   2         B   1   1   2         C    1   1   2
        5   3   11            5   3   11             5   3   11
        2   2   4             2   2   4.0002         2   2   17

在线性模型中，除了线性回归之外，最知名的就是岭回归 和lasso。
有些允许多重共线性的算法：岭回归、lasso， 为了修复多重共线性而产生
    4.2 岭回归

    4.2.1 岭回归解决多重共线性问题
         又称为吉洪诺夫正则化（Tiknonov regularization)


         linear_model.Ridge

    4.3 Lasso
        全称最小绝对收缩和选择算子

    4.3.1 Lasso与多重共线性
        Lasso无法解决特征之间”精确相关“的问题
        Lasso不是从根本上解决多重共线性问题，而是限制多重共线性带来的影响。

'''

import numpy as np
import  pandas as pd
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.model_selection import train_test_split as TTS,cross_val_score
from sklearn.datasets import fetch_california_housing as fch
import matplotlib.pyplot as plt

housevalue = fch()

X = pd.DataFrame(housevalue.data)
y = housevalue.target
X.columns = ["住户收入中位数","房屋使用年代中位数","平均房间数目"
,"平均卧室数目","街区人口","平均入住率","街区的纬度","街区的经度"]

Xtrain, Xtest, Ytrain, Ytest = TTS(X, y, test_size=0.3, random_state=420)

for i in [Xtrain, Xtest]:
    i.index = range(i.shape[0])

##线性回归进行拟合
reg = LinearRegression().fit(Xtrain, Ytrain)
print((reg.coef_*100).tolist())

##岭回归进行拟合
Ridge_ = Ridge(alpha=0).fit(Xtrain, Ytrain)
print((Ridge_.coef_*100).tolist())

##Lasso进行拟合
##以下代码会出错
##1. 正则化系数alpha不能为0， 不然算法不可收敛
#lasso_ = Lasso(alpha=0).fit(Xtrain, Ytrain)

lasso_ = Lasso(alpha=0.01).fit(Xtrain, Ytrain)
print((lasso_.coef_*100).tolist())


##逐步加大正则化系数alpha

#加大正则项系数，观察模型的系数发生了什么变化
Ridge_ = Ridge(alpha=10**4).fit(Xtrain,Ytrain)

##10**4 求出的coef_都是0 ，效果并不好，因此对于Lasso来说10**4的值太大
##证明lasso对alpha的值比较敏感
#lasso_ = Lasso(alpha=10**4).fit(Xtrain,Ytrain)
lasso_ = Lasso(alpha=1).fit(Xtrain,Ytrain)
#将系数进行绘图
plt.plot(range(1,9),(reg.coef_*100).tolist(),color="red",label="LR")
plt.plot(range(1,9),(Ridge_.coef_*100).tolist(),color="orange",label="Ridge")
plt.plot(range(1,9),(lasso_.coef_*100).tolist(),color="k",label="Lasso")
plt.plot(range(1,9),[0]*8,color="grey",linestyle="--")
plt.xlabel('w') #横坐标是每一个特征所对应的系数
plt.legend()
plt.show()
