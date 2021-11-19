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

之前我们在加利佛尼亚房屋价值数据集上使用线性回归，得出的结果大概是训练集上的拟合程度是60%，测试集上的 拟合程度也是60%左右，那这个很低的拟合程度是不是由多重共线性造成的呢?
在统计学中，我们会通过VIF或者各 种检验来判断数据是否存在共线性，然而在机器学习中，我们可以使用模型来判断——如果一个数据集在岭回归中使 用各种正则化参数取值下模型表现没有明显上升(比如出现持平或者下降)，
则说明数据没有多重共线性，顶多是特 征之间有一些相关性。

反之，如果一个数据集在岭回归的各种正则化参数取值下表现出明显的上升趋势，则说明数据 存在多重共线性。

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
Xtrain, Xtest, Ytrain, Ytest = TTS(X,y, test_size=0.3, random_state=420)
for i in [Xtrain, Xtest]: #reset index
    i.index = range(i.shape[0])

#使用岭回归来进行建模
reg = Ridge(alpha=1).fit(Xtrain, Ytrain)
reg.score(Xtest, Ytest)

#交叉验证下，与线性回归相比，岭回归的结果如何变化？
#从以下的图可以看出，加利佛尼亚数据集上，岭回归的结果对比线性回归有轻微上升，随后骤降。
# 可以说，加利佛尼亚房屋价值数据集带有很轻的一部分共线性，这种共线性被正则化参数α消除后，
# 模型的效果提升了一点点，但是对于整个模型而言是杯水车薪
# alpharange = np.arange(1, 1001, 100)
# ridge, lr = [], []
# for alpha in alpharange:
#     reg = Ridge(alpha=alpha)
#     linear = LinearRegression()
#     regs = cross_val_score(reg,X, y , cv=5, scoring='r2').mean()
#     linears = cross_val_score(linear, X, y,cv=5, scoring='r2').mean()
#     ridge.append(regs)
#     lr.append(linears)
# plt.plot(alpharange, ridge, color='red', label='Ridge')
# plt.plot(alpharange, lr, color='black', label='LR')
# plt.title('mean')
# plt.legend()
# plt.show()

#另外，在正则化参数逐渐增大的过程中，我们可以观察一下模型的方差如何变化
# alpharange = np.arange(1, 1001, 100)
# ridge, lr = [], []
# for alpha in alpharange:
#     reg = Ridge(alpha=alpha)
#     linear = LinearRegression()
#     varR = cross_val_score(reg,X,y,cv=5,scoring='r2').var()
#     varLR = cross_val_score(linear,X,y,cv=5,scoring='r2').var()
#     ridge.append(varR)
#     lr.append(varLR)
# plt.plot(alpharange, ridge, color='red', label='Ridge')
# plt.plot(alpharange, lr, color='black', label='LR')
# plt.title('Variance')
# plt.legend()
# plt.show()

'''
遗憾的是，没有人会希望自己获取的数据中存在多重共线性，因此发布到scikit-learn或者kaggle上的数据基本都经过一定的多重共线性的处理的，
要找出绝对具有多重共线性的数据非常困难，也就无法给大家展示岭回归在实际数据中 大显身手的模样。
我们也许可以找出具有一些相关性的数据，
但是大家如果去尝试就会发现，基本上如果我们使用岭 回归或者Lasso，那模型的效果都是会降低的，很难升高，
这恐怕也是岭回归和Lasso一定程度上被机器学习领域冷 遇的原因。
'''

'''4.2.3 选取最佳的正则化参数取值 正则化参数α
虽然上面代码用cross_val_score交叉验证来选取α，但对了岭回归，学习教材中，都推荐使用岭迹图来判断α的最佳取值

但不推荐使用岭迹图来找alpha的最佳取值，还是得用交叉验证
'''
# from sklearn import linear_model
# #创造10*10的希尔伯特矩阵
# X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
# y = np.ones(10)
# #计算横坐标
# n_alphas = 200
# alphas = np.logspace(-10, -2, n_alphas)
# #建模，获取每一个正则化取值下的系数组合
# coefs = []
# for a in alphas:
#     ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
#     ridge.fit(X, y)
#     coefs.append(ridge.coef_)
# #绘图展示结果
# ax = plt.gca()
# ax.plot(alphas, coefs)
# ax.set_xscale('log')
# ax.set_xlim(ax.get_xlim()[::-1]) #将横坐标逆转
# plt.xlabel('正则化参数alpha')
# plt.ylabel('系数w')
# plt.title('岭回归下的岭迹图')
# plt.axis('tight')
# plt.show()

from sklearn.linear_model import RidgeCV, LinearRegression
housevalue = fch()
X = pd.DataFrame(housevalue.data)
y = housevalue.target
X.columns = ["住户收入中位数", "房屋使用年代中位数", "平均房间数目", "平均卧室数目", "街区人口", "平均入住率", "街区的纬度", "街区的经度"]
Ridge_ = RidgeCV(alphas=np.arange(1, 1001, 100)
                 # ,scoring="neg_mean_squared_error"
                 , store_cv_values=True
                 #,cv=5
                 ).fit(X, y)


# 无关交叉验证的岭回归结果
print(Ridge_.score(X,y))
# 调用所有交叉验证的结果
print(Ridge_.cv_values_.shape)
# 进行平均后可以查看每个正则化系数取值下的交叉验证结果
print(Ridge_.cv_values_.mean(axis=0))
# 查看被选择出来的最佳正则化系数
print(Ridge_.alpha_)


