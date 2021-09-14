'''模型特定交叉验证
一些特定的模型，sklearn构建了一些内部含有交叉验证（后面带CV）优化机制的估计器estimator
它们主要在linear_model模块中：，优先选择后面带CV
    ElasticNetCV
    LogisticRegressionCV
    RidgeCV等
'''

from sklearn import datasets
boston = datasets.load_boston()

#从样本中选出2/3作为训练集， 1/3个作为测试集，并打扰数据集
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(boston.data, boston.target,
                                                 test_size=1/3)
#print(len(y_train),len(x_train))

#特征极差标准化
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
scaler.fit(x_train)
x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)

#使用普通Lasso回归
from sklearn.linear_model import Lasso
lareg = Lasso()
lareg.fit(x_train, y_train)
lareg.predict(x_test)
print(lareg.score(x_test, y_test))

#使用含有交叉验证优化机制的LassoCV
from sklearn.linear_model import LassoCV
lacvreg = LassoCV()
lacvreg.fit(x_train, y_train)
lacvreg.predict(x_test)
print(lacvreg.score(x_test,y_test))


'''信息准则优化

'''