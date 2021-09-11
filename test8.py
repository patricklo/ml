'''模型特定交叉验证
一些特定的模型，sklearn构建了一些内部含有交叉验证优化机制的估计器estimator
它们主要在linear_model模块中：
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
print(len(y_train),len(x_train))

