import pandas as pd
import numpy as np
'''8.特征的选择
特征工程：
'''

'''8.1过滤法 filter'''
#8.1.1 方差选择法
from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold
iris=datasets.load_iris()
##方差选择法，返回值 为特征选择后的数据
##参数threshold为方差阈值
vardata=VarianceThreshold(threshold=3).fit_transform(iris.data)
#print(vardata.shape)
#8.1.2 相关系数法
##使用相关系数法，先要计算各个特征对目标值的相关系数。
##用SelectBEST类结合相关系数来选择特征的代码如下：
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
##选择K个最好的特征，返回选择特征后的数据
##第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量
##输入二元组（评分，P值）的数组，数组第1个特征的评分和p值。在此定义为计算相关系数
##参数K为选择的特征个数
f= lambda X,Y:np.array(list(map(lambda x:pearsonr(x,Y)[0],X.T))).T
print(SelectKBest(f,k=2).fit_transform(iris.data,iris.target))
'''8.2 embedded 嵌入法'''

