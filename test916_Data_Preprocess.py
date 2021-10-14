'''
2.3 处理连续型特征： 二值化与分段
   二值化： 根据阈值将数据二值化是指在处理连续型变量时，将特征值设置为0或1
           比如>阈值，设为1； <阈值，设为0；
           对文本计数数据的常见处理方法
     分段：也称为分箱，将连续型变量划分为分类型变量的类，能够将连续型变量排序后按顺序分箱后编码

   preprocessing.Binarizer： 特征\标签使用，二值化处理器
   preprocessing.LableBinarizer: 标签专用，二值化处理器
   preprocessing.KBinsDiscretizer: 分箱处理器


'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import KBinsDiscretizer
data = pd.read_csv(r'./Narrativedata.csv', index_col=0)
data.dropna(inplace=True, axis=0)

#Binarizer 二值化处理
x = data.iloc[:,0].values.reshape(-1,1) ##升维，处理特征专用， 因此要reshape为2维数组
transformer = Binarizer(threshold=30).fit_transform(x)
#print(transformer)

#KBinsDiscretizer，分箱
#比如将年龄 0-10：编成0， 10-30：编成1，30-60：编成2
#
'''
n_bins:分箱个数，
encode:编码方式，默认为onehot；
strategy: 用来定义箱宽的方式，默认为quantile（uniform, kmeans)
    uniform: 表示等宽分箱，即每个特征的每个箱的最大值之间的差为（特征.max() - 特征.min())/(n_bins)
             比如特征是取值为0-100之间分布的数字，分布可能不均匀。当我们要定义5个分箱时，
             那使用Uniform会将按照0-20， 20-40，40-60，60-80，80-100，分成5个分箱。
             分箱中样本数可能很不均衡
    quantile: 表示等位分箱，即每个特征中的每个箱内的样本数量都相同。
    kmeans: 表示按聚类分箱，每个箱的值到最近的一维K均值聚类的簇心的距离都相同
'''
x = data.iloc[:,0].values.reshape(-1,1)
est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
est.fit_transform(x)
#查看转换后的分箱，变成了一列中的三箱
print(set(est.fit_transform(x).ravel()))##ravel降维
est = KBinsDiscretizer(n_bins=3, encode='onehot', strategy='uniform')
#查看转换后的分箱，变成了哑变量
print(est.fit_transform(x).toarray())










