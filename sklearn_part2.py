###sklearn 数据集
'''
1.自带的小数据集    sklearn.datasets.load_
  鸢尾花数据集 load_iris() 可用于分类和聚类
  乳腺癌数据集 load_breast_cancer() 可用于分类
  手写数字数字集 load_digits() 可用于分类

2.在线下载的数据集  sklearn.datasets.fetch_
3.计算机生成的数据集 sklearn.datasets.make_
  优点：灵活，无穷无尽
    make_blobs 可用于聚类和分类
    make_classification 可用于分类
    make_circles 可用于分类
    make_moons 可用于分类
    make_regression 可用于回归
    ....
4.svmlight/libsvm格式的数据集 sklearn.datasets.load_svmlight_file
5.mldata.org在线下载的数据集 sklearn.datasets.fetch_mldata()
'''

'''1.自带的小数据集'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
digits = datasets.load_digits()
boston = datasets.load_boston()
iris['data']
iris['target']
#iris['feature_name']
#iris['target_name']


'''3.计算机生成的数据集'''
##优点： 灵活  / 无穷无尽

###3.1 make_blobs
from sklearn.datasets._samples_generator import make_blobs
center = [[1,1], [-1, -1], [1, -1]]
cluster_std = 0.3
X,labels = make_blobs (n_samples=200, centers=center, n_features=2,
                      cluster_std=cluster_std, random_state=0)
print('X.shape:',X.shape)
print('labels:',set(labels))

df = pd.DataFrame(np.c_[X, labels], columns = ['feature1', 'feature2', 'labels'])
df['labels'] = df['labels'].astype('i2')
print(df)
#df.plot.scatter('feature1','feature2', s= 100,
#                c = list(df['labels']), cmap='rainbow', colorbar = False,
#                alpha=0.8, title='dataset by make_blobs')
#plt.show()

###3.2 make_classification
from sklearn.datasets._samples_generator import make_classification
#X,labels = make_classification(n_samples=300, n_features=2, n_classes=2,n_redundant=0,
#                               n_informative=2,
#                               random_state=0, n_clusters_per_class=2)
##rng = np.random.RandomState(2)
#X += 2*rng.uniform(size=X.shape)
#df = pd.DataFrame(np.c_[X,labels], columns=['feature1','feature2', 'labels'])
#df['labels'] = df['labels'].astype('i2')

#df.plot.scatter('feature1', 'feature2', s=100,
#                c = list(df['labels']), cmap='rainbow', colorbar = False,
#                alpha=0.8, title = 'dataset by make_classification')
#plt.show()

### 3.3 make_circles
from sklearn.datasets._samples_generator import make_circles
X,labels = make_circles(n_samples=200, noise=0.2, factor=0.2, random_state=1)
df = pd.DataFrame(np.c_[X,labels], columns=['feature1', 'feature2','labels'])
df['labels'] = df['labels'].astype('i2')

#df.plot.scatter('feature1', 'feature2', s=100,
#                c = list(df['labels']), cmap='rainbow', colorbar=False,
#                alpha=0.8, title ='dataset by make_circles')
#plt.show()

###3.4 make_regressions
from sklearn.datasets._samples_generator import make_regression
X,Y,coef = make_regression(n_samples=100, n_features=1, n_informative=1,
                           n_targets=1, bias=5, effective_rank=None,
                           tail_strength=0, noise=10, shuffle=True,
                           coef=True, random_state=None)
#df = pd.DataFrame(np.c_[X,Y], columns=['x','y'])
#df.plot('x', 'y', kind = 'scatter', s=50, c='m',edgecolor='k')
#plt.show()


####3.5导入天池比赛数据
###network not reachable


'''4. 数据预处理'''
###特征标准化  1.函数接口.scale()  2.类接口 StandardScaler
##常见标准化方式：
#StandardScaler: 缩放至0均值，1标准差
#MinMaxScaler: 缩放到[0,1]之间，也可以指定范围feature_range
#MaxAbsScaler: 缩放至[-1,1]之间，无偏移
#RobustScaler 缩放有异常的特征
from sklearn import preprocessing
from sklearn import datasets
boston = datasets.load_boston()
df = pd.DataFrame(data=boston.data,
                  columns=list(boston.feature_names))
#df.plot(legend=False)
#plt.show()
#MinMaxScaler
minmaxscaler = preprocessing.MinMaxScaler(feature_range=(0,1))
minmaxscaler.fit(df.values)
data = minmaxscaler.transform(df.values)
dfscaled = pd.DataFrame(data, columns=list(boston.feature_names))
#dfscaled.plot(legend=False)
#plt.show()

#Robust Scaler
x = np.array([10, 1000, 0,0, -30,0,20,0,10,0,0,-10])
x = x.reshape(-1,1)
robustscaler = preprocessing.RobustScaler(with_centering=True, with_scaling=True,
                                          quantile_range=(25.0, 75.0), copy=True)
print(robustscaler.fit_transform(x))


##y为稀疏列矩阵
y = np.array([0] * 95 + [0.0, 20, 0, 10, -10])
y = y.reshape(-1, 1)
##使用maxabsscaler 推荐
scaler = preprocessing.MaxAbsScaler(copy=True)
z = scaler.fit_transform(y)
print(z)

##使用无均值偏移的standardscaler缩放稀疏矩阵
## (i-mean)/std
scaler = preprocessing.StandardScaler(with_mean=False, with_std=True, copy=True)
z = scaler.fit_transform(y)
print(z)


'''5.数据正则化（normalize)
      也叫正规化，归一化
'''
X = [[1., -1, 2.],
     [2., 0., 0.],
     [0., 1., -1.]]
#option1: 使用normalize函数
X_normalized = preprocessing.normalize(X, norm='l2')
print(X_normalized)
#option2: 使用Normalizer类
normalizer = preprocessing.Normalizer(norm='l1')
normalizer.fit(X)
X_normalized = normalizer.transform(X)
print(X_normalized)


'''
6. 特征的二值化： 将数值特征用阀值过滤得到布尔值的过程 
'''

X = [[1., -1, 2.],
     [2., 0., 0.],
     [0., 1., -1.]]
binarizer = preprocessing.Binarizer().fit(X)
print(binarizer.transform(X))

binarizer = preprocessing.Binarizer(threshold=1.1)
print(binarizer.transform(X))


'''
7. 数据的分类特征编码
  在机器学习中，特征经常不是数值型的，而是分类型。 如：一个人的性别可能是male、femail 2者之一。我们可以用0、1表示。但这样做的话，性别特征变得有序了。
  
  为了解决这一问题，我们可以使用一种叫做"one-of-K"或"one-hot"的编码方式。即2个特征值来进行编码性别[1,0]表示male, [0,1] 表示"female"。
  通常使用"one-hot"方式编码后会增加数据的维度和稀疏性
'''
a = [[0, 0,3],[1,1,0],[0,2,1],[1,0,2]]
print(a)
one_hot = preprocessing.OneHotEncoder()
print(one_hot.fit_transform(a).toarray())  #将稀疏矩阵 转化为 普通矩阵
#如果训练集中有丢失的分类特征值，必须显式地设置 n_values
#encoder = preprocessing.OneHotEncoder(n_values=[2,4,4])
#print(encoder.fit_transform(a).toarray())






