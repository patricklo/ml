'''模型的训练
1.根据问题特点选择适当的estimator模型：
1.1 classification 分类模型有：SVC,KNN,LR,NaiveBayes...
1.2 regression 回归模型有：Lasso,ElasticNet,SVR
1.3 clustering
聚类模型：KMeans
1.4 dimensionality reduction 降维模型：PCA
'''
'''#######
··1.1分类模型训练＇
######'''
#1.1.1用决策树算法识别手写字体
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
digits=datasets.load_digits()
#plt.figure()
#plt.axis('off')
#plt.imshow(digits.images[0],cmap=plt.cm.gray_r,interpolation='nearest')
#plt.show()
#从样本中选择2/3作为训练集，1/3作为测试集，并打乱数据集。
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(digits.data, digits.target,
test_size=1/3)
##试用决策权分类器
from sklearn.tree import DecisionTreeClassifier
treeclf =DecisionTreeClassifier()
treeclf.fit(x_train, y_train)
#print(treec1f.predict(x_test))
#print(treeclf.predict(x_test)-y_test)##相减后，结果为0的，就是预测成功的。不是0的就预测不成功
##查看模型在测试集的评分
#print(treec1f.score(x test.v test))

#1.1.2 用随机森林算法识别手写字体
###随机森林是一个集成算法，相当于多种决策树进行投票
###可以显著改善单个决策树的过拟合问题
from sklearn.ensemble import RandomForestClassifier
rfclf = RandomForestClassifier()
rfclf.fit(x_train,y_train)
#print(rfclf.predict(x_test)-y_test)##相减后，结果为0的，就是预测成功的。不是0的就预测不成功
#print(rfc1f.score(x_test,y_test))##查看模型在测试集的评分
#1.1.3 按sklearn 地图指引 https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
##选择Linear SVC
from sklearn import svm
svmclf=svm.SVC(kernel='linear')
svmclf.fit(x_train,y_train)
#print(svmclf.predict(x_test)-y_test)##相减后，结果为0的，就是预测成功的。不是0的就预测不成功
#print(svmc1f.score(x_test,y_test))##查看模型在测试集的评分
#########
'''1.2回归模型训练
'''
boston = datasets.load_boston()
#1.2.1 用决策树算法，拟合boston房价的问题
#从样本中选择2/3作为训练集，1/3作为测试集，并打乱数据集。
x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target, test_size=1/3,random_state=0)
#将特征进行MinMax标准化
from sklearn import preprocessing
scaler=preprocessing.MinMaxScaler(feature_range=(0,1))
scaler.fit(x_train)
x_train,x_test=scaler.transform(x_train),scaler.transform(x_test)

from sklearn import tree
treereg=tree.DecisionTreeRegressor()
treereg.fit(x_train,y_train)
#print(treereg.predict(x_test))
#print(np.abs(treereg.predict(x_test)/y_test-1))##相减后，结果为0的，就是预测成功的。不是0的就预测不成功
##查看模型在测试集的评分
#print(treereg.score(x_test,y_test))
#1.2.2 用随机森林回归算法，
from sklearn.ensemble import RandomForestRegressor
rfreg=RandomForestRegressor()
rfreg.fit(x_train,y_train)
#print(np.abs(rfreg-predict(x_test)/y_test-1))##相减后，结果为0的，就是预测成功的。不是0的就预测不成功
##查看模型在测试集的评分
#print(rfreg.score(x_test,y_test))
#1.2.3 按sklearn 地图指引，使用Lasso/ElasticNet回归 或 RidgeRegression
from sklearn.linear_model import LarsCV
lareg=LarsCV()
lareg.fit(x_train,y_train)
#print(lareg-predict(x_test))
#print(lareg-score(x_test,y_test))
###用ElasticNet回归
from sklearn.linear_model import ElasticNetCV
netReg=ElasticNetCV()
netReg.fit(x_train,y_train)
#print(netReg-predict(x_test))
#print(netReg.score(x_test,y_test))
###用RidgeRegression
from sklearn.linear_model import RidgeCV
rgReg=RidgeCV()
rgReg.fit(x_train,y_train)
#print(rgReg-predict(x_test))
#print(rgReg.score(x_test,y_test))
##用svr
from sklearn.svm import SVR
svr=SVR(kernel='linear',C=1000)
svr.fit(x_train,y_train)
#print(svr.predict(x_test))
#print(svr.score(x_test,y_test))
'''###############
1.3 聚类模型训练
###############'''
##Kmeans算法－基本思想
###随机选择K个点作为初始核心，当簇发生变化或小于最大迭代数，将每个点指派到最近的质心，形成k个簇，重新计算每个簇的质心／核心
from sklearn.datasets import make_blobs
center = [[1,1],[-1,-1],[1,-1]] ##指定中心
cluster_std  = 0.2
x,labels=make_blobs(n_samples=200,centers=center,n_features=2,
                    cluster_std=cluster_std,random_state=0)
df=pd.DataFrame(np.c_[x,labels],columns=['feature1','feature2','labels'])
df['labels']=df['labels'].astype('i2')
#df.plot.scatter('feature1','feature2',s=100,
#               c= list(df["labels']),cmap='rainbow',colorbar=False,
#               alpha=0.8,title='dataset by make_blobs')
#plt.show()
from sklearn.cluster import KMeans
clt=KMeans(n_clusters=3)
clt.fit(x)
#print(clt.predict(x))
#print(clt.cluster_centers_)##对比初始的指定中心位置，非常接近
centers = clt.cluster_centers_
#fig·df.plot.scatter('feature1','feature2',s=100,
#               c=list(df['labels']),cmap='rainbow', colorbar=False,
#               alpha=0.8,title='dataset by make_blobs')
#fig.axes.scatter(x=centers[:,0],y=centers[:,1],facecolor='k',s=10)
#plt.show()
'''
1.4 降维模型训练·
·,###############'''
#PCA(principal components analysis)是最常使用的降维算法，其基本思想如下：
##将原先的n个特征用数目更少的m个特征取代，新特征是旧特征的线性组合，这些线性组合最大化样本方案
##从而保留样本尽可能多的信息，并且m个特征互不相关
##用几何观点来看，PCA主成分分析方法可以看成通过正交变换，对坐标系进行旋转和平移，并保留样本点投影坐标方差最大的前几个新的坐
###通过PCA主成分分析，可以帮助去除样本中的噪声信息，便于进一步做回归分析
boston = datasets.load_boston()
x_train,x_test,y_train,y_test=train_test_split(boston.data,boston.target,
                                                test_size=1/3,random_state=0)
##特征极值标准化
scaler=preprocessing.MinMaxScaler(feature_range=(0,1))
scaler.fit(x_train)
x_train,x_test=scaler.transform(x_train),scaler.transform(x_test)
from sklearn.decomposition import PCA
pca=PCA(n_components=6)
pca.fit(x_train)
x_train_pca,x_test_pca = pca.transform(x_train),pca.transform(x_test)
#print(pca.explained variance_)0
#plt.figure()
#plt.plot(pca.explained_variance_,'k', linewidth=2)
#plt.xlabel('components_n', fontsize-16)
#plt.ylabel('explained_var',fontsize=16)
#plt.show()
#对降维后的数据进行回归分析
from sklearn.linear_model import ElasticNetCV
netreg=ElasticNetCV()
netreg.fit(x_train_pca,y_train)
print(netreg.predict(x_test_pca))
print(netreg.score(x_test_pca,y_test))##降维后，分数会比原来有所降低，但大部分特征保留了。









