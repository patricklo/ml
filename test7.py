'''模型的优化
1.提升评分结果 方向：特征工程，模型选择，模型参数优化
超参数是不直接在估计器内学习的参数，在scikit=learn包中，它们作为估计器类中构造函数的参数进行传递。
典型的例子有：用于支持向量分类器的C,Kerne1和gamma,用于1asso的alpha等
搜索超参数空间以便获得最好 交叉验证分数 的方法是可能的 而且是值得提倡的
搜索超参数空间以优化超参数需明确以下方面：
a.估计器
b.超参数空间
c.交叉验证方案
d.打分函数
e.搜寻或采样方法（网格搜索法 或 随机搜索法）
优化模型的方法包括：网格搜索法，随机搜索法，模型特定交叉验证，信息准则优化
'''
'''一、网格搜索法 GridSearchCV'''
#网格搜索法在指定的超参数空间中对每一种可能的情况进行交叉验证评分并选出最好的超参数组合
#使用网格搜索法或随机搜索法 可以对pipeline进行参数优化，也可以指定多个评估指标
from sklearn import svm,datasets
from sklearn.model_selection import GridSearchCV,ShuffleSplit
iris =datasets.load_iris()
#估计器
svc=svm.SVC()
#超参数空间
param_grid=[{'C':[1,10,100,1000], 'kernel':['linear']},
            {'C':[1,10,100,1000],'gamma':[0.001,0.0001],'kernel':['rbf']}
            ]
#交叉验证方案
cv = ShuffleSplit(n_splits=10,test_size=0.3,random_state=0)
#打分函数
scoring = 'accuracy'
#指定搜索或采样方法
clf=GridSearchCV(svc,param_grid,scoring=scoring,cv= cv)
clf.fit(iris.data,iris.target) #得到的clf是一个优化了的分类器
#print(clf.predict(iris.data))##用优化的c1f对数据进行分类
#print(clf.get_params())
#print(clf.best_params_)
'''二、随机搜索法 RandomizedSearchCV'''
#网格搜索法只能在有限的超参数空间进行暴力搜索
#但随机搜索法可以在无限的超参数空间进行随机搜索
from sklearn.model_selection import ParameterSampler
from scipy.stats.distributions import expon
import numpy as np
np.random.seed(0)
param_grid ={'a':[1,2], 'b':expon()} #b按指数分布
param_list=list(ParameterSampler(param_grid,n_iter=10)) #n_iter 参数列表个数
print(param_list)
from sklearn.model_selection import RandomizedSearchCV
iris=datasets.load_iris()
#估计器
svc=svm.SVC()
#超参数采样空间
param_dist={'C':[1,10,100,100],
    'gamma':[0.001,0.0001,0.00001,0.000001],
    'kernel':['rbf','linear']}
#交叉验证方案
cv=ShuffleSplit(n_splits=3,test_size=0.3,random_state=0)
#打分函数
scoring='accuracy'
#指定搜索或采样方法
clf=RandomizedSearchCV(svc,param_dist,
            scoring=scoring,cv=cv,n_iter=20)
clf.fit(iris.data,iris.target)#得到的c1f是一个优化了的分类器
print(clf.predict(iris.data))
clf.get_params()
print(clf.best_params_)
print(clf.best_score_)



