from sklearn import datasets
from sklearn import  preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
##%matplotlib inline -> plt.show()

## 1. 数据获取get data

iris = datasets.load_iris();

##df = pd.DataFrame(iris.data, columns=iris.feature_names)
##df['target'] = iris['target']
##df.plot()
##plt.show()

## 2. 数据预处理
## 标准化(0-1之间), 归一化，二值化，非线性转换，数据特编码，处理缺失值
## 线性化 y = a*x +b  ->  数据标准化
scaler = preprocessing.MinMaxScaler(); ##->归一化 将样本特征值线性缩放到0-1之间
scaler.fit(iris.data) ##先fit
data = scaler.transform(iris.data)  ##再transfrom
target = iris.target
##print(data)

##3. 模型的训练
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(data,target,test_size=1/3)

from sklearn import svm
clf = svm.SVC(kernel='linear', C=1, probability=True)
clf.fit(x_train,y_train)  ##用训练集数据喂养模型
clf.predict(x_test) ##对比y_test
##模型概率
prob = clf.predict_proba(x_test)
print(prob)
##模型得分
score = clf.score(x_test, y_test) #对不同类型的模型有不同的评分算法，由score方法内部定义
print(score)

##4.模型的评估
###score 可以粗劣评估模型质量
###还可以用metrics模块 - 针对不同的问题类型提供了各种评估指标并且可以创建用户自定义的评估指标
##可以使用交叉验证方法评估模型的泛化能力
###k折交叉难示意图 （K=10）


##分类模型评分报告
from sklearn.metrics import classification_report
print(classification_report(target,
                            clf.predict(data),
                            target_names=iris.target_names))
##k折交叉验证
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, data, target, cv=5)  ##采用5折交叉验证
print(scores)

##平均得分和95%置信区间
print("Accuracy: %0.2f (+/-%0.2f)"%(scores.mean(),scores.std()*2))


##5.模型的优化
##优化模型的方法包括 网格搜索法，随机搜索法，模型特定交叉验证，信息准则优化

##网格搜索法在指定的超参数空间中对每一种可能的情况进行交叉验证评分并选出最好的超参数组合
from sklearn.model_selection import GridSearchCV
###估计器
svc = svm.SVC()

##超参数空间
##可能最优参数范围
param_grid = [{'C':[0,1,1,10,100,1000], 'kernel':['linear']},
              {'C':[0,1,1,10,100,1000], 'kernel':['rbf'],'gamma':[0.001, 0.01]}]

###打分函数
scoring = 'accuracy'

#指定采样方法
clf = GridSearchCV(svc, param_grid,scoring=scoring, cv=10)
clf.fit(data,target) ##得到的clf是一个优化了的分类器

clf.predict(data)  ##用优化了的分类器进行分类
print(clf.get_params()) #查看全部参数
print(clf.best_params_)  #查看最优参数

print(clf.best_score_)

###6. 模型的保存
###Option1. 用python自带的pickle模块将训练好的模型保存到磁盘 或或保存成字符串
###Option2. 可以使用sklearn的joblib，但只能保存到磁盘而不能保存成为字符串
###Option1
import pickle
s = pickle.dumps(clf) ##保存成string
print(s)
clf_loaded = pickle.loads(s)

##Option2
import joblib
joblib.dump(clf,'filename.pkl')  ###保存模型文件



























