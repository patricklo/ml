'''4.实例 泰坦尼号幸存者的预测
Titanic的沉没是世界上最严重的海难之一，今天 我们通过分类树模型来预测一下生存比率
数据集：www.kaggle.com/c/titanic

train: 已经带有标签的数据集，已经是成型数据。
test: 未带标签的训练，需要在接下来中做操作。
'''
#1. 导入所需要的库
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

#2. 导入数据集，探索数据
data = pd.read_csv(r'./titanic_train.csv')

#3.数据预处理
data.drop(['Cabin','Name','Ticket'], inplace=True, axis=1)
data['Age'] =  data['Age'].fillna(data['Age'].mean())
data.dropna(inplace=True)

##将embard转成数字，利用index设置数据0，1，2，然后assign back to embarked
labels = data['Embarked'].unique().tolist()
data['Embarked'] = data['Embarked'].apply(lambda x: labels.index(x))
##同理 ，将性别转成数字
data['Sex'] = (data['Sex']=='male').astype('int')
#print(data.head())

#4.提取标签和特征矩阵，分测试集和训练集
x = data.iloc[:, data.columns !='Survived']
#标签列
y = data.iloc[:, data.columns =='Survived']
xtrain,xtest,ytrain,ytest = train_test_split(x, y, test_size=0.3)
#重置测试集和训练集的索引，可以不调整
for i in [xtrain, xtest, ytrain, ytest]:
    i.index = range(i.shape[0])
#查看分类好的
print(xtrain.head())

#5,导入模型，粗略跑一下查看结果
clf = DecisionTreeClassifier(random_state=25
                             ,max_depth=3
                             #,criterion='entropy'
                            ,criterion='gini'
                             )
clf = clf.fit(xtrain, ytrain)
score_ = clf.score(xtest, ytest)
#print(score_) ##0.76，默认参数下
score = cross_val_score(clf, x, y, cv=10).mean()
#print(score)  ##0.74，分数较低，说明参数不是最优，可以通过第6步来得出较优的参数(max_depth,criterion等）

#6。在不同的二叉树max_depth下观察模型的拟合情况，可以得到最优的参数(max_depth,criterion等）
tr = []
te = []
for i in range(10):
    ## 为什么用entropy呢
    ## 因为我们观察到，在最大深度=3时，模型拟合不足，
    ## 在训练集和测试集上的表现接近，但却都不那么理想，只能够达到83%左右 ，所以用entropy
    clf = DecisionTreeClassifier(random_state=25
                                 ,max_depth=i+1
                                 #,criterion='entropy'
                                 , criterion='gini'
                                 )
    clf = clf.fit(xtrain,ytrain)
    score_tr = clf.score(xtrain, ytrain)
    score_te = cross_val_score(clf,x, y,cv=10).mean()
    tr.append(score_tr)
    te.append(score_te)
print(max(te))
#plt.plot(range(1,11),tr,color='red',label='train')
#plt.plot(range(1,11),te,color='blue',label='test')
#plt.xticks(range(1,11))
#plt.legend()
#plt.show()


# 7. 用网格搜索 也可以调整参数
#缺点：慢，一个个枚举
gini_thresholds = np.linspace(0,0.5,20)
parameters = {'splitter' :('best', 'random')
              ,'criterion':('gini','entropy')
              ,'max_depth':[*range(1,10)]
              ,'min_samples_leaf':[*range(1,10)]
              ,'min_impurity_decrease':[*np.linspace(0,0.5,20)]
              }
clf = DecisionTreeClassifier(random_state=25)
gs = GridSearchCV(clf,parameters, cv=10)
gs.fit(xtrain, ytrain)
print(gs.best_params_)
print(gs.best_score_)






