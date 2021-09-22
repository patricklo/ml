'''RandomForrest随机森林是一种集成算法

集成学习(ensemble learning)是时下非常流行的机器学习算法，它本身不是一个单纯的机器学习算法，而是通过在数据上构建多个模型，集成所有模型的建模结果。
基本上所有的机器学习领域都可以看到集成学习的身影，在现实中集成学习也有相当大的作用。
它可以用来做市场营销模拟的建模，统计客户来源，保留和流失，
也可以用来预测疾病的风险和病患者的易感性。
在现在的各种算法竞赛中，随机森林、梯度提升树（GBDT), Xgboost等集成算法的身影也随处可见，可见效果很好，应用之广。

集成算法的目标： 集成算法会考虑多个评估器的建模结果，汇总之后得到一个综合的结果，以此获取比单个模型更好的回归或分类表现

多个模型集成成为的模型叫做集成评估器(ensemble estimator)，组成集成评估器的每个模型都叫做基评估器（base estimator)，
通常来说，有三类集成算法：装袋法(Bagging), 提升法（Boosting)和Stacking
    bagging: 核心是构建多个相互独立的评估器，然后对其预测进行平均或多数表决原则来决定集成评估器的结果。代表模型为随机森林
                随机森林是非常具有代表性的Bagging集成算法，它的所有基评估器都是决策树，
                分类树组的森林就叫随机森林分类器RandomForrestClassifier, 回归树所集成的随机森林回归器RandomForrestRegressor

    boosting: 基评估器是相关的，是按顺序一一构建的，其核心是结合弱评估器的力量一次次对难以评估的样本进行预测，从而构成一个强评估器，
               代表模型有：Adaboost和梯度提升树
'''

# RandomForrestClassifier随机森林分类器
## 1.导入所需包
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
## 2。导入数据集
wine = load_wine()
## 3.复习建模流程
xtrain,xtest,ytrain,ytest = train_test_split(wine.data,wine.target, test_size=0.3)
clf = DecisionTreeClassifier(random_state=0)
rfc = RandomForestClassifier(random_state=0, n_estimators=17)
clf = clf.fit(xtrain, ytrain)
rfc = rfc.fit(xtrain, ytrain)
score_c = clf.score(xtest, ytest)
score_r = rfc.score(xtest, ytest)
print('Singel Tree:{}'.format(score_c),
      'Random Tree:{}'.format(score_r))
## 4,画出随机森林和决策树在一组交叉验证下的效果对比
# 交叉验证：将数据集划分为n份，依次取每一份数据集，每n-1份做训练集
#label = 'RandomForrest'
#for model in [RandomForestClassifier(n_estimators=25), DecisionTreeClassifier()]:
#    score = cross_val_score(model, wine.data, wine.target, cv=10)
#    plt.plot(range(1, 11), score, label=label)
#    plt.legend()
#    label = 'DecisionTree'

#plt.show()

## 5. 画出随机森林和决策树在十组交叉验证下的效果对比
#rfc_1 = []
#clf_1 = []
#for i in range(10):
#    rfc = RandomForestClassifier(n_estimators=25)
#    rfc_s = cross_val_score(rfc,wine.data, wine.target,cv=10).mean()
#    rfc_1.append(rfc_s)
#    clf = DecisionTreeClassifier()
#    clf_s = cross_val_score(clf, wine.data, wine.target, cv=10).mean()
#    clf_1.append(clf_s)
#plt.plot(range(1,11), rfc_1, label='random forrest')
#plt.plot(range(1,11), clf_1, label='decision tree')
#plt.legend()
#plt.show()

## 6.n_estimator的学习曲线
### 2mins
#superpa = []
#for i in range(200):
#    rfc = RandomForestClassifier(n_estimators=i+1, n_jobs=-1)
#    rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10).mean()
#    superpa.append(rfc_s)
#print(max(superpa), superpa.index(max(superpa)))
#plt.figure(figsize=[20, 5])
#plt.plot(range(1,201), superpa)
#plt.show()






