'''决策树
有监督学习：必须指定数据标签，或特征
非参数： 不限制参数类型

Decision Tree 是一种非参数的有监督学习方法，能够从一系列有特征和标签的数据中总结出出决策规则。
并用树状图来呈现这些规则，以解决分类和回归问题。决策树算法容易理解，适用各种数据。
在解决各种问题时都有良好表现，尤其是以树模型为核心的各种集成算法，在各个行业和领域都有广泛的应用。

sklearn流程：
1.实例化，建立评估模型对象 ->  2.通过模型接口训练模型 -> 3.通过模型接口提取需要的信息
from sklearn import tree
clf = tree.DecisionTreeClassifier() step:1
clf = clf.fit(x_train, y_train)  step:2
result = clf.score(x_test,y_test) step:3

DecisionTreeClassifier(criterion="gini",  //不纯度的算法选择 基尼系统(gini impurity) 或 信息熵(entropy)
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 class_weight=None,
                 ccp_alpha=0.0)
'''
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
wine = load_wine()
data_df = pd.concat([pd.DataFrame(wine.data), pd.DataFrame(wine.target)], axis=1)
##print(data_df)

xtrain,xtest,ytrain,ytest = train_test_split(wine.data,wine.target, test_size=1/3)
clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0) ##random_state 控制随机性，保证score出来每次都一样
clf = clf.fit(xtrain, ytrain)
score = clf.score(xtest, ytest) #返回预测的准确度accuracy
#print(score)
print([*zip(wine.feature_names, clf.feature_importances_)])

#feature_name = wine.feature_names## ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酠','非黄烷类酚类','花青素','颜色强度','','','']
#print(feature_name)
import graphviz
# dot_data = tree.export_graphviz(clf, feature_names=feature_name,
#                                 class_names=['琴酒','雪梨','贝尔摩德'],
#                                 filled=True,rounded=True)
# graph = graphviz.Source(dot_data)
#graph.render('dtree_render_',view=True)


'''random_state / splitter'''
##random_state用来设置分支中的随机模式的参数，默认None，
###在高维度时随机性会表现更明显（即很多特性个数）
###低维度时，随机性几乎不会显现（如鸢尾花数据，只有4个特征，低维度）
##splitter也是用来控制决策树中的随机选项的，有2种输入值，
###输入'best'， 决策树在分支时虽然随机，但是还是会优先选择更重要特征进行分支（重要性也可以通过属性feature_importances查看）
###输入'random'， 决策树在分支时会更加随机，树会更深，对训练集的拟合性将会降低，这也是防止过拟合的一种方式
###当你预测到你的模型会过拟合 ， 用这2个参数帮助你降低过拟合的可能性。


'''剪枝参数'''
#在不加限制的情况下，一棵决策树会生长到衡量不纯度的指标最优，或者没有更多的特征可用为止。
#这样的决策树往往会过拟合，
'''这就是说，它会在训练集上表现很好，但在测试集上却表现很糟糕'''
#我们收集的样本数据不可能和整体的状况完全一致，因此当一棵决策树对训练数据有了过于优秀的解释性，
#它找出规则必须包含了训练样本中的噪声，并使它对未知数据的拟合程度不足

##查看拟合程度
##训练集是100%，测试集是98% ， 不太能说明 过拟合
###除非训练集是100%，但测试集很低
score_for_train = clf.score(xtrain, ytrain)
print(score_for_train)
score = clf.score(xtest, ytest) #返回预测的准确度accuracy
print(score)

''' 为了让决策树有更好的泛化性，我们要对决策村进行剪枝，正确的剪枝策略是优化决策树算法的核心 '''
##剪枝参数1： max_depth：限制树的最大深度
## min_samples_leaf 5开始 / min_samples_split 限制叶子节点
##min_samples_leaf：限定一个节点在分枝后的每个子节点必须包含至少min_saples_leag个训练样本，否则分枝不会发生
##                 或，分枝会朝着满足每个子节点都包含min_samples_leaf个样本的方向去发生
##一般搭配max_depth使用，在回归树中会有神奇的效果，可以让模型变得 更加平滑。建议从5开始，太小则会过拟合，太大则会限制模型的树生长

##min_samples_split：限定一个节点必须要包含至少min_samples_split个训练样本，这个节点才允许被分枝，否则分枝不会发生
clf =tree.DecisionTreeClassifier(criterion='entropy'
                                 ,random_state=30
                                 ,splitter='random'
                                 #,max_depth=3
                                 #,min_samples_leaf=10
                                 #,min_samples_split=10
                                )
clf.fit(xtrain,ytrain)

score = clf.score(xtest,ytest)
#print('剪枝后的分数：',score) ###0.91  ##没有多大区别，所以说明 剪枝并不影响原有的准确率，但好处是层数少了，节省计算空间，


#dot_data = tree.export_graphviz(clf, feature_names=wine.feature_names,
#                                 class_names=['琴酒','雪梨','贝尔摩德'],
#                                 filled=True,rounded=True)
#graph = graphviz.Source(dot_data)
#graph.render('dtree_render_2',view=True)#brew install graphviz


####剪枝参数2： max_features & min_impurity_decrease
# 一般max_depth的使用，用作树的"精修"
# max_feature限制分枝时考虑的特征个数， 超过限制个数的特征都会被舍弃。和max_depth异曲同工
# max_feature是用来限制高维度数据的过拟合的剪枝参数，强行限制数据的信息量。
# 但其方法比较暴力，是直接限制可以使用的特征数量，而强行使决策树停下的参数。
# 在不知道决策树中的各个特征的重要性的情况下，强行设定这个参数可能会导致模型学习不足。
# 如果希望通过降维的方式防止过拟合，建议使用PCA\ICA或者特征选择模块中的降维算法

# min_impurity_decrease是限制信息增益的大小，信息增益小于设定数据的分枝不会发生。->信息增益是 父子节点之间的信息熵的差
# 这是在sklearn 0.19版本的更新。 在0.19之前是min_impurity_split



##最后，有这么多剪枝参数可以使用，如何来确定最优的剪枝参数值呢？
# 这时间，我们就需要使用超参数的曲线来进行判断了。
# 超参数的学习曲线，是一个以超参数的取值为横坐标，以模型的度量指标为纵坐标的曲线
# 它用来衡量不同超参数取值下模型的表现的线。
# 在我们建好的决策树里，我们的模型度量指标就是score

import matplotlib.pyplot as plt
test =[]
for i in range(10):
    clf = tree.DecisionTreeClassifier(max_depth=i+1
                                      ,criterion='entropy'
                                      ,random_state=0
                                      ,splitter='random'
                                      )
    clf = clf.fit(xtrain,ytrain)
    score = clf.score(xtest,ytest)
    test.append(score)
plt.plot(range(1,11), test, color='red', label = 'max_depth')
plt.legend()
plt.show()

