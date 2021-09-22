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
#plt.plot(range(1,11), test, color='red', label = 'max_depth')
##plt.legend()
#plt.show()



###class_weight & min_weight_fraction_leaf 目标权重参数
# 完成样本标签平衡的参数
# 样本不平衡是指在一组数据中，标签的一类天生占有很大的比例。
#   比如说，在银行要判断'一个办了信用卡的人是否会违约'，就是 '是' vs '否' （1% ： 99%）的比例。
# 这种分类状况下，即便模型什么也不做，全把结果预测成'否'，正确率也有99%。
# 因此我们要使用class_weight参数对样本标签进行一定的均衡，给少量的标签更多的权重，让模型更偏向于少量的类。




'''交叉验证cross_val_score
是用来观察模型的稳定性的一种方法，我们将数据划分为n份，依次使用其中一份作为测试集，其他n-1份作为训练集，多次计算模型的精确性来评估模型的平均准确程度。
训练集和测试集的划分会干扰模型的结果 ，因此用交叉验证n次的结果求出的平均值，是对模型效果的一个更好的度量。
'''
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_boston
boston = load_boston()  ##回归类型的数据，与红酒load_wine分类数据不一样，
##回归模型
regressor = DecisionTreeRegressor(random_state=0)
#交叉验证分数：
### regressor: 可以是分类、回归、支持向量机等
### data:完整数据， 不需要分训练集和测试集
###target:完整标签
###CV： 测试次数
###scoring: 衡量模型的分数算法   (回归regressor，默认返回的是r square分数)
cross_val_score(regressor, boston.data, boston.target, cv=10,scoring='neg_mean_squared_error')




'''一维回归的图像绘制 

接下来我们到二维平面上来观察决策树是怎样拟合一条曲线的，我们用回归树来拟合正弦曲线，
创建一条有噪声的正弦曲线
来观察回归树的表现
'''
#1。
import numpy as np
import matplotlib.pyplot as plt
##2.创建一条有噪声的正弦曲线
##在这一步，我们的基本思路是，先创建一组随机的，分布在0-5上的横坐标的取值（x),然后将这一组值放到sin函数中去生成纵坐标的值(y), 接着再到y上去添加噪声信息。
##全程将使用numpy来为我们生成这个正弦曲线
rng = np.random.RandomState(1) #生成随机数据
x = np.sort(5 * rng.rand(80,1), axis=0)  #随机80行，1列的数据
y = np.sin(x).ravel()  ##ravel降维函数 二维降为一维
y[::5] += 3*(0.5-rng.rand(16))  ##加入噪声信息， 每隔5个取一个数，一共16个，
#np.random.rand(数据结构) 生成 随机 数的函数

#3.实例化 & 训练模型
#用2个模型：因为想知道在不同模型的拟合效果
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(x,y)
regr_2.fit(x,y)

#4.测试集导入模型，预测结果
#np.newaxis 增维的效果(跟reval用法相反），将下面的随机数组（一维,n个数字的数组）,变成二级（ n行 1列， n*1）
#x_test.shape
x_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]  ##生成 从0到5结束的随机数，且有顺序，步长为0.01
y_1 = regr_1.predict(x_test) #predict, 结果是 导入测试集，输出每个样本点的回归/分类的结果， 也就是输入x，得出y
y_2 = regr_2.predict(x_test)

#5.绘图
plt.figure()
plt.scatter(x,y,s=20, edgecolors='black', c='darkorange', label='data')
plt.plot(x_test, y_1, color='cornflowerblue', label='max_dep_2', linewidth=2)
plt.plot(x_test, y_2, color='yellowgreen', label='max_dep_5', linewidth=2)
plt.xlabel('data')
plt.ylabel('target')
plt.title('Decision Tree Regression')
plt.legend()
plt.show()












