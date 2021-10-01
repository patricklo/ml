'''随机森林在乳腺癌数据上的调参
'''

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as np
import numpy as np

data = load_breast_cancer()

##(569,30) 一共569行数据，却有30个特征，特征比非常低，很容易拟合
#print(data.data.shape)
## 结果标签， 0,1 是2分型分类，一共2个类型
#print(data.target.shape)
## 进行一次简单的建模，看看模型本身在数据集上的表现
rfc = RandomForestClassifier(n_estimators=100, random_state=90)
score_pre = cross_val_score(rfc, data.data, data.target, cv=10, scoring='accuracy').mean()
## 从下面分数可以看出随机森林在这个数据上的表现不错，在现实数据集中，基本不可能什么都不用调就能达到95%以上的准确率
#print(score_pre)

# 调参1： 先调n_estimator
''' 其实在这里我们可以直接使用网格搜索
但是作为学习来说，一步步来才能看到调参的效果
'''
scorel = []
#for i in range(0,200,10):
#    rfc = RandomForestClassifier(n_estimators=i+1,
#                                 n_jobs=1,
#                                 random_state=90)
#    score = cross_val_score(rfc,data.data,data.target, cv=10).mean()
#    scorel.append(score)
#print(max(scorel), (scorel.index(max(scorel))*10) +1)
#plt.figure(figsize=(20,5))
#plt.plot(range(1,201,10), scorel)
#plt.show()

#再进一步细化
# for i in range(65, 75):
#     rfc = RandomForestClassifier(n_estimators=i,
#                                  n_jobs=1,
#                                  random_state=90
#                                  )
#     score = cross_val_score(rfc,data.data,data.target, cv=10).mean()
#     scorel.append(score)
# print(max(scorel), (scorel.index(max(scorel))*10) +1)
# plt.figure(figsize=(20,5))
# plt.plot(range(65, 75), scorel)
# plt.show()

# 为网格搜索做准备，书写网络搜索的参数
'''
有一些参数是没有参照的，很难说清一个范围，这种情况下我们使用上面的学习曲线，看趋势从曲线跑出的结果中选取一个更小的区间，再跑曲线
        param_grid = {'n_estimator':np.arange(0, 200, 10)}
        param_grid = {'max_depth': np.arange(1, 20, 1)}
        param_grid = {'max_leaf_nodes': np.arange(25,50,1)}
            对于大型数据集，可以尝试从1000来构建，先输入1000，每100个叶子一个区间，然后再缩小范围

而有一些参数是可以找到一个范围的，
    param_grid = {'criterion':['gini','entropy']}
    param_grid = {'min_sample_split':np.arange(2,2+20, 1)}
    param_grid = {'min_sample_leaf':np.arange(1,1+10,1)}
    param_grid = {'max_features':np.arange(5,30,1)}
'''

#开始按照参数对模型整体准确率的影响程序进行调参，1. max_depth
param_grid = {'max_depth': np.arange(1, 20, 1)}
rfc = RandomForestClassifier(n_estimators=39,
                             random_state=90)
gs = GridSearchCV(rfc,param_grid=param_grid)
gs.fit(data.data, data.target)
print(gs.best_params_)
print(gs.best_score_)
'''
在这里，我们注意到，将max_depth设置为有限之后，模型的准确率下降了。限制max_depth，是让模型变得简 单，把模型向左推，而模型整体的准确率下降了，即整体的泛化误差上升了，这说明模型现在位于图像左边，即泛 化误差最低点的左边(偏差为主导的一边)。通常来说，随机森林应该在泛化误差最低点的右边，树模型应该倾向 于过拟合，而不是拟合不足。这和数据集本身有关，但也有可能是我们调整的n_estimators对于数据集来说太大， 因此将模型拉到泛化误差最低点去了。然而，既然我们追求最低泛化误差，那我们就保留这个n_estimators，除非 有其他的因素，可以帮助我们达到更高的准确率。
当模型位于图像左边时，我们需要的是增加模型复杂度(增加方差，减少偏差)的选项，因此max_depth应该尽量 大，min_samples_leaf和min_samples_split都应该尽量小。这几乎是在说明，除了max_features，我们没有任何 参数可以调整了，因为max_depth，min_samples_leaf和min_samples_split是剪枝参数，是减小复杂度的参数。 在这里，我们可以预言，我们已经非常接近模型的上限，模型很可能没有办法再进步了。
那我们这就来调整一下max_features，看看模型如何变化。
'''
#调整max_features
'''
max_features是唯一一个即能够将模型往左(低方差高偏差)推，也能够将模型往右(高方差低偏差)推的参数。
我 们需要根据调参前，模型所在的位置(在泛化误差最低点的左边还是右边)来决定我们要将max_features往哪边调。 
现在模型位于图像左侧，我们需要的是更高的复杂度，因此我们应该把max_features往更大的方向调整，
可用的特征 越多，模型才会越复杂。max_features的默认最小值是sqrt(n_features)，因此我们使用这个值作为调参范围的 最小值。
'''
param_grid = {'max_features':np.arange(5,30,1)}
rfc = RandomForestClassifier(n_estimators=39
                             ,random_state=90
                            )
GS = GridSearchCV(rfc,param_grid,cv=10)
GS.fit(data.data,data.target)
print(GS.best_params_)
print(GS.best_score_)

# 调整min_samples_leaf
param_grid={'min_samples_leaf':np.arange(1, 1+10, 1)}
# 对于min_samples_split和min_samples_leaf,一般是从他们的最小值开始向上增加10或20 #面对高维度高样本量数据，如果不放心，也可以直接+50，对于大型数据，可能需要200~300的范围 #如果调整的时候发现准确率无论如何都上不来，那可以放心大胆调一个很大的数据，大力限制模型的复杂度
rfc = RandomForestClassifier(n_estimators=39
                             , random_state=90
                             )
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(data.data, data.target)
print(GS.best_params_)
print(GS.best_score_)



