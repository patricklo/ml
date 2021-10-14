'''3. 特征选择 feature_selection
'''

'''特征工程第一步： 理解业务'''

import pandas as pd
import numpy as np

##用这个数据可能相对夸张，如果使用支持向量机和神经网络，很可能会直接跑不出来。使用KNN跑一次需要半个小时。
##所以，使用这个数据举例，更能体现出特征工程的重要性。
data = pd.read_csv(r'./digit recognizor.csv')
x = data.iloc[:,1:]
y = data.iloc[:,0]
#print(x.shape) ##(42000, 783)  783个特征

''' 过滤特征##'''

'''## 3.2 Embedded 嵌入法
    嵌入法是一种让算法自己决定使用哪些特征的方法，即特征的选择和算法的训练同时进行
在使用嵌入法时，我们先使用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，
根据权值系数从大到小选择特征。这些权值系数往往代表了特征对于模型的某种贡献或重要性。

缺点： 1. 过滤法中使用的统计量可以使用统计知识和常识来查找范围(如p值应当低于显著性水平0.05)，
         而嵌入法中的权值系数却没有这样的范围，
         ---或者我们可以说，权值系数为0的特征对模型丝毫没有作用，
         但当大量特征都对 模型有贡献且贡献不一时，我们就很难去界定一个有效的临界值。
         这种情况下，模型权值系数就是我们的超参数， 我们或许需要学习曲线，
         或者根据模型本身的某些性质去判断这个超参数的最佳值究竟应该是多少。
         在我们之后的学习当中，每次讲解新的算法，我都会为大家提到这个算法中的特征工程是如何处理，
         包括具体到每个算法的嵌入 法如何使用。
       2. 计算量可能会很大
    
    feature_selection.SelectFromModel
        嵌入法使用的类，SelectFromModel是一个元变换器，可以与任何在拟合后具有coef_, feature_importances_属性或参数中可选惩罚项的评估器一起使用
        （比如随机森林和树模型就具有属性feature_importances_，逻辑回归就带有l1和l2惩罚项，线性支持向量机也支持l2惩罚项）
        重要参数：
            参数名                   说明 
           estimator                使用的模型评估器，只要是带有feature_importances_或者coef_属性，或带有l1和l2惩罚项的模型都可以使用
           threshold                特征重要性的阈值，重要性低于这个阈值的特征都将被删除
           prefit                   默认False，判断是否将实例化后的模型直接传递给构造函数。如果为True，则必须直接 调用fit和transform，不能使用fit_transform，并且SelectFromModel不能与 cross_val_score，GridSearchCV和克隆估计器的类似实用程序一起使用。
           norm_order                k可输入非零整数，正无穷，负无穷，默认值为1 在评估器的coef_属性高于一维的情况下，用于过滤低于阈值的系数的向量的范数的阶 数。
           max_features              在阈值设定下，要选择的最大特征数。要禁用阈值并仅根据max_features选择，请设置 threshold = -np.inf
    接下来，会介绍随机森林和决策树模型的嵌入法。
    
    
    在嵌入法下，我们很容易就能够实现特征选择的目标:减少计算量，提升模型表现。
    因此，比起要思考很多统计量的过滤法来说，嵌入法可能是更有效的一种方法。
    然而，在算法本身很复杂的时候，过滤法的计算远远比嵌入法要快，所以大型数据中，我们还是会优先考虑过滤法。
'''
## 随机森林
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as RFC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
RFC_ = RFC(n_estimators=10, random_state=0)
#threshold取多少为好？用学习曲线 -> 得出
x_embedded = SelectFromModel(RFC_, threshold=0.000564).fit_transform(x,y)
print(x_embedded.shape)  ## 只剩47个特征数
print(cross_val_score(RFC_,x_embedded,y,cv=5).mean())

##threshold学习曲线
#print(RFC_.fit(x,y).feature_importances_)
# threshold = np.linspace(0,(RFC_.fit(x,y).feature_importances_).max(),20) ## linspace 有限个数
# score = []
# for i in threshold:
#    X_embedded = SelectFromModel(RFC_,threshold=i).fit_transform(x,y)
#    once = cross_val_score(RFC_,X_embedded,y,cv=5).mean()
#    score.append(once)
# plt.plot(threshold,score)
# plt.show()