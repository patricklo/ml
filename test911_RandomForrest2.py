'''RandomForrestRegressor随机森林回归

参数与RandomForrestClassifier相同
  criterion:
      'mse': 圴方误差(mean squared error)，父节点和叶子节点之间的均方误差
      'friedman_mse': 费尔德曼均方误差(
      'mae': 绝对平均误差（mean absolute error)
'''

## 1.导入所需包
import numpy as np
import pandas as pd
from scipy.special import comb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import metrics

#boston = load_boston()
#regressor = RandomForestRegressor(n_estimators=100, random_state=0)
#cross_val_score(regressor, boston.data, boston.target, cv=10,
#                scoring='neg_mean_squared_error')
#print(sorted(metrics.SCORERS.keys())) ##所有模型评估的指标， scoring=''

##实例1：用随机森林回归填补缺失值
# SimpleImputer 专门填补缺失数据
#导入数据
dataset = load_boston()
x_full, y_full = dataset.data, dataset.target
n_samples = x_full.shape[0]
n_features = y_full.shape[0]

# 为完整数据集放入缺失值
## 首先确定我们希望放入的缺失数据的比例，在这里我们假设是50%，那就总共要放入3289个缺失的数据点
rng = np.random.RandomState(0)
missing_rate = 0.5
# np.floor 向下取整
n_missing_samples = int(np.floor(n_samples * n_features * missing_rate))

## 所有数据要随机遍布在数据集的各行各列当中， 而一个缺失的数据会需要一个行和列的索引
## 如果能够创造一个数组，包含3289个分布在0~506行中间的行索引，和3289个分布在0~13之间的列索引
## 那我们就可以利用索引来为数据中的任意3289个位置赋空值
## 然后我们利用0， 均值 和随机森林3种方法来填充这些缺失值，然后查看回归的结果如何
missing_features = rng.randint(0, n_features, n_missing_samples)
missing_samples = rng.randint(0, n_samples, n_missing_samples)

## 现在我们采样了3289个数据， 远远超过我们的样本量506个，所以我们使用随机抽取函数randint，
## 但如果我们需要的数据量小于我们的样本量506，那我们可以采用np.random.choice抽样，choice会随机抽取不重复的随机数。
## 因此可以帮助我们让数据更加分散，确保数据不会集中在一些行中
x_missing = x_full.copy()
y_missing = y_full.copy()
x_missing[missing_samples, missing_features] = np.nan
# 转换为DF，是为了后续方便各种操作。
x_missing = pd.DataFrame(x_missing)

# 使用0和均值填补缺失值
## 使用均值进行填补
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
x_missing_mean = imp_mean.fit_transform(x_missing)

## 使用0进行填补
imp_0 = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value=0)
x_missing_0 = imp_0.fit_transform(x_missing)

## 使用随机森林进行填补
''' 任何回归都是从特征矩阵中学习，然后求解连续型标签y的过程，之所以能够实现这个过程，是因为回归算法认为，特征矩阵和标签之间存在着某种联系。
    实际上，标签和特征是可以相互转换的，比如说，在一个"用地区，环境，附近学校数量" 来预测 "房价" 的问题中，
    我们既可以用"地区，环境，附近学校"的数据来预测房价，也可以用"环境，附近学校，房价"来预测"地区"。而回归填补缺失值，正是利用这种原理。
    
    对于一个有n个特征的数据来说，其中特征T有缺失值，我们就把特征T当作标签，其他n-1个特征和原本的标签组成新的特征矩阵。
    那对于T来说，它没有缺失的部分，就是我们的y_test，这部分数据既有标签也有特征，而它缺失的部分只有特征没有标签，就是我们需要预测的部分。
    
    特征T不缺失的值  对应 其它n-1特征 + 本来的标签 ：x_train
    特征T不缺失的值： y_train
    
    特征T缺失的值 对应 其它n-1特征 + 本来的标签：x_test
    特征T缺失的值：未知，我们需要预测的y_test
    
    这种做法，对于某一个特征大量缺失，其他特征却很完整的情况，非常适用。
    
    那如果数据中除了特征T，其他特征也有缺失值，怎么办？
    答案 就是遍历所的的特征，从缺失最少的开始填补。
'''

































