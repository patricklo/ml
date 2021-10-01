'''2 数据预处理Preprocessing & Impute
2.1 数据无量纲化 non-dimensionalize 或者dimensionless
   在机器学习算法实践中，我们往往有着将不同规格的数据转换到同一规格，或不同分布的数据转换到特定范围的需求
       如：让数据保持正态分布，让数据保持在0-1之间等需求

    数据的无量纲化可以是线性的，也可以是非线性的。
        线性的无量纲化包括中心化处理（zero-centered / mean-subtraction) 和缩放处理(Scale).
           中心化的本质是让所有记录加/减去一个固定值，即让数据样本平移到某个位置  -> 一般的处理目的是： 将整个曲线平移到某个位置 ， 一般是往Y轴中心位置平移
             缩放的本质是通过除以一个固定值，将数据固定在某个范围之中，取对数也算是一种缩放处理  -> 一般的处理目的是:将曲线压扁(缩小) 或 拉宽(放大)

    2.1.1 preprocessing.MinMaxScaler()
      当数据(x)按照最小值中心化（即所有记录都减去最小值）后，再按极差(极差 = 最大值 - 最小值， 即所有记录都除以极差)缩放，数据移动了最小值个单位（中心化的作用），并且会被收敛到[0,1]之间(缩放后的作用）
      而这个过程，就叫做数据归一化(Normalization, 又称Mix-Max Scaling)。

        ##注意： Normalization是归一化，不是正则化，真正的正则化是regularization， 不是数据预处理的一种手段

      归一化之后的数据服从正太分布：公式如下：
             x_nor = （x - min(x)) / (max(x) - min(x))  （上面是中心化，下面是缩放）


      在sklearn当中，我们使用preprocessing.MinMaxScaler来实现这一功能。
      feature_range:控制我们希望把数据压缩到的范围，默认是[0,1]
      #当数据中的特征数量非常多的时候，fit会报错并表示数据量太大，此时可以用partial_fit作为训练接口
      #scaler.partial_fit(data)
'''
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
data = [[-1,2],[-0.5, 6], [0, 10], [1, 18]]
scaler = MinMaxScaler()
scaler = scaler.fit(data)
result = scaler.transform(data)
#print(result)
#result_ = scaler.fit_transform(data)  ##上面2步，可以精减为1步
#print(scaler.inverse_transform(result)) ##将归一化后的结果逆转

data = [[-1,2],[-0.5, 6], [0, 10], [1, 18]]
scaler = MinMaxScaler(feature_range=[5,10])
result_ = scaler.fit_transform(data)
#print(result_)

##使用numpy 实现归一化
x =np.array( [[-1,2],[-0.5, 6], [0, 10], [1, 18]])
x_nor = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
#print(x_nor)
#逆转归一化
x_returned = x_nor * (x.max(axis=0) - x.min(axis=0)) + x.min(axis=0)
#print(x_returned)


'''2.1.2 preprocessing.StandardScaler标准化
   将数据(x)按均值中心化后，再按标准差进行缩放，数据就会服从为均值为0，方差为1的正太分布
   而这个过程，就叫做数据标准化（Standardization, 或者Z-score normalization): 公式如下：
       x_standard = (x - µ均值）/ ∂(方差）
'''
from sklearn.preprocessing import StandardScaler
data = [[-1,2],[-0.5, 6], [0, 10], [1, 18]]
scaler = StandardScaler()
scaler.fit(data)
print(scaler.mean_)
print(scaler.var_)
x_std = scaler.transform(data)
print(x_std)
print(x_std.mean())
print(x_std.std())

























