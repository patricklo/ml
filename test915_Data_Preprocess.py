'''
2.3 处理分类型特征： 编码与哑变量
   编码： 指的是将文字型的数据转换为数字
         在sklearn当中，除了专用来处理文字的算法，其他算法只能接收数字型数据
         比如：学历的描述通常为['小学','初中']等，在放入sklearn进行计算时，必须转换成对应的数字型数据
 哑变量：

   preprocessing.LabelEncoder 标签专用，能够将分类转换成分类数字值数据
   preprocessing.OrdinalEncoder 特征专用，能够将分类特征转换为分类的数字值数据
   preprocessing.OneHotEncoder 特征专用，独热编码，创建哑变量
   preprocessing.LableBinarizer: 标签专用，创建哑变量，跟OneHotEncoder一样作用，只不过用于标签

'''
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

data = pd.read_csv(r'./Narrativedata.csv', index_col=0)
# y = data.iloc[:,-1]  ##原始的数据，因为这里的标签数据，所以不要求是2维数据；如果是特征矩阵，则会需要要求2维数组
# le = LabelEncoder()
# le = le.fit(y)
# label = le.transform(y)
# print(le.classes_)
# print(label)
# data.iloc[:, -1] = LabelEncoder().fit_transform(data.iloc[:,-1])  #上面可换成一行

## 特征转换为数字值的数据
data_ = data.copy()
data_.dropna(inplace=True, axis=0)
# print(OrdinalEncoder().fit(data_.iloc[:,1:-1]).categories_)
data_.iloc[:, 1:-1] = OrdinalEncoder().fit_transform(data_.iloc[:, 1:-1])

## OneHotEncoder 创建哑变量
''' preprocessing.OneHotEncoder
    上面的代码已经用OrdinalEncoder将2列Sex/Embarked都转换成数字值数据.
    在舱门转换中，我们将[S,C,Q]转换为[0,1,2]代表3个不同的舱门，然而这种转换是正确的吗？

    我们考虑以下3种不同性质的分类数据：
    1. 舱门(S,C,Q)  :名义变量，即这些取值相互间都没有关系
    2. 学历(小学，初中，高中）： 有序变量，即取值并不完全独立，是顺序的，而且取值之间不可以计算的，我们不能说 小学 + 某个取值 = 初中
    3. 体重(>45KG, >90KG, >135kg)：有距变量，各个取值之间有联系，且是可以互相计算的，比如 120 - 40 = 90KG,分类之间可以通过数学计算互相转换

    常用数据分类
    数据类型     数据名称 数学含义 描述                                                                                举例
    离散，定性   名义     =，!=   名义变量，用不同的名字，表示不同含义                                                    邮编，性别，眼睛的颜色，职工号
    离散，定性   有序     <, >    有序变量，为数据的相对大小提供信息，但数据之间大小的间隔不具有固定含义，因此不能加减。         材料的硬度，学历
    连续，定量   有距     +,-     有距变量，数据之间的间隔是有固定意义的，可以加减                                          日期，温度
    连续，定量   比率     *,/     比变量，其之间的间隔和比例都是有意义的，可以加减可乘除                                     以开尔文为量纲的温度，货币数量，计数，年龄，质量，长度，电流


    所以我们在对特征进行编码的时候，简单地将数据转换为[0,1,2]，并不精确。因为在算法看来， 这3个值是连续且可以相互计算，并且是有距变量。
    因此我们可以使用OneHotEncoder来处理这种类型的转换：
      如：'S' 0,  -> [1, 0, 0]
         'Q' 1,  -> [0, 1, 0]
         'C' 2   -> [0, 0, 1]
    这样算法就能够原来这3个取值是没有可计算性质的 ,是'有你就没有我'的不等概念。
    在我们的数据中，性别和舱门，都是这样的名义变量。因此我们要使用OneHotEncoder，将2个特征都转换为哑变量

    sparse matrix稀疏矩阵： 由0，1组成的矩阵
'''
from sklearn.preprocessing import OneHotEncoder

data.dropna(inplace=True, axis=0)
x = data.iloc[:, 1:-1]
enc = OneHotEncoder(categories='auto').fit(x)
result = enc.transform(x).toarray()
# print(result)
##还原
enc.inverse_transform(result)
# 列出result中，哪一列的值对应哪个原始特征的值
# print(enc.get_feature_names())

##下一步，将result放回原数据中
new_data = pd.concat([data, pd.DataFrame(result)], axis=1)
new_data.drop(['Sex', 'Embarked'], axis=1, inplace=True)
new_data.columns = ['Age', 'Survived', 'Female', 'Male', 'Embarded_C', 'Embarded_Q', 'Embarded_S']
# print(new_data.head())

'''LabelBinarizer可以对标签创建哑变量，许多算法都可以处理多标签问题（比如说决策树），但是这样的做法在现实中并不常见，因此不做详细介绍
'''