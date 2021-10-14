'''2 数据预处理Preprocessing & Impute
  2.2 缺失值处理
      impute.SimpleImputer
        strategy : mean/ median/ most_frequent/ constant
'''

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
data = pd.read_csv(r'./Narrativedata.csv', index_col=0)

age = data.loc[:,'Age'].values.reshape(-1,1) ##sklearn当中的矩阵必须是2维的,(891,1)
#print(age[:10])
imp_mean = SimpleImputer()
im_median = SimpleImputer(strategy='median')
im_0 = SimpleImputer(strategy='constant', fill_value=0)
im_mode = SimpleImputer(strategy='most_frequent')
imp_mean = imp_mean.fit_transform(age)
im_median = im_median.fit_transform(age)
im_mode = im_mode.fit_transform(age)
im_0 = im_0.fit_transform(age)
#print(imp_mean[:20])
#print(im_median[:20])
#print(im_0[:20])
#print(im_mode[:10])
























