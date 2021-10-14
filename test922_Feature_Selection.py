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

'''## 3.3 Wrapper 包装法
    包装法也是一个特征选择和算法训练同时进行 的方法，与嵌入法十分相似，它也是依赖于算法自身的选择。比如coef_属性或feature_importances_属性来完成特征的选择。
    不同的是，我们往往使用一个目标函数作为黑盒来帮助我们选取特征，而不是自己输入某个评估指标或统计量的阈值。
    而是专门的数据挖掘算法。
    
    feature_selection.RFE
'''
## 随机森林
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier as RFC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

RFC_ = RFC(n_estimators=10, random_state=0)
selector = RFE(RFC_, n_features_to_select=50, step=50).fit(x,y)
#print(selector.support_.sum())
#print(selector.ranking_)
x_wrapper = selector.transform(x)
#print(cross_val_score(RFC_, x_wrapper, y, cv=5).mean())
score = []
for i in range(1, 751, 50):
    X_wrapper = RFE(RFC_,n_features_to_select=i, step=50).fit_transform(x,y)
    once = cross_val_score(RFC_,X_wrapper,y,cv=5).mean()
    score.append(once)
plt.figure(figsize=[20, 5])
plt.plot(range(1, 751, 50),score)
plt.xticks(range(1, 751, 50))
plt.show()

