'''############################'''
'''3 案例：用逻辑回归制作评分卡： 个人消费贷款数据'''

'''开发流程：
一个完整的模型开发，需要有以下流程：
    获取数据 -> 数据清洗，特征工程 -> 模型开发 -> 模型检验与评估 ->模型上线 -> 监测与报告
 
'''

'''###########'''
'''3.1 导库，获取数据'''
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv(r'./rankingcard.csv', index_col=0)
#print(data.shape)
#print(data.info())


'''3.2.1 去除重复值'''
data.drop_duplicates(inplace=True)
data.index = range(data.shape[0])
#print(data.isnull().sum()/data.shape[0]) ##查看缺失值比例

'''3.2.2 填补缺失值'''
data['NumberOfDependents'].fillna(int(data['NumberOfDependents'].mean()), inplace=True)

##填补缺失值 - MonthlyIncome 月收入
## 用随机森林填补缺失值
def fill_missing_rf(x, y, to_fill):
    '''
    使用随机森林填补一个特征的缺失值的函数
    参数:
    X:要填补的特征矩阵
    y:完整的，没有缺失值的标签
    to_fill:字符串，要填补的那一列的名称 '''
    df = x.copy()
    fill = df.loc[:, to_fill]
    df = pd.concat([df.loc[:, df.columns != to_fill], pd.DataFrame(y)], axis=1)
    ytrain = fill[fill.notnull()]
    ytest = fill[fill.isnull()]
    xtrain = df.iloc[ytrain.index, :]
    xtest = df.iloc[ytest.index, :]

    from sklearn.ensemble import RandomForestRegressor as RFR
    rfr = RFR(n_estimators=100)
    rfr = rfr.fit(xtrain, ytrain)
    ypredict = rfr.predict(xtest)
    return ypredict

x = data.iloc[:, 1:]          #除去标签的所有数据
y = data['SeriousDlqin2yrs']  #标签
y_pred = fill_missing_rf(x, y, 'MonthlyIncome')
#确认我们的结果合理之后，我们就可以将数据覆盖了
data.loc[data.loc[:,"MonthlyIncome"].isnull(),"MonthlyIncome"] = y_pred

'''3.2.3 描述性统计处理异常值'''
#描述性统计
print(data.describe([0.01,0.1,0.25,.5,.75,.9,.99]).T)
#异常值也被我们观察到，年龄的最小值居然有0，这不符合银行的业务需求，即便是儿童账户也要至少8岁，我们可以 查看一下年龄为0的人有多少
#(data["age"] == 0).sum() #发现只有一个人年龄为0，可以判断这肯定是录入失误造成的，可以当成是缺失值来处理，直接删除掉这个样本
data = data[data["age"] != 0]
"""
另外，有三个指标看起来很奇怪:
"NumberOfTime30-59DaysPastDueNotWorse"
"NumberOfTime60-89DaysPastDueNotWorse"
"NumberOfTimes90DaysLate"
这三个指标分别是“过去两年内出现35-59天逾期但是没有发展的更坏的次数”，“过去两年内出现60-89天逾期但是没 有发展的更坏的次数”,“过去两年内出现90天逾期的次数”。这三个指标，在99%的分布的时候依然是2，最大值却是 98，看起来非常奇怪。一个人在过去两年内逾期35~59天98次，一年6个60天，两年内逾期98次这是怎么算出来的?
我们可以去咨询业务人员，请教他们这个逾期次数是如何计算的。如果这个指标是正常的，那这些两年内逾期了98次的 客户，应该都是坏客户。在我们无法询问他们情况下，我们查看一下有多少个样本存在这种异常:
"""
data[data.loc[:,"NumberOfTimes90DaysLate"] > 90].count()
#有225个样本存在这样的情况，并且这些样本，我们观察一下，标签并不都是1，他们并不都是坏客户。因此，我们基 本可以判断，这些样本是某种异常，应该把它们删除。
data = data[data.loc[:,"NumberOfTimes90DaysLate"] < 90] #恢复索引
data.index = range(data.shape[0])

'''3.2.4 为什么不统一量纲化，也不标准化数据分布
因为要为业务服务，量纲化/标准化带来的结果是业务人员看不懂结果，无法直接使用
'''
'''3.2.5 样本不均衡问题'''
'''样本是严重不均衡。
虽然大家都在努力防范信用风险，但实际违约的人并不多。
并且，银行并不会真的一棒子打死所有会违约的人，很多人是会还钱的，只是忘记了还款日，很多人是不愿意欠人钱的，但是当时真的很困难，资金周转不过来，所以发生逾期，但一旦他有了钱，他就会把钱换上。
对于银行来说，只要你最后能够把钱还上，我都愿意借钱给你，因为我借给你就有收入(利息)。
所以，对于银行来说，真正想要被判别出来的其实 是”恶意违约“的人，而这部分人数非常非常少，样本就会不均衡。
这一直是银行业建模的一个痛点:我们永远希望捕捉少数类。

之前提到过，逻辑回归中使用最多的是上采样方法来平衡样本。
'''
# 探索标签的分布
x = data.iloc[:, 1:]
y = data.iloc[:, 0]
print(y.value_counts())
n_sample = x.shape[0]
n_1_sample = y.value_counts()[1]
n_0_sample = y.value_counts()[0]
print('样本个数:{}; 1占{:.2%}; 0占 {:.2%}'.format(n_sample, n_1_sample / n_sample, n_0_sample / n_sample))

# imblearn是专门用来处理不平衡数据集的库，在处理样本不均衡问题中性能高过sklearn很多
# #imblearn里面也是一个个的类，也需要进行实例化，fit拟合，和sklearn用法相似
import imblearn
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
x, y = sm.fit_resample(x, y) ##返回已经上采样过的特征矩阵和标签
n_sample_ = x.shape[0]
print(pd.Series(y).value_counts())
n_1_sample = pd.Series(y).value_counts()[1]
n_0_sample = pd.Series(y).value_counts()[0]
##这样我们就实现了样本平衡，样本量也增加了。
print('样本个数:{}; 1占{:.2%}; 0占 {:.2%}'.format(n_sample_, n_1_sample / n_sample_, n_0_sample / n_sample_))


'''3.2.6 分训练集和测试集'''
x = pd.DataFrame(x)
y = pd.DataFrame(y)
X_train, X_vali, Y_train, Y_vali = train_test_split(x,y,test_size=0.3,random_state=420)
model_data = pd.concat([Y_train, X_train], axis=1)
model_data.index = range(model_data.shape[0])
model_data.columns = data.columns
vali_data = pd.concat([Y_vali, X_vali], axis=1)
vali_data.index = range(vali_data.shape[0])
vali_data.columns = data.columns
model_data.to_csv(r'./model_data.csv')  #train model data  (xtrain, ytrain)
vali_data.to_csv(r'./vali_data.csv')   #validate model data after we trained model  (xtest, ytest)


'''3.3 分箱'''