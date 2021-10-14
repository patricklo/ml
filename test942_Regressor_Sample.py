'''############################'''
'''3 案例：用逻辑回归制作评分卡： 个人消费贷款数据'''

'''开发流程：
一个完整的模型开发，需要有以下流程：
    获取数据 -> 数据清洗，特征工程 -> 模型开发 -> 模型检验与评估 ->模型上线 -> 监测与报告
 
'''


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

'''3.3 分箱'''
'''
分箱的本质：其实就是离散化连续型变量，好让拥有不同属性的人被分成不同的类别（打上不同的分数），其实本质比较类似于聚类。
    （1）首先，要分多少个箱子才合适？ 小于10个为佳
        制作评分卡，最好能在4~5个为最佳。
        为了衡量特征上的信息量以及特征对预测函数的贡献，银行业定义了概念Information value(IV):= ∑(good%-bad%) * WOE
        WOE = ln(good%/bad%)
    （2）其次，分箱要达成什么样的效果？ 同一箱子内的人属性应尽量相似的。
    
'''
'''3.3.1 等频分箱 '''
model_data = pd.read_csv(r'./model_data.csv')
model_data['qcut'], updown = pd.qcut(model_data['age'], retbins=True, q=20)
#print(model_data['qcut'])
#print(updown)
# 统计每个分箱中0和1的数量：
# 这里使用了数据透视表的功能groupby
count_y0 = model_data[model_data['SeriousDlqin2yrs'] == 0].groupby(by='qcut').count()['SeriousDlqin2yrs']
count_y1 = model_data[model_data['SeriousDlqin2yrs'] == 1].groupby(by='qcut').count()['SeriousDlqin2yrs']
num_bins = [*zip(updown,updown[1:], count_y0, count_y1)]
print(num_bins)


'''3.3.2 '''
'''3.3.3 定义WOE和IV函数'''
#计算WOE和BAD RATE
#BAD RATE与bad%不是一个东西
#BAD RATE是一个箱中，坏的样本所占的比例 (bad/total) #而bad%是一个箱中的坏样本占整个特征中的坏样本的比例
def get_woe(num_bins):
    # 通过 num_bins 数据计算 woe
    columns = ["min","max","count_0","count_1"]
    df = pd.DataFrame(num_bins,columns=columns)
    df["total"] = df.count_0 + df.count_1
    df["percentage"] = df.total / df.total.sum()
    df["bad_rate"] = df.count_1 / df.total
    df["good%"] = df.count_0/df.count_0.sum()
    df["bad%"] = df.count_1/df.count_1.sum()
    df["woe"] = np.log(df["good%"] / df["bad%"])
    return df
#计算IV值
def get_iv(df):
    rate = df["good%"] - df["bad%"]
    iv = np.sum(rate * df.woe)
    return iv