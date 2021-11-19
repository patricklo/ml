''' 线性回归大家族
回归是一种应用广泛的预测建模技术，核心在于预测的结果是连续型变量。（即不适用于分类型数据）

'''

from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_california_housing as fch #加利福尼亚房屋价值数据集
import pandas as pd
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

house_val = fch()
X = pd.DataFrame(house_val.data)
y = house_val.target

X.columns = house_val.feature_names
#print(X.shape)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)
for i in [Xtrain, Xtest]:
    i.index = range(i.shape[0])
#如果希望进行数据标准化，还记得怎么做吗？
#先用训练集训练(fit)标准化的类，然后再用训练好的类分别转化(transform)训练集和测试集
'''#4.建模'''
reg = LR().fit(Xtrain, Ytrain)
yhat = reg.predict(Xtest)
#yhat:
#从yhat的最大、最小值可以大概知道模型的拟合效果
#print(yhat.min())
#print(yhat.max())
'''#5. 探索建好的模型'''
# 可以发现w都非常的人，因此需要考虑是否对数据进行无量纲化或标准化
# a. 可以在参数中打开标准化 normalize=True,但基本没有帮助.
# b. 可以使用专门的标准化函数
print(reg.coef_) # w, 系数的向量
print(reg.intercept_) #W0
print([*zip(Xtrain.columns, reg.coef_)])

#print([*zip(Xtrain.columns,reg.coef_)])\\\



'''3. 回归类的模型的评估指标
2个角度： 是否预测了正确的数值 ， 是否拟合到了足够的信息 
3.1 角度1： 是否预测了正确的数值
    RSS残差平方和，本质是预测值与真实值之间的差异。
    RSS既是我们的损失函数，也是我们回归类模型的模型评估指标之一。
       但RSS是一个无界的和，我们都追求RSS接近于0是最好，但不知道究竟多小才算好
    因此我们在sklearn中使用RSS的变体，MSE均方误差（mean squared error)来衡量 
        MSE是在RSS的基础上除以样本数量，得到了每个样本上的平均误差。 
    (除了MSE,还有MAE可以用，mean absolute error绝对均值误差）
        
'''
from sklearn.metrics import mean_squared_error as MSE
print(MSE(yhat, Ytest)) #0.53, 也就是平均每个样本差异在0.5左右，均值是2，也就是平均每个差异20%，很高了!
import sklearn
sorted(sklearn.metrics.SCORERS.keys())
#neg_mean_squared_error : negative?? 为什么是负mse?
#因为sklearn将mse划分为模型中的一种损失(loss),loss都用负数表示
#因此需要*（-1）得到真正的值
cross_val_score(reg, X, y, cv=10, scoring='neg_mean_squared_error')*(-1)

'''3.2 角度2: 是否拟合了足够的信息
   衡量模型捕捉的信息量： R平方(R2):
   r2越接近1越好
'''
from sklearn.metrics import r2_score
r2_score(yhat, Ytest) #方法1

r2 = reg.score(Xtest, Ytest) #方法2
print(r2)
#方法1和方法2 出来的结果不一样
##线性回归的大坑 二号：相同的评估指标不同的结果
## 应在方法1中指定哪个是预测值，哪个是真实值
print(r2_score(y_true=Ytest, y_pred=yhat))

print(cross_val_score(reg, X, y, cv=10, scoring='r2')) #方法3
print(cross_val_score(reg, X, y, cv=10, scoring='r2').mean())


##R2值在0.60左右， 说明我们大部分数据被拟合得较好，但图像的开头或结尾处的少数数据却有着较大的拟合误差。
## 画图示意：
import matplotlib.pyplot as plt
#sorted(Ytest)
#plt.plot(range(len(Ytest)), sorted(Ytest), c='black', label='Data')
#plt.plot(range(len(yhat)), sorted(yhat), c='red', label='Predict')
#plt.legend()
#plt.show()


##注意！！ 如果R2算出来是负的，如果你检查了所有的代码，也确定了你的预处理没有问题，
# 但你的也还是负的，
# 那这就证明，线性回归模型不适合你的数据，试试看其他的算法吧。
# 如果你试了其它模型，R2还是负的，那说明你的数据并没有任何规律性
# 观察如下随机数据的R2
rng = np.random.RandomState(42)
X = rng.randn(100, 80)
y = rng.randn(100)
print(cross_val_score(LR(), X, y, cv=5, scoring='r2'))








