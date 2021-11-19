'''
2.2 非线性SVM与核函数
  引入r，进行升维这个操作非常巧妙，但也有一些现实的问题。首先，我们不知道应该什么样的数据使用什么类型的映射函数，来确保在变换空间中找出线性决策边界。
  为解决这个问题，我们引入了核函数 kernal (SVC(kernal='linear'))
    核函数能够帮助我们解决三个问题:
        第一，有了核函数之后，我们无需去担心 究竟应该是什么样，因为非线性SVM中的核函数都是正定核函数 (positive definite kernel functions)，他们都满足美世定律(Mercer's theorem)，
        确保了高维空间中任意两个向量 的点积一定可以被低维空间中的这两个向量的某种计算来表示(多数时候是点积的某种变换)。

        第二，使用核函数计算低维度中的向量关系比计算原本的 要简单太多了。

        第三，因为计算是在原始空间中进行，所以避免了维度诅咒的问题。

   输入     含义        解决问题   核函数的表达式              参数(gamma) 参数(degree) 参数(coef0)
   linear   线性核       线性     K(x,y) = x ** T * y       No          No          No
   poly     多项式核     偏线性    K(x,y) = x ** T * y          YEs          Yes          Yes
   sigmoid  双曲正切核   非线性    K(x,y) = tanh(x ** T * y          Yes          No          Yes
   rbf      高斯径向基    偏非线性  K(x,y) = x ** T * y          Yes          No          No

'''

'''
2.2.4 探索核函数的优势和缺陷

看起来，除了Sigmoid核函数，其他核函数效果都还不错。但其实rbf和poly都有自己的弊端，
我们使用乳腺癌数据集作为例子来展示一下:
'''

from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn.datasets import make_circles,make_blobs,make_classification,make_moons,load_breast_cancer
from sklearn.model_selection import train_test_split
from time import time
import datetime

data = load_breast_cancer()
x = data.data
y = data.target
np.unique(y) #？？
xtrain,xtest,ytrain,ytest = train_test_split(x,y, test_size=0.3, random_state=420)

##其实下面代码poly多项式核函数跑不出结果来（原因是degree=3，维度太高）
#这证明，多项式核函数此时此刻要消耗大量的时间，运算非常的缓慢。
#kernels = ['linear', 'poly', 'rbf', 'sigmoid']
# kernels = ['linear', 'rbf', 'sigmoid']
# for kernel in kernels:
#     time0 = time()
#     clf = SVC(kernel=kernel
#               ,gamma='auto'
#               ##, degree =1  ##因为默认是3，维度太高，poly核函数基本跑不出来
#               ,cache_size=6000 #MB内存大小，进行计算
#               ).fit(xtrain, ytrain)
#     print('the accuracy under kernerl %s is %f' % (kernel, clf.score(xtest, ytest)))
#     print(datetime.datetime.fromtimestamp(time() - time0).strftime('%M:%S:%f'))
'''上面说到的真正问题其实是数据的量纲问题。
## 为什么之前rbf表现很好， 为什么现在这么差（0.59）， 其实就是数据的量纲问题
##查看数据的量纲（单位？）'''
#data = pd.DataFrame(x)

'''##可以看到
## 行：30个特征
## 列：count/mean/std/90%...
问题：
    量纲：从mean/std中可以看到，不同特征的值差距很大，严重量纲不统一
    分布：同一特征的值分布还算正常，个别特征（特别是值比较大的特征）有一点偏态

因此，要对数据做标准化
'''
#print(data.describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).T)
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)

'''再跑结果，rbf达到0.97
量纲统一之后，可以观察到，所有核函数的运算时间都大大地减少了，尤其是对于线性核来说，而多项式核函数（degree=1,线性条件下)居然变成了计算最快的。
其次，rbf表现出了非常优秀的结果。

经过我们的探索，我们可以得到的结论是:
    1. 线性核，尤其是多项式核函数在高次项时计算时非常缓慢
    2. rbf和多项式（poly)函数都不擅长处理量纲不统一的数据集
'''
xtrain,xtest,ytrain,ytest = train_test_split(x,y, test_size=0.3, random_state=420)
kernels = ['linear', 'rbf', 'sigmoid']
for kernel in kernels:
    time0 = time()
    clf = SVC(kernel=kernel
              ,gamma='auto'
              ##, degree =1  ##因为默认是3，维度太高，poly核函数基本跑不出来
              ,cache_size=6000 #MB内存大小，进行计算
              ).fit(xtrain, ytrain)
    print('the accuracy under kernerl %s is %f' % (kernel, clf.score(xtest, ytest)))
    print(datetime.datetime.fromtimestamp(time() - time0).strftime('%M:%S:%f'))

'''
2.2.5 选取与核函数相关的参数 degree & gamma & coef0
    gamma就是表达式中的 γ，
    degree就是多项式核函数的次数d，
    coef0就是常数项r。
    其中，高斯径向基核 函数受到gamma的影响，而多项式核函数受到全部三个参数的影响。
        
   输入     含义        解决问题   核函数的表达式                  参数(gamma) 参数(degree) 参数(coef0)
   linear   线性核       线性     K(x,y) = x ** T * y             No          No          No
   poly     多项式核     偏线性    K(x,y) = （γ（x * ) + r) ** d    YEs          Yes          Yes
   sigmoid  双曲正切核   非线性    K(x,y) = tanh(γ(x*y)+r)          Yes          No          Yes
   rbf      高斯径向基    偏非线性  K(x,y) = e ** -(γ∥x-y∥**2), γ>0  Yes          No          No

参数     含义
degree   整数，可不填，默认3 多项式核函数的次数('poly')，如果核函数没有选择"poly"，这个参数会被忽略
gamma    浮点数，可不填，默认=0.0 核函数中的常数项，它只在参数kernel为'poly'和'sigmoid'的时候有效。
         核函数的系数，仅在参数Kernel的选项为”rbf","poly"和"sigmoid”的时候有效 
         输入“auto"，自动使用1/(n_features)作为gamma的取值 
         输入"scale"，则使用1/(n_features * X.std())作为gamma的取值 
         输入"auto_deprecated"，则表示没有传递明确的gamma
coef0    浮点数，可不填，默认=0.0 核函数中的常数项，
         它只在参数kernel为'poly' / 'sigmoid'有效
'''
