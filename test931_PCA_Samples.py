'''降维算法：PCA
还记得我们上一周在讲特征工程时，使用的手写数字的数据集吗?数据集结构为(42000, 784)，用KNN跑一次半小时，得到准确率在96.6%上下，用随机森林跑一次12秒，准确率在93.8%，虽然KNN效果好，但由于数据量太大，KNN计算太缓慢，所以我们不得不选用随机森林。
最后使用嵌入法SelectFromModel选出了324个特征，将随机森林的效果也调到了96%以上。
但是，因为数据量依然巨大，还是有300多个特征。
現在，我们就来试着用PCA处理一下这个数据。
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import  cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC

data = pd.read_csv(r'./digit recognizor.csv')
x = data.iloc[:,1:]
y = data.iloc[:,0]

##1.首先画出方差贡献率曲线，找到最佳的降维后的维度大小范围，决定n_components
## 看图像可以看出，降维后维度在[100, 200]左右可以获取原有维度90%以上的信息，【1-100]左右就可以获取到80%-90%内的信息
#pca_line = PCA().fit(x)
#plt.figure(figsize=[20,5])
#plt.plot(np.cumsum(pca_line.explained_variance_ratio_))
#plt.xlabel('number of components after dimension reduction')
#plt.ylabel('cumulative explained variance ratio')
#plt.show()

##2. 试试用上面算出的[1,100]的维度，看看用随机森林分类器中做交叉验证，看分类如何
## 可以从画出的图像看到范围在20左右到达顶峰
# score = []
# for i in range(1,101,10):
#     x_dr = PCA(i).fit_transform(x)
#     once = cross_val_score(RFC(n_estimators=10, random_state=0), x_dr, y, cv=5).mean() ## cv=5, 5折交叉验证，返回5个结果，然后求均值
#     score.append(once)
# plt.figure(figsize=[20,5])
# plt.plot(range(1,101,10), score)
# plt.show()

##3. 再次根据上面得出的结果，继续画图
## 可以精确的知道 当维度是23时，效果最佳
# score = []
# for i in range(10,25):
#     x_dr = PCA(i).fit_transform(x)
#     once = cross_val_score(RFC(n_estimators=10, random_state=0), x_dr, y, cv=5).mean() ## cv=5, 5折交叉验证，返回5个结果，然后求均值
#     score.append(once)
# plt.figure(figsize=[20,5])
# plt.plot(range(10, 25), score)
# plt.show()


##4. 导入找出的最佳维度进行降维，查看模型效果
x_dr = PCA(23).fit_transform(x)
#print(cross_val_score(RFC(n_estimators=10, random_state=0), x_dr,y,cv=5).mean()) #0.9175
##5. 0.9175的精确度其实比我们之间用嵌入法特征选择过后的0.96高
'''那有什么办法可以提升精确度？ 1. 调整随机森林参数： n_estimator  2. 换模型(KNN)'''
'''1.调整随机森林参数： 效果还是不如嵌入法'''
#print(cross_val_score(RFC(n_estimators=100, random_state=0), x_dr,y,cv=5).mean()) #0.9464

'''2. 换模型 - KNN
   之前因为计算量太大，我们一直使用速度较快的随机森林， 但KNN的效果比随机森林 更好。
   KNN在未调参的状况下已经达到96%的准确率，而随机森林在未调参前只能达到93%，这是模型本身的限制带来的，这个数据使用KNN效果就是会更好。
   现在我们的特征数量已经降到不足原来的3%，可以使用KNN了吗?
'''
from sklearn.neighbors import KNeighborsClassifier as KNN
#print(cross_val_score(KNN(), x_dr, y, cv=5).mean()) #在未调整KNN参数的前提下，已达到0.9698！

##接下来调整KNN参数 -  KNN调参：k值学习曲线
# 从图像可以得出：在k值=4时，到达最高点
# score = []
# X_dr = PCA(23).fit_transform(x)
# for i in range(10):
#     once = cross_val_score(KNN(i + 1), X_dr, y, cv=5).mean()
#     score.append(once)
# plt.figure(figsize=[20, 5])
# plt.plot(range(10), score)
# plt.show()


print(cross_val_score(KNN(2), x_dr, y, cv=5).mean()) #调整KNN参数，已达到0.9688！

