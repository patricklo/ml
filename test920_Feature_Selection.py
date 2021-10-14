'''3. 特征选择 feature_selection
数据预处理后(data preprocessing)，就要开始特征工程了。
包含： 3.1 feature extraction(特征提取)
          从文字、图像、声音等其他非结构化的数据中提取新信息作为特征。
          比如： 从淘宝宝贝的名称中提取出产品类别，颜色，是否为网红产品
      3.2 feature creation（特征创造）
          把现有特征进行组合，或互相计算，得到新的特征，。
          比如：我们有一列特征是速度，一列是距离，那就可以通过让2列相除，得到新的特征：时间
      3.3 feature selection(特征选择）
           从所有的特征中，选择出有意义，对模型有帮助的特征，以避免必须将所有特征都导入模型去训练的情况。

在做特征选择之前，有3件非常重要的事情：跟数据提供者开会！跟数据提供者开会！！跟数据提供者开会！！！
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

'''## 3.1 filter 过滤法'''
### 过滤方法通常用作预处理步骤，特征选择完全独立于任何机器 学习算法之外，它是根据各种统计检验算法的各项指标秋选择特征
###  全部特征 -> 最佳特征子集 -> 算法 -> 模型评估
'''### 3.1.1 方差过滤'''
'''#### 3.1.1.1 Variance Threshold'''
##  这是通过特征本身的方差来筛选特征的类。
### 比如一个特征本身的数据方差很小，就表示样本在这个特征上基本没有差异，可能特征中的大多数值都一样。那这个特征对于样本的区分没有什么作用。
### 所以要先消除方差为0的特征

from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold()
x_var0 = selector.fit_transform(x)
#print(x_var0.shape) ##(42000, 708) 剩下708个特征， 去掉方差为0的特征

## 需要进一步的特征选择，应该选择将方差小于中位数的特征都去除
x_var0 = VarianceThreshold(np.median(x.var().values)).fit_transform(x)
#print(x_var0.shape) ##(42000, 708) 剩下708个特征， 去掉方差小于median的特征

# 当特征是二分类时，特征的取值就是伯努利随机变量，这些变量的方差可以计算为： var(x) = p(1-p), 其中x是特征矩阵，p是二分类特征的一类在这个特征中所占的概率
# 若特征是伯努利随机变量，假设p=0.8 即二分类特征中某种分类点到80%以上的时候删除特征
x_bvar = VarianceThreshold(0.8 * (1-0.8)).fit_transform(x)
#print(x_bvar.shape)

'''#### 3.1.1.3 方差过滤对模型的影响'''
### 用KNN和随机森林作为例子
### KNN 是k近邻算法中的分类算法，其原理非常简单，是利用每个样本到其他样本点的距离来判断每个样本点的相似度，然后对样本进行分类。KNN必须遍历每个特征和每个样本，因而特征越多，KNN的计算也就会越缓慢。
from sklearn.ensemble import RandomForestRegressor as RFC
from sklearn.neighbors import  KNeighborsClassifier as KNN
from sklearn.model_selection import  cross_val_score

x = data.iloc[:,1:]
y = data.iloc[:,0]
x_fsvar = VarianceThreshold(np.median(x.var().values)).fit_transform(x)
print(x_fsvar)
##KNN方差过滤前
#print(cross_val_score(KNN(), x, y, cv=5).mean())
##KNN方差过滤后
#print(cross_val_score(KNN(), x_fsvar, y, cv=5).mean())

##随机森林过滤前
#print(cross_val_score(RFC(n_estimators=10, random_state=0), x, y, cv=5).mean())
##随机森林过滤后
#print(cross_val_score(RFC(n_estimators=10, random_state=0), x_fsvar, y, cv=5).mean())





'''### 3.1.2 相关性过滤：  卡方过滤 / F检验'''
'''#### 3.1.2.1 卡方过滤'''
## 卡方过滤是专门针对离散型标签（即分类问题）的相关性过滤。
## 卡方检验类feature_selection.chi2 计算每个非负特征和标签之间的卡方统计量，并依照卡方统计量由高到低为特征排名。再结合 feature_selection.SelectKBest这个可以输入"评分标准"来选出前K个分数最高的特征的类。
from sklearn.feature_selection import  SelectKBest
from sklearn.feature_selection import chi2
x_fschi = SelectKBest(chi2, k=300).fit_transform(x_fsvar, y)
#print(x_fschi.shape)
#print(cross_val_score(RFC(n_estimators=10, random_state=0), x_fschi, y, cv=5).mean()) ##实际比之前过滤后的准确性是降低了。
## 这说明我们在设定k=300的时候删除了与模型相关且有效的特征，我们的k值设置得太小，要么调整K值，要么放弃相关性过滤。

'''#### 3.1.2.2 选取超参数K，获取最优K值'''
##方法1，传统方法
import matplotlib.pyplot as plt
#score = []
#for i in range(390,200,-10):
#    x_fschi = SelectKBest(chi2, k=i).fit_transform(x_fsvar, y)
#    once = cross_val_score(RFC(n_estimators=10, random_state=0), x_fschi, y , cv=5).mean()
#    score.append(once)
#plt.plot(range(390,200,-10), score)
#plt.show()

## 方法2：看P值选择k，少用
chivalue,pvalues_chi = chi2(x_fsvar, y)
#print(chivalue)
#print(pvalues_chi)
#k取多少？ 我们想要消除所有p值>设定值的特征，比如>0.05 或 0.01
k = chivalue.shape[0] - (pvalues_chi>0.05).sum()  # 特征总数 - 所有p值>0.05的特征数
#print(k)


'''#### 3.1.2.3 F检验'''
''' 又称ANOVA，方差齐性检验，是用来捕捉每个特征与标签之间的线性关系的过滤方法。
它既可以做回归也可以做分类，因此包含feature_selection.f_classif(F检验分类)和
                               feature_selection.f_regression(F检验回归)两个类。
其中F检验分类用于标签是离散型变量的数据，而F检验回归用于标签是连续型变量的数据。

F检验的本质是寻找两组数据之间的线性关系，其原假设是”数据不存在显著的线性关系“。
它返回F值和p值两个统计量。和卡方过滤一样，我们希望选取p值小于0.05或0.01的特征，这些特征与标签时显著线性相关的，
而p值大于 0.05或0.01的特征则被我们认为是和标签没有显著线性关系的特征，应该被删除。
以F检验的分类为例，我们继续 在数字数据集上来进行特征选择:
'''
from sklearn.feature_selection import f_classif
f,pvalues_f = f_classif(x_fsvar,y)
k = f.shape[0] - (pvalues_f > 0.05).sum()
#print(k)


'''#### 3.1.2.4 互信息法'''
'''互信息法是用来捕捉每个特征与标签之间的任意关系(包括线性和非线性关系)的过滤方法。
和F检验相似，它既 可以做回归也可以做分类，并且包含两个类feature_selection.mutual_info_classif(互信息分类)和 
                                                feature_selection.mutual_info_regression(互信息回归)。
这两个类的用法和参数都和F检验一模一样，不过 互信息法比F检验更加强大，F检验只能够找出线性关系，而互信息法可以找出任意关系。
互信息法不返回p值或F值类似的统计量，它返回“每个特征与目标之间的互信息量的估计”，
这个估计量在[0,1]之间 取值，为0则表示两个变量独立，为1则表示两个变量完全相关。

 所有特征的互信息量估计都大于0，因此所有特征都与标签相关。
当然了，无论是F检验还是互信息法，大家也都可以使用学习曲线，只是使用统计量的方法会更加高效。当统计量 判断已经没有特征可以删除时，无论用学习曲线如何跑，删除特征都只会降低模型的表现。当然了，如果数据量太 庞大，模型太复杂，我们还是可以牺牲模型表现来提升模型速度，一切都看大家的具体需求。

'''
from sklearn.feature_selection import mutual_info_classif as MIC
result = MIC(x_fsvar, y)
k = result.shape[0] - sum(result<=0)
print(k)

'''### 3.1.3 过滤法总结
到这里我们学习了常用的基于过滤法的特征选择，
包括方差过滤，基于卡方，F检验和互信息的相关性过滤，
讲解了各个过滤的原理和面临的问题，以及怎样调这些过滤类的超参数。
通常来说，我会建议，先使用方差过滤，
然后使用互信息法来捕捉相关性，不过了解各种各样的过滤方式也是必要的。
'''