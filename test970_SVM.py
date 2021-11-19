'''sklearn中的支持向量机SVM(下)

1.2 参数C的理解进阶
  有一些数据，可能是线性可分的，但在线性可分状况下训练准确率不能达到100%，即无法让训练误差为0，这样的数据我们称为"存在软间隔的数据"。
   （即可以找出的一条决策边界线，但有可能有几个数据被分错）
    那这时，我们需要让我们决策边界能够忍受一小部分训练误差，我们就不能单纯地寻求最大边际（margin)
  因为对于软间隔地数据来说，边际越大被分错的样本也就会越多，因此我们需要找出一个"最大边际"与"被分错的样本数量"之间的平衡。
  因此，我们引入松弛系数ζ 和松弛系数C作为一个惩罚项，来惩罚我们对最大边际的追求。
'''
'''
时此刻，所有可能影响我们的超平面的样本可能都会被定义为支持向量，所以支持向量就不再是所有压在虚线超
平面上的点，而是所有可能影响我们的超平面的位置的那些混杂在彼此的类别中的点了。观察一下我们对不同数据
集分类时，支持向量都有哪些?软间隔如何影响了超平面和支持向量，就一目了然了。
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
from sklearn.preprocessing import StandardScaler
# n_samples = 100
# datasets = [
#     make_moons(n_samples=n_samples, noise=0.2, random_state=0),
#     make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=1),
#     make_blobs(n_samples=n_samples, centers=2, random_state=5),
#     make_classification(n_samples=n_samples,n_features =2,n_informative=2,n_redundant=0, random_state=5)
#     ]
# Kernel = ["linear"]
# #四个数据集分别是什么样子呢?
# # for X,Y in datasets:
# #     plt.figure(figsize=(5,4))
# #     plt.scatter(X[:,0],X[:,1],c=Y,s=50,cmap="rainbow")
# nrows=len(datasets)
# ncols=len(Kernel) + 1
# fig, axes = plt.subplots(nrows, ncols,figsize=(10,16)) #第一层循环:在不同的数据集中循环
# for ds_cnt, (X,Y) in enumerate(datasets):
#     ax = axes[ds_cnt, 0]
#     if ds_cnt == 0:
#         ax.set_title("Input data")
#     ax.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,edgecolors='k')
#     ax.set_xticks(())
#     ax.set_yticks(())
#     for est_idx, kernel in enumerate(Kernel):
#         ax = axes[ds_cnt, est_idx + 1]
#         clf = SVC(kernel=kernel, gamma=2).fit(X, Y)
#         score = clf.score(X, Y)
#         ax.scatter(X[:, 0], X[:, 1], c=Y
#                    , zorder=10
#                    , cmap=plt.cm.Paired, edgecolors='k')
#         ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
#                    facecolors='none', zorder=10, edgecolors='white')
#         x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
#         y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
#         XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
#         Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
#         ax.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
#         ax.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
#                    levels=[-1, 0, 1])
#         ax.set_xticks(())
#         ax.set_yticks(())
#         if ds_cnt == 0:
#             ax.set_title(kernel)
#         ax.text(0.95, 0.06, ('%.2f' % score).lstrip('0')
#                 , size=15
#                 , bbox=dict(boxstyle='round', alpha=0.8, facecolor='white')  # 为分数添加一个白色的格子作为底色
#                 , transform=ax.transAxes  # 确定文字所对应的坐标轴，就是ax子图的坐标轴本身
#                 , horizontalalignment='right'  # 位于坐标轴的什么方向
#          )
# plt.tight_layout()
# plt.show()


'''1.3 二分类SVC中的样本不均衡问题：重要参数class_weight
样本不均衡指，在一组数据中，标签的一类值天生占有很大比例（如：对潜在犯罪和普通人进行区别，那标签中普通人的比例非常大）
  带来总是：
    1. 首先，分类模型天生会倾向于多数的类，让多数类更容易被判断正确，少数类被牺牲掉。
    2. 其次，模型评估指标会失去意义
   解决办法：
     1.首先要让算法意识到数据的标签是不均衡的，通过施加一些惩罚或者改变样本本身，来让模型向着捕获少数类的方向建模。
     2. 然后，我们要改进我们的模型评估指标，使用更加针对于少数类的指标来优化模型。
在之前的课程中，对于解决办法1，我们有上采样/下采样方法来增加采样的总数，但对于支持向量机来说-》影响速度 、 影响决策边界

因此，对于SVC支持向量机，我们要大力使用调节样本均衡的参数:SVC类中的class_weight和接口fit中可以设定的sample_weight。
     但在SVM中，我们的分类判断是基于决策边界的，而最终决定究竟使用怎样的支持向量和决策边界的参数是参数C，所以所有的样本均衡都是通过参数C来调整的。
     那调整的class_weight / sample_weight 怎么到C呢？原来是通过weight*C 来得到新C
'''
'''class_weight在样本不均衡中的使用'''

#1. 创建样本不均衡的数据集
class_1 = 500 #类别1的数量
class_2 = 50  #类别2的数量
centers = [[0.0, 0.0], [2.0, 2.0]] #设定2个类别的中心
clusters_std = [1.5, 0.5] #设定2个类别的方差，通常来说，样本量比较大的类别会更加松散（std大）
x, y = make_blobs(n_samples=[class_1, class_2],
                  centers=centers,
                  cluster_std=clusters_std,
                  random_state=0, shuffle=False)
#先画图看看数据长什么样
# plt.scatter(x[:,0], x[:,1], c= y, cmap='rainbow', s=10)
# plt.show()
#2. 在数据集上分别建模
#不设定class_weight
clf = SVC(kernel='linear', C=1.0)
clf.fit(x,y)

#设定class_weight
w_clf = SVC(kernel='linear',class_weight={1: 10})
w_clf.fit(x,y)
#准确率accuracy
print(clf.score(x,y)) #不做样本均衡，0.94

print(w_clf.score(x,y)) #做了样本均衡，0.91，下降了。
#样本均衡，的确会导致准确率下降，原因如下：
#做样本均衡前，只有部分的少数类划分正确。
#做了样本均衡，可以把大部分少数类划分正确，但同时也误分类了一部分的多数类，使其错误划分为少数类。
#但，结合我们的现实，判断潜在犯罪者和普通人，做样本均衡，显然是对我们有利的。不放过一个坏人，有可能会误伤好人

#3. 绘制2个模型下数据的决策边界
#首先要有数据分布
# plt.figure(figsize=(6,5))
# plt.scatter(x[:, 0], x[:, 1], c=y, cmap="rainbow",s=10)
# ax = plt.gca() #获取当前的子图，如果不存在，则创建新的子图
#绘制决策边界
### 第一步:要有网格
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# xx = np.linspace(xlim[0], xlim[1], 30)
# yy = np.linspace(ylim[0], ylim[1], 30)
# YY, XX = np.meshgrid(yy, xx)
# xy = np.vstack([XX.ravel(), YY.ravel()]).T
# ### 第二步:找出我们的样本点到决策边界的距离
# Z_clf = clf.decision_function(xy).reshape(XX.shape)
# a = ax.contour(XX, YY, Z_clf, colors='black', levels=[0], alpha=0.5, linestyles=['-'])
# Z_wclf = w_clf.decision_function(xy).reshape(XX.shape)
# b = ax.contour(XX, YY, Z_wclf, colors='red', levels=[0], alpha=0.5, linestyles=['-'])
# ### 第三步:画图例
# plt.legend([a.collections[0], b.collections[0]], ["non weighted", "weighted"],loc="upper right")
# #plt.show()


'''
上面例子我们可以看到，做了样本均衡，反而准确率下降了。
其实我们的目标是希望尽量捕获少数类，那准确率这个单一的评估模型逐渐失效，因此我们需要新的模型评估指标

在现实中，我们往往寻找捕获少数类的能力和将多数类判错后需要付出的成本的平衡。
为了评估这个能力，我们引入新的模型评估指标：混淆矩阵和ROC曲线

2. SVC的模型评估指标 

2.1 混淆矩阵 confusion matrix

在混淆矩阵中，我们将少数类认为是正例（1），多数类认为是负例（0），一般使用{0，1}表示。

               预测值
             1        0
真实值   1    11       10
        0    01       00
11：真实值是1，预测值是1，，
00：真实值是0，预测值也是0
01：真实值是0，预测值是1 
10：真实值是1，预测值是0
全部样本之和： 11 + 10 + 01 + 00



2.1.1 模型整体效果： 准确率
accuracy = (11 + 00) / (11 + 10 + 01 + 00)

2.1.2 捕捉少数类的艺术：精确度，召回率 和F1 score
精确度 precision =  11/ (11+01)  , 又叫查准率，表示所有被我们预测为少数类的样本中，真正少数类的占比。
        精确度是”将多数类判 错后所需付出成本“的衡量。
'''
# 所有判断正确并确实为1的样本 / 所有被判断为1的样本 #对于没有class_weight，没有做样本平衡的灰色决策边界来说:
#print((y[y == clf.predict(x)] == 1).sum() / (clf.predict(x) == 1).sum())
# 对于有class_weight，做了样本平衡的红色决策边界来说:
#print((y[y == w_clf.predict(x)] == 1).sum() / (w_clf.predict(x) == 1).sum())

'''
召回率 Recall, 敏感度(Sensitivity),真正率，查全率，表示所有真实为1的样本中，被我们预测正确的样本所点的比例
   Recall = 11/(11+10)
   
'''
#所有predict为1的点 / 全部为1的点的比例
#对于没有class_weight，没有做样本平衡的灰色决策边界来说:
#print((y[y == clf.predict(x)] == 1).sum()/(y == 1).sum())
#对于有class_weight，做了样本平衡的红色决策边界来说:
#print((y[y == w_clf.predict(x)] == 1).sum()/(y == 1).sum())

'''
1. 如果我们希望不计一切代价，找出少数类(比如找出潜在犯罪者的例子)，那我们就会追求高召回率，
如果我们的目标不是尽量捕获少数类，那我们就不需要在意召回率。
     注意召回率和精确度的分子是相同的(都是11)，只是分母不同。
     而召回率和精确度是此消彼长的，两者之间的平衡代表了捕捉少数类的需求和尽量不要误伤多数类的需求的平衡。
     究竟要偏向于哪一方，取决于我们的业务需求: 究竟是误伤多数类的成本更高，还是无法捕捉少数类的代价更高。

'''


'''2.1.3 判错了多数类的考量： 特民度与假正率
特异度(Specificity)表示所有真实为0的样本中，被正确预测为0的样本所占的比例。在支持向量机中，可以形象地 表示为，决策边界下方的点占所有紫色点的比例。
   Specificity = 00 / (01+00)
   特异度衡量了一个模型将多数类判断正确的能力，
   而1 - specificity就是一个模型将多数类判断错误的能力，这种能力被计算如下，并叫做假正率(False Positive Rate):


2.1.4. sklearn中的混淆矩阵计算函数

sklearn.metrics.confusion_matrix  混淆矩阵
sklearn.metrics.accuracy_score
sklearn.metrics.precision_score
sklearn.metrics.recall_score
sklearn.metrics.precision_recall_curve
sklearn.metrics.f1_score

'''


'''2.2 ROC 曲线以及相关问题
ROC曲线，The Receiver Operating Characteristic Curve, 是用不同阈值下的假正率FPR为横坐标，不同阈值的召回率recall为纵坐标的曲线。
'''
class_1_ = 7
class_2_ = 4
centers_ = [[0.0, 0.0], [1,1]]
clusters_std = [0.5, 1]
X_, y_ = make_blobs(n_samples=[class_1_, class_2_],
                  centers=centers_,
                  cluster_std=clusters_std,
                  random_state=0, shuffle=False)
#plt.scatter(X_[:, 0], X_[:, 1], c=y_, cmap="rainbow",s=30)
#plt.show()
'''2. 建模，调用概率'''
from sklearn.linear_model import LogisticRegression as LogiR
clf_lo = LogiR().fit(X_,y_)
prob = clf_lo.predict_proba(X_)
#将样本和概率放到一个DataFrame中 import pandas as pd
prob = pd.DataFrame(prob)
prob.columns = ["0","1"]
print(prob)
'''3. 使用阈值0.5，大于0.5的样本被预测为1，小于0.5的样本被预测为0'''
#手动调节阈值，来改变我们的模型效果
for i in range(prob.shape[0]):
    if prob.loc[i,"1"] > 0.5:
        prob.loc[i,"pred"] = 1
    else:
        prob.loc[i,"pred"] = 0
prob["y_true"] = y_
prob = prob.sort_values(by="1",ascending=False)
#print(prob)

'''4， 使用混淆矩阵查看结果'''
from sklearn.metrics import confusion_matrix as CM, precision_score as P, recall_score as R
print(CM(prob.loc[:, 'y_true'], prob.loc[:, 'pred'], labels =[1 ,0]))
print(P(prob.loc[:,"y_true"],prob.loc[:, 'pred'],labels=[1,0]))
print(R(prob.loc[:,"y_true"],prob.loc[:,"pred"],labels=[1,0]))

'''5. 假如使用0.4作为阈值'''
for i in range(prob.shape[0]):
    if prob.loc[i,'1'] > 0.4:
        prob.loc[i, 'pred'] = 1
    else:
        prob.loc[i, 'pred'] = 0
print(CM(prob.loc[:, 'y_true'], prob.loc[:, 'pred'], labels =[1 ,0]))
print(P(prob.loc[:,"y_true"],prob.loc[:, 'pred'],labels=[1,0]))
print(R(prob.loc[:,"y_true"],prob.loc[:,"pred"],labels=[1,0]))


'''2.2.4 sklearn中的ROC曲线和AUC面积'''
from sklearn.metrics import roc_curve
class_1 = 500 #类别1有500个样本
class_2 = 50 #类别2只有50个
centers = [[0.0, 0.0], [2.0, 2.0]] #设定两个类别的中心
clusters_std = [1.5, 0.5] #设定两个类别的方差，通常来说，样本量比较大的类别会更加松散
X, y = make_blobs(n_samples=[class_1, class_2],
                  centers=centers,
                  cluster_std=clusters_std,
                  random_state=0, shuffle=False)
clf_proba = SVC(kernel="linear",C=1.0,probability=True).fit(X,y)
FPR, recall, thresholds = roc_curve(y,clf_proba.decision_function(X), pos_label=1)
print(FPR)
print(recall)
print(thresholds)

from sklearn.metrics import roc_auc_score as AUC
area = AUC(y,clf_proba.decision_function(X))
# plt.figure()
# plt.plot(FPR, recall, color='red',
#          label='ROC curve (area = %0.2f)' % area)
# plt.plot([0, 1], [0, 1], color='black', linestyle='--')
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('Recall')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()

maxindex = (recall - FPR).tolist().index(max(recall - FPR))
#最佳阈值就这样选取出来了，由于现在我们是使用decision_function来画ROC曲线，所以我们选择出来的最佳阈值 其实是最佳距离。如果我们使用的是概率，我们选取的最佳阈值就会使一个概率值了。只要我们让这个距离/概率 以上的点，都为正类，让这个距离/概率以下的点都为负类，模型就是最好的:即能够捕捉出少数类，又能够尽量 不误伤多数类，整体的精确性和对少数类的捕捉都得到了保证。
#而从找出的最优阈值点来看，这个点，其实是图像上离左上角最近的点，离中间的虚线最远的点，也是ROC曲线的 转折点。如果没有时间进行计算，或者横坐标比较清晰的时候，我们就可以观察转折点来找到我们的最佳阈值。
#到这里为止，SVC的模型评估指标就介绍完毕了。但是，SVC的样本不均衡问题还可以有很多的探索。
# 另外，我们 还可以使用KS曲线，或者收益曲线(profit chart)来选择我们的阈值，都是和ROC曲线类似的用法。
# 大家若有余力， 可以自己深入研究一下。模型评估指标，还有很多深奥的地方。
thresholds[maxindex] #我们可以在图像上来看看这个点在哪里
#plt.scatter(FPR[maxindex],recall[maxindex],c="black",s=30)
#把上述代码放入这段代码中: plt.figure()
plt.plot(FPR, recall, color='red',
         label='ROC curve (area = %0.2f)' % area)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.scatter(FPR[maxindex],recall[maxindex],c="black",s=30)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('Recall')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()