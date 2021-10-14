'''sklearn中的聚类算法K-Means'''

'''1 概述'''
'''1.1 无监督学习与聚类算法
    有监督学习算法： 即模型在训练的时候，既需要特征矩阵X，也需要真实标签y （决策树，随机森林，逻辑回归等）
    无监督学习算法：只需要特征矩阵X，不需要标签y (PCA降维算法，聚类算法）
    
聚类算法：
聚类算法又叫做“无监督分类”，其目的是将数据划分成有意义或有用的组(或簇)。这种划分可以基于我们的业务需求或建模需求来完成，也可以单纯地帮助我们探索数据的自然结构和分布。
比如在商业中，如果我们手头有大量的当前和潜在客户的信息，我们可以使用聚类将客户划分为若干组，以便进一步分析和开展营销活动，
最有名的客户价值判断模型RFM，就常常和聚类分析共同使用。
再比如，聚类可以用于降维和矢量量化(vector quantization)，可以将高维特征压缩到一列当中，
常常用于图像，声音，视频等非结构化数据，可以大幅度压缩数据量。

聚类：在不知道类别的情况下，将所有数据分成不同类别/簇
分类：在已知类别中，将一个新的未知类别，归为其中一种

         聚类                                分类
核心     将数据分成多个组                      从已经分组的数据中去学习
         探索每个组的数据是否有联系             把新数据放到已经分好的组中去
         
学习类型  有监督，需要标签进行训练               无监督，无需标签进行训练

典型算法  K-Means，DBSCAN，层次聚类，光谱聚类      决策树，贝叶斯，逻辑回归

算法输出  聚类结果是不确定的                      分类结果是确定的
         不一定总是能够反映数据的真实分类          分类的优劣是客观的
         同样的聚类，根据不同的业务需求            不是根据业务或算法需求决定
         可能是一个好结果，也可能是一个坏结果
'''


'''1.2 sklearn中的聚类算法：
       类： cluster.AffinityPropagation  / cluster.KMeans / Birch / DBSCAN / FeatureAgglomeration / MiniBatchKMeans / SpectralClustering / MeanShift
     函数:  cluster.affinity_propagation
'''
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


''' 2 KMeans'''
'''2.1 KMeans 是如何工作的
        簇与质心：KMeans算法将一组N个样本的特征矩阵X划分为K个无交集的簇，直观上来看簇是一组一组聚集在一起的数据。
                在一个簇中的数据被认为是一类，簇就是聚类的结果表现。
                簇中所有数据的均值μ通常被称为这个簇的"质心" （centroids)。 在一个二维平面中，一簇数据点的质心的横坐标就是这一簇数据点的横坐标的均值
    在KMeans算法中，簇的个数K是一个超参数，找出K个最优的质心，并将离这些质心最近的数据分别分配到这些质心代表的簇中去。
    具体过程如下：    
        顺序      过程
        1       随机抽取K个样本作为最初的质心
        2       开始循环:
        2.1     将每个样本点分配到离他们最近的质心，生成K个簇
        2.2     对于每个簇，计算所有被分到该簇的样本点的平均值作为新的质心
        3       当质心的位置不再发生变化，迭代停止，聚类完成
    
    2.2 簇内误差平方和的定义和解惑
    
    2.3 KMeans算法的时间复杂度
        除了模型本身的效果之外，我们还使用另一种角度来度量算法：算法复杂度。算法的复杂度分为时间和空间复杂度。
        和KNN一样，KMeans算法是一个计算成本很大的算法，KMeans算法的平均复杂度是O(k*n*T)，
        其中k是我们的超参数，所需要输入的簇数，n是整个数据集中的样本量， T是所需要的迭代次数(相对的，KNN的平均复杂度是O(n))。
'''

'''3 sklearn.cluster.KMeans

3.1 重要参数n_clusters
n_clusters是KMeans中的k，表示着我们告诉模型我们要分几类。
这是KMeans当中唯一一个必填的参数，默认为8类，但通常我们的聚类结果会是一个小于8的结果。通常，在开始聚类之前，我们并不知道n_clusters究竟是多少， 因此我们要对它进行探索。
3.1.1 先进行一次聚类看看吧
'''

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#自己创建数据集
##x: 2维特征矩阵（取决于n_features)， y: 1维标签
##random_state=1 保证每次随机生成的数据一样
x,y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)
#如果我们想要看见这个点的分布，怎么办?
# color = ["red","pink","orange","gray"]  ##centers=4,因为make_blobs生成 的就是是4个center
# fig, ax1 = plt.subplots(1)
# for i in range(4):
#     ax1.scatter(
#          x[y==i, 0]
#         ,x[y==i, 1]
#         ,marker='o'  #点的形状
#         ,s=8 #点的大小
#         ,c=color[i]
#     )
# plt.show()

n_clusters = 3
cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(x)
##重要属性labels_, 查看聚好的类别，每个样本所对应的类
y_pred = cluster.labels_
#print(y_pred)
'''KMeans因为并不需要建立模型或者预测结果，因此我们只需要fit就能够得到聚类结果了
KMeans也有接口predict和fit_predict，表示学习数据X并对X的类别进行预测
但所得到的结果和我们不调用 predict，直接fit之后调用属性labels一模一样'''
pre = cluster.fit_predict(x)
#print(pre == y_pred)

'''那为什么还需要提供predict / fit_predict功能呢？
因为：当数据量太大时，一开始我们可以不用全部数据来寻找质心(fit方法）,少量的数据就可以帮助我们确定质心了
剩下的数据的聚类结果，使用predict来调用
举例： 
'''
cluster_smallsub = KMeans(n_clusters=n_clusters, random_state=0).fit(x[:200])
y_pred_ = cluster_smallsub.predict(x)
##但是这里，其结果未必与全部数据时一致，有可能是因为本身我们的数据集太小相关。 如果是大量数据，结果会相对比较接近
##但这种分数据集来fit的方法肯定与全部数据一起fit的结果是不太一致的。
##print(y_pred == y_pred_) ##这时候大部分的结果是true

##质心
centroid = cluster.cluster_centers_
# print(centroid)
# print(centroid.shape)
##Inertia 簇内平方和
# inertia = cluster.inertia_ ##这个公式被称为簇内平方和(cluster Sum of Square)， 又叫做Inertia。
# print(inertia)

# color = ["red","pink","orange"]
# fig, ax1 = plt.subplots(1)
# for i in range(n_clusters):
#     ax1.scatter(x[y_pred==i, 0], x[y_pred==i, 1]
#             ,marker='o'
#             ,s=8
#             ,c=color[i]
#            )
#     ax1.scatter(centroid[:,0]
#                 ,centroid[:,1]
#                ,marker="x"
#                ,s=15
#                ,c="black")
# plt.show()


'''但是我们怎么知道聚类效果好不好？或者说n_cluster是否合理
  原则1：inertia应尽量小 (然而，当n_clusters越大，inertia肯定也越小）
        因此我们不能说调整了n_clusters就说模型变好了，而是在n_cluster不变的情况下调整
'''
n_clusters = 4
cluster_ = KMeans(n_clusters=n_clusters, random_state=0).fit(x)
inertia_ = cluster_.inertia_
#print(inertia_) ##比n_cluster=3的时候小

n_clusters = 5
cluster_ = KMeans(n_clusters=n_clusters, random_state=0).fit(x)
inertia_ = cluster_.inertia_
#print(inertia_) ##比n_cluster=4的时候小

n_clusters = 6
cluster_ = KMeans(n_clusters=n_clusters, random_state=0).fit(x)
inertia_ = cluster_.inertia_
#print(inertia_) ##比n_cluster=5的时候小

'''########'''
'''3.1.2 聚类算法的模型评估指标
什么样的模型评估指标才是好呢？
    聚类模型的结果不是某种标签输出，并且聚类的结果是不确定的，其优劣由业务需求或者算法需求来决定，
    并且没有永远的正确答案。那我们如何衡量聚类的效果呢?
KMeans的目标是确保“簇内差异小，簇外差异大”，我们就可以通过衡量簇内差异来衡量聚类的效果。
'''

'''3.1.2.1 当真实标签已知时
当数据中带有真实的标签数据：
模型评估指标                                                                                           说明

互信息分
    普通互信息分 metrics.adjusted_mutual_info_score (y_pred, y_true)                                  取值范围在(0,1)之中 越接近1，聚类效果越好 在随机均匀聚类下产生0分
    调整的互信息分 metrics.mutual_info_score (y_pred, y_true)
    标准化互信息分 metrics.normalized_mutual_info_score (y_pred, y_true)


V-measure:基于条件上分析的一系列直观度量
    同质性:是否每个簇仅包含单个类的样本 metrics.homogeneity_score(y_true, y_pred)                         取值范围在(0,1)之中         
    完整性:是否给定类的所有样本都被分配给同一个簇中 metrics.completeness_score(y_true, y_pred)              越接近1，聚类效果越好 由于分为同质性和完整性两种度量，可以更仔细地研究，模型到底哪个任务 做得不够好                     
    同质性和完整性的调和平均，叫做V-measure metrics.v_measure_score(labels_true, labels_pred)             对样本分布没有假设，在任何分布上都可以有不错的表现 在随机均匀聚类下不会产生0分                      
    三者可以被一次性计算出来: metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)                                  

调整兰德系数 metrics.adjusted_rand_score(y_true, y_pred)                                                取值在(-1,1)之间，负值象征着簇内的点差异巨大，甚至相互独立，
                                                                                                      正类的兰德系数比较优秀，越接近1越好 对样本分布没有假设，在任何分布上都可以有不错的表现，尤其是在具 有"折叠"形状的数据上表现优秀

'''
'''3.1.2.2 当真实标签未知的时候:轮廓系数 sklearn.metrics.silhouette_score
99%的情况下，标签是未知的。
轮廓系数是最常用的聚类算法的评价指标。它是对每个样本来定义的：
1)样本与其自身所在的簇中的其他样本的相似度a，等于样本与同一簇中所有其他点之间的平均距离 
2)样本与其他簇中的样本的相似度b，等于样本与下一个最近的簇中的所有点之间的平均距离 根据聚类的要求”簇内差异小，簇外差异大“，我们希望b永远大于a，并且大得越多越好。
轮廓系数聚会在(1,-1)之间，（1, 0)之间代表聚类好，（0，-1）之间代表聚类不好。
'''
#from sklearn.metrics import silhouette_score
#from sklearn.metrics import silhouette_samples
#print(silhouette_score(x,y_pred))
#print(silhouette_score(x,cluster_.labels_))


'''3.1.2.3 当真实标签未知的时候：Calinski-Harabaz Index
除了轮廓系数是最常用的，我们还有Calinski-Harabaz Index(CHI 方差比标准） 、 Davies-Bouldin（戴维斯-布尔丁指数） 、Contingency Matrix(权变矩阵）可以使用

卡林斯基-哈拉巴斯指数 sklearn.metrics.calinski_harabaz_score (X, y_pred)   （值越高越好）
 戴维斯-布尔丁指数 sklearn.metrics.davies_bouldin_score (X, y_pred)
权变矩阵sklearn.metrics.cluster.contingency_matrix (X, y_pred)

'''
#from sklearn.metrics import calinski_harabasz_score

'''3.1.3 案例： 基于轮廓系数来选择n_clusters'''
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm  #colormap

for n_clusters in [2,3,4,5,6,7]:  #画出当n_clusters在取值范围内，所有簇的情况
    n_clusters = n_clusters

    #创建画布，画布上共有一行两列的2个图
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7) #plt.figure(figsize=(18, 7))
    '''第一个图是我们的轮廓系数图像，是由各个簇的轮廓系数组成的横向条形图
    横坐标x是轮廓系数取值[-1,1]， 纵坐标y是每个样本
    '''
    ax1.set_xlim([-0.1, 1])  ###取较好的轮廓系数取值范围[-0.1,1], <-0.1的就不显示了。
    ax1.set_ylim([0, x.shape[0]+(n_clusters+1)*10])


    ##开始建模
    clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(x)
    cluster_labels = clusterer.labels_
    silhouette_avg = silhouette_score(x, cluster_labels)  ##返回整体轮廓系数
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(x, cluster_labels)  ##返回每个样本点的轮廓系数，作为横坐标
    #print(sample_silhouette_values)

    y_lower = 10  #y轴初始取值
    #接下来，对4个簇中的每一个簇进行循环
    for i in range(n_clusters):
        #从每个样本的轮廓系数结果中抽取出属于第i个簇的所有样本的轮廓系数，并进行排序（仅为了显示好看）
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        #查看这一个簇中有多少个样本
        size_cluster_i = ith_cluster_silhouette_values.shape[0]

        #当前簇在y轴上的取值，应该是由初始值 + 当前簇的样本数据
        y_upper = y_lower + size_cluster_i

        ##colormap库中，是使用nipy_spectral(浮点数)来调用颜色的函数
        ##我们希望每一个簇的颜色都不一样，float(i)/n_clusters确保每一个簇的值都 不一样，就确保了颜色不一样
        color = cm.nipy_spectral(float(i)/n_clusters)

        #开始填充ax1中的内容
        ax1.fill_betweenx(np.arange(y_lower, y_upper)
                          ,ith_cluster_silhouette_values
                          ,facecolor=color
                          ,alpha=0.7
                          )
        ax1.text(-0.05
                 ,y_lower + .5 * size_cluster_i
                 ,str(i))
        y_lower = y_upper+10
    ax1.set_title('the silhouette plot for the vairous custers.')
    ax1.set_xlabel('the silhouette coef values')
    ax1.set_ylabel('Cluster label')

    ax1.axvline(x=silhouette_avg, color='red', linestyle='--')
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    colors = cm.nipy_spectral(cluster_labels.astype(float)/n_clusters)

    ax2.scatter(x[:, 0], x[:,1]
                ,marker='o'
                ,s=8
                ,c=colors)
    centers = clusterer.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1],
                marker = 'x'
                ,c='red'
                ,alpha=1
                ,s=200)
    ax2.set_title('the visualization of the clustered data.')
    ax2.set_xlabel('Feature space for the 1st feature')
    ax2.set_ylabel('Feature space for the 2nd feature')
    plt.suptitle('silhouette analysis for KMeans clustering on sample data with n_clusters= %d' % n_clusters
                 ,fontsize=14, fontweight='bold')
plt.show()










