'''降维算法：PCA


   主成分分析
  sklearn.decomposition.PCA                     主成分分析（PCA）
  sklearn.decomposition.IncrementalPCA          增量主成分分析（IPCA）
  sklearn.decomposition.KernelPCA               核主成分分析（KPCA)
  sklearn.decomposition.MiniBatchSparserPCA     小批量稀疏主成分分析
  sklearn.decomposition.SparsePCA               稀疏主成分分析（SparsePCA)
  sklearn.decomposition.TruncatedSVD            截断的SVD(aka LSA)

  因子分析
  sklearn.decomposition.FactorAnalysis          因子分析（FA）

  独立成分分析
  sklearn.decomposition.FastICA

  字典学习
  sklearn.decomposition.DictionaryLearning               字典学习
  sklearn.decomposition.MiniBatchDictionaryLearning      小指字典学习
  sklearn.decomposition.dict_learning                    字典学习用于矩阵分解
  sklearn.decomposition.dict_learning_online             在线字典学习用于矩阵分解

  高级矩阵分解（
  sklearn.decomposition.LatentDirichletAllocation        具有在线变分贝叶斯算法的隐含狄利克雷分布
  sklearn.decomposition.NMF                              非负矩阵分析（NMF）


降维：
    PCA / SVD 是降维算法中入门的算法，
    在降维过程中，我们会减少特征的数量，这意味着删除数据，数据量变少则表示模型可以获取的信息会变少，模型的表现可能会因此受影响。
    同时，在高维度的数据中，必须也是有一些特征是不带有有效信息的（如：噪音数据），或者有一些特征带有的信息是和其他特征一起重复的（如：线性相关）

    我们希望能够找出一种办法来帮助我们衡量特征上所带的信息量，让我们在降维的过程中，能够即减少特征的数量，又保留大部分有效信息
        1. 将那些带有重复信息的特征合并，
        2. 并删除那些带无效信息的特征等等
    最后逐渐创造出能够代表原特征矩阵大部分信息的，特征更少的，新特征矩阵。

    在上文（方差过滤）,我们知道，如果一个特征的方差很小，则意味着这个特征上的数据取值很有可能是相同的，那这个特征的取值对样本而言就没有区分度，不带有有效信息。
    从方差的这种应用我们就可以推断出，如果一个特征的方差很大， 则说明这个特征上带有大量信息。
    因此，在降维中，PCA使用的信息量衡量指标，就是样本的方差，又称可解释性方差，方差越大，特征所带的信息量越多。

    Var = ∑（x^-x~)**2 / n-1 (方差计算公式中为什么除数是n-1? 这是为了得到样本方差的无偏估计)

'''

'''
思考:PCA和特征选择技术都是特征工程的一部分，它们有什么不同?

    1. 特征工程中有三种方式:特征提取，特征创造和特征选择。仔细观察上面的降维例子和上周我们讲解过的特征 选择，你发现有什么不同了吗?
        特征选择是从已存在的特征中选取携带信息最多的，选完之后的特征依然具有可解释性，我们依然知道这个特征在原数据的哪个位置，代表着原数据上的什么含义。
    2. 而PCA，是将已存在的特征进行压缩，降维完毕后的特征不是原本的特征矩阵中的任何一个特征，而是通过某些方式组合起来的新特征。
        通常来说，在新的特征矩阵生成之前，我们无法知晓PCA都建立了怎样的新特征向量，新特征矩阵生成之后也不具有可读性，我们无法判断新特征矩阵的特征是从原数据中的什么特征组合而来，新特征虽然带有原始数据的信息，却已经不是原数据上代表着的含义了。 
        以PCA为代表的降维算法因此是特征创造(feature creation，或feature construction)的一种。
        可以想见，PCA一般不适用于探索特征和标签之间的关系的模型(如线性回归)，因为无法解释的新特征和标签之间的关系不具有意义。
        在线性回归模型中，我们使用特征选择。

'''

'''2.1 降维究竟是怎样实现？


'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import  load_iris
from sklearn.decomposition import PCA

iris = load_iris()
y = iris.target
X = iris.data #作为数组，X是几维?
print(X.shape) #shape= (150,4),X作为数组，是个2维数组
'''数组维度 != 特征矩阵维度
但作为特征矩阵(数据表)，X是几维? 把x放到dataframe中，就可以看出是4维
       0    1    2    3
0    5.1  3.5  1.4  0.2
1    4.9  3.0  1.4  0.2
2    4.7  3.2  1.3  0.2
'''
#print(pd.DataFrame(X))

'''## 2.2 重要参数n_components： n_components的可选值'''
'''### 2.2.1 n_components=最优值  --  如何得出最优n_components， 迷你案例:高维数据的可视化'''
#调用PCA
pca = PCA(n_components=2)
pca = pca.fit(X)
X_dr = pca.transform(X)
#print(type(X_dr))
#print(X_dr)
#也可以fit_transform一步到位 #X_dr = PCA(2).fit_transform(X)
#实例化 #拟合模型 #获取新矩阵
'''s4. 可视化'''
#要将三种鸢尾花的数据分布显示在二维平面坐标系中，对应的两个坐标(两个特征向量)应该是三种鸢尾花降维后的 x1和x2，怎样才能取出三种鸢尾花下不同的x1和x2呢?
#X_dr[y == 0, 0] #这里是布尔索引，看出来了么? https://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/AdvancedIndexing.html
#print(y==0)
# plt.figure()
# plt.scatter(X_dr[y==0, 0], X_dr[y==0, 1], c="red", label=iris.target_names[0])
# plt.scatter(X_dr[y==1, 0], X_dr[y==1, 1], c="black", label=iris.target_names[1])
# plt.scatter(X_dr[y==2, 0], X_dr[y==2, 1], c="orange", label=iris.target_names[2])
# plt.legend()
# plt.title('PCA of IRIS dataset')
# plt.show()
'''s6. 探索降维后的数据'''
#属性explained_variance_, 查看降维后每个新特征向量上所带的信息量大小（可解释性方差的大小）
#[4.22824171 0.24267075] 可见大部分信息都汇集在第一个新特征上
#print(pca.explained_variance_)
#属性explained_variance_ratio，查看降维后每个新特征向量所占的信息量占原始数据总信息量的百分比
#又叫做可解释方差贡献率
#print(pca.explained_variance_ratio_)
#大部分信息都被有效地集中在了第一个特征上
#print(pca.explained_variance_ratio_.sum())
'''s7. 选择最好的n_components： 累积可解释方差贡献率曲线
当参数n_components中不填写任何值，则默认返回X原来的特征数（会取min(X.shape)，样本总量行数和特征数，一般样本总量都是远大于特征数的），
所以什么都不填就相当于转换了新特征空间，但没有减少特征的个数。
'''
#pca_line = PCA().fit(X)
#plt.plot([1,2,3,4],np.cumsum(pca_line.explained_variance_ratio_))
#plt.xticks([1,2,3,4]) #这是为了限制坐标轴显示为整数
#plt.xlabel("number of components after dimension reduction")
#plt.ylabel("cumulative explained variance ratio")
#plt.show()


'''### 2.2.2 n_components='mle' -- 最大似然估计自选超参数'''
'''除了输入整数，n_components还有其它的选择。
之前我们提到过，矩阵分解的理论发展在业界独树一帜，勤奋智慧的数学大神Minka, T.P.在麻省理工学院媒体实验室做研究时找出了让PCA用最大似然估计(maximum likelihood estimation)自选超参数的方法，
输入“mle”作为n_components的参数输入，就可以调用这种方法。
'''
pca_mle = PCA(n_components="mle")
pca_mle = pca_mle.fit(X)
X_mle = pca_mle.transform(X)
#可以发现，mle为我们自动选择了3个特征
#print(X_mle)
#得到了比设定2个特征时更高的信息含量，对于鸢尾花这个很小的数据集来说，3个特征对应这么高的信息含量，并不需要去纠结于只保留2个特征，毕竟三个特征也可以可视化
#print(pca_mle.explained_variance_ratio_.sum())

'''### 2.2.3 n_component=[0,1]的浮点数（代表降维后的总解释性方差百分比），
         svd_solver =='full'    ---  按信息量占比选超参数'''
'''
输入[0,1]之间的浮点数，并且让参数svd_solver =='full'，表示希望降维后的总解释性方差占比大于n_components 指定的百分比，
即是说，希望保留百分之多少的信息量。
比如说，如果我们希望保留97%的信息量，就可以输入 n_components = 0.97，PCA会自动选出能够让保留的信息量超过97%的特征数量。
'''

pca_f = PCA(n_components=0.97,svd_solver="full")
pca_f = pca_f.fit(X)
X_f = pca_f.transform(X)
#print(pca_f.explained_variance_ratio_)
# import numpy as np
# pca_line = PCA().fit(X) plt.plot([1,2,3,4],np.cumsum(pca_line.explained_variance_ratio_)) plt.xticks([1,2,3,4]) #这是为了限制坐标轴显示为整数
# plt.xlabel("number of components after dimension reduction") plt.ylabel("cumulative explained variance ratio")
# plt.show()

'''## 2.3 PCA中的SVD（奇异值分解）'''
'''### 2.3.1 svd_solver="full"  PCA中的SVD从哪里来？？？'''
#print(PCA(2).fit(X).components_)##(2,4)的数组，V(k, n) -> （新特征数，原特征数）  -> 降维过后的新特征空间
#print(pd.DataFrame(PCA(2).fit(X).components_))
#print(PCA(2).fit(X).components_.shape)

'''### 2.3.2 svd_solver参数的取值'''
'''
    'auto':
    'full':
    'arpack':
    'randomized':
'''

'''### 2.3.3 重要的属性： components_'''
'''V(k,n)是新特征空间，是我们要将原始数据进行映射的那些新特征向量组成的矩阵。
我们用它来计算新的特征矩阵，但我们希望获取的毕竟是X_dr,而不是V(k,n)
那么V(k,n)到底代表什么呢？里面的数据有什么意义？
    - 我们知道拿特征选择和PCA进行对比，特征选择后的特征矩阵是可解读的（即可逆转的），但PCA降维后是不可逆转的。
    - 也就是说在新的矩阵生成之前，我们不知道PCA建立了怎样的新特征向量。
    - 但是其实，在矩阵分解时，PCA是有目标的:在原有特征的基础上，找出能够让信息尽量聚集的新特征向量。
    - 在sklearn使用的PCA和SVD联合的降维方法中，这些新特征向量组成的新特征空间其实就是V(k,n)。
    - 当V(k,n)是数字 时，我们无法判断V(k,n)和原有的特征究竟有着怎样千丝万缕的数学联系。
    - 但是，如果原特征矩阵是图像，V(k,n)这 个空间矩阵也可以被可视化的话，我们就可以通过两张图来比较，就可以看出新特征空间究竟从原始数据里提取了什么重要的信息
    
    下面来看看，用人脸识别应用来看components_的应用。
'''
#s1 导库
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_lfw_people
#s2
faces = fetch_lfw_people(min_faces_per_person=60)
##faces下面有images/dat/target标签/target_names
print(faces.images.shape)  #图像数据： (1348, 62, 47) 三维数组 ，1348行是矩阵中的图像个数（样本数），62是每个图像的特征矩阵的行，47是每个图像特征矩阵的列
##图像特征矩阵大小是（62，47）= 62*47 = 2914   其实就是图片的像素大小, 也就我们要吧拿（62，47）的数据来可视化
print(faces.data.shape)    #(1348, 2914) 2914个特征 1348行是样本数， 2914列是特征数（将图片的像素转成1维表示，62*47）
X = faces.data
#s3 可视化图像（62，47）
#subplots: 画子图, 4行5列的子图，空白的大画面的大小 ：（8,4)
#只画20张图
#fig, axes = plt.subplots(4, 5
#                        ,figsize=(8,4)
#                        ,subplot_kw={'xticks':[],'yticks':[]}  ##显示子图的坐标，可不用
#                        )
#print(axes.flat)
#axes[0][0].imshow(faces.images[0,:,:]) ##faces.images[0,:,:] 第1维取0，第2，3维取出全部
#for i,ax in enumerate(axes.flat):   ##enumerate 遍历  ，axes.flat降到1维数组
#    ax.imshow(faces.images[i,:,:]    ##获取每个图像的（62，47）的像素矩阵数据
#              ,cmap='gray')
#plt.show()
pca =PCA(150).fit(X)
V= pca.components_     ##相当于是降维后的骨头
#print(V.shape)   ##(150, 2914)降为150个特征，原来是2914
#fig, axes = plt.subplots(4,5,figsize=(8,4),subplot_kw = {"xticks":[],"yticks":[]})
#for i, ax in enumerate(axes.flat):
#    ax.imshow(V[i,:].reshape(62, 47),cmap="gray")     ##2914 reshape成（62，47）

# X_dr = pca.transform(X) ##相当于是降维后的肉体 （1277，150）
# fig, axes = plt.subplots(4,5,figsize=(8,4),subplot_kw = {"xticks":[],"yticks":[]})
# for i, ax in enumerate(axes.flat):
#     ax.imshow(X_dr[i,:].reshape(62,47),cmap="gray")     ##2914 reshape成（62，47）

plt.show()

'''2.4 重要接口 inverse_transform'''
'''2.4.1 人脸识别的inverse_transform'''
face = fetch_lfw_people(min_faces_per_person=60)
X = face.data
pca = PCA(150)
x_dr = pca.fit_transform(X)
X_inverse = pca.inverse_transform(x_dr)      ##对于PCA，inverse_transform并不是将原来的矩阵inverse,而只是升维
#fig, ax = plt.subplots(2, 10
#                        ,figsize=(10,2.5)
#                        ,subplot_kw={'xticks':[],'yticks':[]}  ##显示子图的坐标，可不用
#                        )

'''可以明显看出，这两组数据可视化后，由降维后再通过inverse_transform转换回原维度的数据画出的图像和原数 据画的图像大致相似，但原数据的图像明显更加清晰。这说明，inverse_transform并没有实现数据的完全逆转。 
  这是因为，在降维的时候，部分信息已经被舍弃了，X_dr中往往不会包含原数据100%的信息，所以在逆转的时候，即便维度升高，原数据中已经被舍弃的信息也不可能再回来了。
  所以，降维不是完全可逆的。
  Inverse_transform的功能，是基于X_dr中的数据进行升维，将数据重新映射到原数据所在的特征空间中，而并非恢复所有原有的数据。
  但同时，我们也可以看出，降维到300以后的数据，的确保留了原数据的大部分信息，
  所以 图像看起来，才会和原数据高度相似，只是稍稍模糊罢了。
'''
#for i in range(10):
#    ax[0,i].imshow(face.images[i,:,:],cmap="binary_r")
#    ax[1,i].imshow(X_inverse[i].reshape(62,47),cmap="binary_r")
#plt.show()

'''2.4.2 用PCA做噪音过滤'''
'''利用inverse_transform作噪音过滤处理'''
from sklearn.datasets import load_digits      ##手写数字 图像集
digits = load_digits() ##手写数字 图像集
def plot_digits(data, title):
    fig, axes = plt.subplots(4,10,figsize=(10,4)
                             ,num=title
                            ,subplot_kw = {"xticks":[],"yticks":[]}

                            )
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8,8),cmap="binary")

#画出原有数据形状：
plot_digits(digits.data,'Original')

#加入噪音数据
np.random.RandomState(42)
noisy = np.random.normal(digits.data,2)
plot_digits(noisy, 'Data With Noisy')
#
##PCA降维
pca = PCA(0.5,svd_solver='full').fit(noisy)
x_dr = pca.transform(noisy)
##inverse_transform去升维
without_noise = pca.inverse_transform(x_dr)
plot_digits(without_noise,'Inversed without Noisy')
plt.show()