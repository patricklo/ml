'''sklearn中的聚类算法K-Means'''
'''4. 案例：聚类算法用于降维， KMeans的矢量量化应用（图像/音频压缩）
K-Means聚类最重要的应用之一是非结构数据(图像，声音)上的矢量量化(VQ，压缩)。
非结构化数据往往占用比较多的储存空间，文件本身也会比较大，运算非常缓慢，我们希望能够在保证数据质量的前提下，尽量地缩小非结构化数据的大小，
或者简化非结构化数据的结构。

矢量量化就可以帮助我们实现这个目的。

KMeans聚类的矢量量化 本质是一种降维运用，
但它与我们之前学过的任何一种降维算法的思路都不相同。
    特征选择的降维是直接选取对模型贡献最大的特征；
    PCA的降维是聚合信息；
    而矢量量化的降维是在同等样本量上压缩信息的大小，即不改变特征的数目也不改变样本的数目，只改变在这些特征下的样本上的信息量。
       （即让数据值 = 簇中的质心值）
'''

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm  #colormap
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from sklearn.datasets import load_sample_image
china = load_sample_image("china.jpg")

##3. 决定超参数，数据预处理
n_clusters = 64 #why 64?
#plt.imshow在浮点数上表现非常优异，在这里我们把china中的数据，转换为浮点数，（归一）压缩到【0，1】之间
china = np.array(china, dtype=np.float64) / china.max() #??压缩数据值
w,h,d = original_shape = tuple(china.shape)
assert d==3
image_array = np.reshape(china, (w*h, d))  #3维变为2维，以便KMeans使用


## 4. 对数据进行K-Means的矢量量化(方法1）
image_array_sample = shuffle(image_array, random_state=0)[:1000] ## why need shuffle?? 因为需要先决定质心，因此需要随机选出1000个数据来决定质心
kmeans = KMeans(n_clusters=n_clusters,random_state=0).fit(image_array_sample)
print(kmeans.cluster_centers_) ##查看生成的质心,一共64个质心
labels = kmeans.predict(image_array)
print(labels.shape)  ##predict整个image数组，一共(273280,),
print(labels)  ##labels里面的值就是对应的质心值
image_kmeans = image_array.copy()
for i in range(w*h):
    image_kmeans[i] = kmeans.cluster_centers_[labels[i]]  ###？？ 将原数据替换为对应的质心
#print(image_kmeans)
#print(pd.DataFrame(image_kmeans).drop_duplicates().shape)
image_kmeans = image_kmeans.reshape(w,h,d) ##恢复图片维度


## 5. 对数据进行随机的矢量量化（方法2）
centroid_random = shuffle(image_array, random_state=0)[:n_clusters] ##随机抽取质心
##pairwise_distances_argmin：用来计算x2中每个样本到x1中的每个样本点的距离，并返回 与x2相同形状的，x1中对应的最近的样本点的索引
labels_random = pairwise_distances_argmin(centroid_random, image_array, axis=0) ##
image_random = image_array.copy()
for i in range(w*h):
    image_random[i] = centroid_random[labels_random[i]]
image_random = image_random.reshape(w,h,d)
#6. 将原图，按KMeans矢量量化和随机矢量量化的图像绘制出来
# plt.figure(figsize=(10,10))
# plt.axis('off')
# plt.title('Original image (96,615 colors)')
# plt.imshow(china)
#
# plt.figure(figsize=(10,10))
# plt.axis('off')
# plt.title('Quantized image (64 colors, K-Means)')
# plt.imshow(image_kmeans)
#
# plt.figure(figsize=(10,10))
# plt.axis('off')
# plt.title('Quantized image (64 colors, Random)')
# plt.imshow(image_random)
# plt.show()








