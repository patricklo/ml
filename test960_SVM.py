'''支持向量机 SVM

线性SVM的拉格朗日函数
支持向量机（SVM，也称为支持向量网络），是机器学习领域中获得关注最多的算法。它源于统计学习理论，是我们除了集成算法之外，接触的第一个强学习器。
有多强，几乎囊括胃有监督和无监督学习的所有算法功能。

有监督学习   Linear Support Vector Classification 线性二分类与多分类
            Support Vector Classification(SVC)  非线性二分类和多分类
            Support Vector Regression           普通连续型变量的回归
            Bayesian SVM                        概率型连续变量的回归

无监督学习  Support Vector Clustering(SVC)        支持向量聚类
          One-class SVM                          异常值检测

半监督学习  Transductive Support Vector Machines(TSVM)  转导支持向量机
'''
'''1.1 支持向量机分类器是如何工作的
简单来说：支持向量机的分类方法，是在一组分布中找出一个超平面作为决策边界，
        使模型在数据分类误差尽量接近于小，尤其是在未知数据集上的分类误差（泛化误差）尽量小 

超平面：在几何中，超平面是一个空间的子空间，它是维度比所在空间小一维的空间。
        如果数据空间是３维的，那其超平面就是２维的
        如果数据空间是２维的，那其超平面就是１维的直线

    1维 -  （1） 可以找到多条超平面直线
           （2）当我们把超平面直线往2边平移，直到碰到离这条线最近的数据后停下，这时候这2边的直线之间的距离，我们称之为margin(边际）        
            （相信多维上面的应用也跟1维类似）

支持向量机，就是通过找出边际(margin)最大的决策边界，来对数据进行分类的分类器。也因此，SVM分类器又叫做最大边际（margin)分类器。
（这个过程在二维平面中看起来十分简单，但将上述过程使用数学表达出来，就不是一件简单的事情了。）


2.1.4 线性SVM决策过程的可视化

'''
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
# X,y = make_blobs(n_samples=50, centers=2, random_state=0,cluster_std=0.6)
# plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
def plot_svc_decision_function(model,ax=None):
    if ax is None:
        ax = plt.gca()#获取当前的子图，如果不存在，则创建新的子图
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x = np.linspace(xlim[0],xlim[1],30)
    y = np.linspace(ylim[0],ylim[1],30)
    #制作网格函数
    Y,X = np.meshgrid(y,x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    #contour是专门用于绘制等高线的函数。
    ax.contour(X, Y, P,colors="k",levels=[-1,0,1],alpha=0.5,linestyles=["--","-","--"])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
#clf = SVC(kernel = "linear").fit(X,y)
#plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
#plot_svc_decision_function(clf)
#plt.show()

#clf.predict(X)
#clf.score(X,y)  #=1, 因为前面没有分测试集和训练集
#clf.support_vectors_ ##返回支持向量 （3个）
#clf.n_support_ ##返回每个类中支持向量的个数

'''
#推广到非线性情况（如环形）
#很明显，如果按照前面的代码，画出直线来说是不能很好的划分环形数据集。
#这个时候，如果我们能能够在原来的x,y基础上，添加一个维度r，变成三维，我们可视化化这个数据，来看看添加维度带来的影响
'''

X,y = make_circles(100, factor=0.1, noise=.1)
#很明显，如果按照前面的代码，画出直线来说是不能很好的划分环形数据集。
# clf = SVC(kernel = "linear").fit(X,y)
# plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
# plot_svc_decision_function(clf)
# plt.show()

#这个时候，如果我们能能够在原来的x,y基础上，添加一个维度r，变成三维，我们可视化化这个数据，来看看添加维度带来的影响
#计算r,并将r作为数据的第三维度来将数据升维的过程，被称为"核变换"，即是将数据投影到高维空间中，以寻找能够将数据完美分割的超平面
#为了详细解释这个过程，我们要下面会引入SVM中的核心概念：核函数
r = np.exp(-(X**2).sum(1))
rlim = np.linspace(min(r), max(r), 100)

from mpl_toolkits import mplot3d
#定义绘制3D图像的函数
#elev表示上下旋转的角度
#azim表示平等旋转的角度
'''可以看见，此时此刻我们的数据明显是线性可分的了:我们可以使用一个平面来将数据完全分开，并使平面的上方
的所有数据点为一类，平面下方的所有数据点为另一类。'''
def plot_3D(elev=30,azim=30,X=X,y=y):
    ax = plt.subplot(projection="3d")
    ax.scatter3D(X[:,0],X[:,1],r,c=y,s=50,cmap='rainbow')
    ax.view_init(elev=elev,azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("r")
    plt.show()
plot_3D()