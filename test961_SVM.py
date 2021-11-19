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
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import make_circles,make_blobs,make_classification,make_moons
# def plot_svc_decision_function(model,ax=None):
#     if ax is None:
#         ax = plt.gca()#获取当前的子图，如果不存在，则创建新的子图
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#     x = np.linspace(xlim[0],xlim[1],30)
#     y = np.linspace(ylim[0],ylim[1],30)
#     #制作网格函数
#     Y,X = np.meshgrid(y,x)
#     xy = np.vstack([X.ravel(), Y.ravel()]).T
#     P = model.decision_function(xy).reshape(X.shape)
#     #contour是专门用于绘制等高线的函数。
#     ax.contour(X, Y, P,colors="k",levels=[-1,0,1],alpha=0.5,linestyles=["--","-","--"])
#     ax.set_xlim(xlim)
#     ax.set_ylim(ylim)
# X,y = make_circles(100, factor=0.1, noise=.1)
# ##同样环形数据，使用kernel = 'rbf' ，决策边界被完美地找了出来
# clf = SVC(kernel = "rbf").fit(X,y)
# plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow")
# plot_svc_decision_function(clf)
# plt.show()


'''2.2.3 探索核函数在不同数据集上的表现
'''

##创建数据集，定义核函数的选择
n_samples = 100
datasets = [
    make_moons(n_samples=n_samples, noise=0.2, random_state=0),
    make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=1),
    make_blobs(n_samples=n_samples, centers=2, random_state=5),
    make_classification(n_samples=n_samples,n_features =2,n_informative=2,n_redundant=0, random_state=5)
    ]
Kernel = ["linear","poly","rbf","sigmoid"]
#四个数据集分别是什么样子呢?
# for X,Y in datasets:
#     plt.figure(figsize=(5,4))
#     plt.scatter(X[:,0], X[:, 1], c=Y, s=50, cmap='rainbow')
# plt.show()

#构建子图
nrows = len(datasets)
ncols = len(Kernel) + 1
fig, axes = plt.subplots(nrows, ncols, figsize=(20, 16))
#开始进行子图循环

#第一层循环:在不同的数据集中循环
for ds_cnt, (X,Y) in enumerate(datasets):
    #在图像中的第一列，放置原数据的分布
    ax = axes[ds_cnt, 0]
    if ds_cnt == 0:
        ax.set_title("Input data")
    ax.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,edgecolors='k')
    ax.set_xticks(())
    ax.set_yticks(())
    #第二层循环:在不同的核函数中循环
    #从图像的第二列开始，一个个填充分类结果
    for est_idx, kernel in enumerate(Kernel):
        #定义子图位置
        ax = axes[ds_cnt, est_idx + 1]
        #建模
        clf = SVC(kernel=kernel, gamma=2).fit(X, Y)
        score = clf.score(X, Y)
        #绘制图像本身分布的散点图
        ax.scatter(X[:, 0], X[:, 1], c=Y,zorder=10,cmap=plt.cm.Paired,edgecolors='k') #绘制支持向量
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=50,facecolors='none', zorder=10, edgecolors='k')
        #绘制决策边界
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        #np.mgrid，合并了我们之前使用的np.linspace和np.meshgrid的用法 #一次性使用最大值和最小值来生成网格
        #表示为[起始值:结束值:步长] #如果步长是复数，则其整数部分就是起始值和结束值之间创建的点的数量，并且结束值被包含在内
        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        # np.c_，类似于np.vstack的功能
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
        # 填充等高线不同区域的颜色
        ax.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
        #绘制等高线
        #levels等高线距离相同的都连接到一起
        ax.contour(XX, YY, Z, colors=['k', 'k', 'k']
                   , linestyles=['--', '-', '--']
                   , levels=[-1, 0, 1])
        #设定坐标轴为不显示
        ax.set_xticks(())
        ax.set_yticks(())
        #将标题放在第一行的顶上
        if ds_cnt == 0:
            ax.set_title(kernel)
        #为每张图添加分类的分数
        #在每张子图上添加文字
        ax.text(0.95, 0.06, ('%.2f' % score).lstrip('0')
                , size=15
                , bbox=dict(boxstyle='round', alpha=0.8, facecolor='white')
                # 为分数添加一个白色的格子作为底色
                , transform=ax.transAxes #确定文字所对应的坐标轴，就是ax子图的坐标轴本身
                , horizontalalignment='right' #位于坐标轴的什么方向
        )


plt.tight_layout()
plt.show()

'''
上面图的对比可以看出：
    rbf，高斯径向基核函数基本在任何数据集上都表现不错，属于比较万能的核函数。
    linear线性核函数和poly多项式核函数即便有扰动项也可以表现不错，可见多项式核函数是虽然也可以处理非线性情况，但更偏向于线性的功能。
    Sigmoid核函数就比较尴尬了，它在非线性数据上强于两个线性核函数，但效果明显不如rbf，它在线性数据上完全 比不上线性的核函数们，对扰动项的抵抗也比较弱，所以它功能比较弱小，很少被用到。

我个人的经验是，无论如何先 试试看高斯径向基核函数，它适用于核转换到很高的空间的情况，在各种情况下往往效果都很不错，如果rbf效果 不好，那我们再试试看其他的核函数。另外，多项式核函数多被用于图像处理之中。

'''