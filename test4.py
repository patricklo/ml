'''管道pipeline的训练
使用管道可以减少训练步骤；有时候，可以用管道pipeline功能把多个estimator串联起来一次性训练数据。
管道的原理是把上一个estimator的输出 作为 下一个estimator的输入
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
'''将boston房价问题的数据预处理、数据降维 和 回归分析过程构建成pipeline'''
boston=datasets.load_boston()
x_train,x_test,y_train,y_test=train_test_split(boston.data,boston.target,
                                test_size=1/3,random_state=0)
#构建管道，使用make_pipeline函数
pipe=Pipeline([('scaler',preprocessing.MinMaxScaler()),('pca',PCA()),('net',ElasticNetCV())])
#可以在管理定义时设置参数，也可以统一设置参数，注意参数的名字的调整方式
pipe.set_params(scaler__feature_range=(0,1),pca__n_components=6)
#用数据喂养管理
pipe.fit(x_train,y_train)
##使用管理进行预测
#print(pipe.predict(x_test))
#print(pipe.get_params())
##查看模型在训练集的得分
#print(pipe.score(x_train,y_train))
##查看模型在测试集的得分
#print(pipe.score(x_test,y_test))
'''使用特征联合FeatureUnion
Pipeline 是estimator的串联，而featureUnion则是estimator的并联。
但FeatureUnion并联的只能是transformer转换器。
FeatureUnion合并了多个转换器对象，形成一个新的转换器，该转换器合并了他们的输出。
可以结合FeatureUnion和Pipeline来创建更加复杂的模型
'''
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
boston = datasets.load_boston()
###3+5 transform之后会有8个特征
united=FeatureUnion([('linear_pca',PCA(n_components=3)),
                     ('kernel_pca',KernelPCA(n_components=5))])
print(united.fit_transform(boston.data).shape)




'''管道pipeline的训练
使用管道可以减少训练步骤；有时候，可以用管道pipeline功能把多个estimator串联起来一次性训练数据。
管道的原理是把上一个estimator的输出 作为 下一个estimator的输入
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
'''将boston房价问题的数据预处理、数据降维 和 回归分析过程构建成pipeline'''
boston = datasets.load_boston()
x_train,x_test,y_train,y_test=train_test_split(boston.data,boston.target,
                                            test_size=1/3,random_state=0)
#构建管道，使用make_pipeline函数
pipe =Pipeline([('scaler',preprocessing.MinMaxScaler()),('pca',PCA()),('net',ElasticNetCV())])
#可以在管理定义时设置参数，也可以统一设置参数，注意参数的名字的调整方式
pipe.set_params(scaler__feature_range=(0,1),pca__n_components=6)
#用数据喂养管理
pipe.fit(x_train,y_train)
##使用管理进行预测
#print(pipe.predict(x_test))
#print(pipe.get_params())
##查看模型在训练集的得分
#print(pipe.score(x_train,y_train))
##查看模型在测试集的得分
#print(pipe.score(x_test,y_test))
'''使用特征联合FeatureUnion
Pipeline 是estimator的串联，而featureUnion则是estimator的并联。
但FeatureUnion并联的只能是transformer转换器。
FeatureUnion合并了多个转换器对象，形成一个新的转换器，该转换器合并了他们的输出。
可以结合FeatureUnion和Pipeline来创建更加复杂的模型
'''
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
boston = datasets.load_boston()
###3+5 transform之后会有8个特征
united=FeatureUnion([('linear_pca',PCA(n_components=3)),
                     ('kernel_pca',KernelPCA(n_components=5))])
print(united.fit_transform(boston.data).shape)

