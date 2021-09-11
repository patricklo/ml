import pandas as pd
import numpy as np
'''5.處理缺失值
因为各种原因，数据可能有缺失，这类数据常表现为nan/NaN等，这些数据不能与sklearn学习算法兼容
处理这些缺失值有2种方向：
1.舍弃这些缺失的值，用Pandas dropna等方法
2.使用sklearn 内置方法，对缺失值按照给定策略进行填充：
2.1 从已有的数据推断出缺失的值
'''
from sklearn.impute import SimpleImputer
#策略为mean/most frequent/median
#allowed_strategies =["mean","median","most_frequent","constant"]
imp=SimpleImputer(strategy='mean')
imp.fit([[1,2],[np.nan,3],[7,6]])
#fit求得第一列特征均值为4,第二列特征均值为11/3
x=[[np.nan,2],[6,np.nan],[7,6]]
#print(imp.transform(x))
'''6.生成多项式特征？
针对模型在训练集中得分过低，即欠拟合问题：可以通过增加模型的复杂度－》增加特征的数量，
从原为特征中，自动生成新增一些非线性的特征
在机器学习中，通过增加一些输入数据的非线性特征来增加模型的复杂度通常是有效的，
一个简单通用的方法是使用多项式特征，这可以获得特征的更高次数和互相间关系的项
如现有：X1,X2线性关系 x2=a*x1+b=》x1^2 x2^2 x1*x2 增加平方数 x1^2+x2^2=1
'''
from sklearn.preprocessing import PolynomialFeatures
x=np.arange(6).reshape(3,2)
poly=PolynomialFeatures(2)
##x 原特征为（x1,x2)
#print(x)
##x新特征为（1,x1,x2,x1^2,x1*x2,x2^2)
#print(poly.fit_transform(x))
##有些情况下，只需特征间的交互项，
#可以设置intraction_only=True
##x新特征为 (1,x1,x2,x1*x2)

poly=PolynomialFeatures(2,interaction_only=True)
#print(poly.fit_transform(x))
'''7.特征的提取
简单的特征提取方法：
1.字典加载特征 DictVectorizer
对输出的分类特征会采用独热编码one=hot
2.文本特征提取：词频向量（CountVectorizer)TF=IDF向量（TfidfVectorize,TfidfTransformer)特征哈希向量
3.图像特征提取：提取像素矩阵 提取边缘和兴趣点
'''
###7.1 字典加载特征
measurement = [
    {'city':'Dubai','temperature':33.},
{'city':'London','temperature':12.},
{'city':'Beijing','temperature':18.}
]
from sklearn.feature_extraction import DictVectorizer
vec=DictVectorizer()
#print(vec.fit_transform(measurement).toarray())
#print(vec.get_feature_names())
#7.2 文本特征提取
##7.2.1字频向量（CountVectorizer)
###词库模型（Bag=of=words model)是文字模型化最常用方法，它为每个单词设置一个特征值。依据是用类似单词的文章，意思也差不多
###CountVector类会将文章全部转换成小写，然后把句子分割成词块（token,通常是单词）或有意义的字母序列，并统计它们出现的次数。
###词块大多数是单词，也有可能是一些短语，字母长度小于2的词块（如I,a)会被略去
from sklearn.feature_extraction.text import CountVectorizer
corpus=[
'UNC played Duke in basketball',
'Duke lost the basketball game,game over',
'I ate a sandwich'
]
vectorizer = CountVectorizer(stop_words='english')##stop_words:省略统计的单词
#print(vectorizer.fit _transform(corpus))
#print(vectorizer.fit transform(corpus).todebse())##todense=》将稀疏矩阵转换成普通矩阵
##单词以及单词在数组中位置
#print(vectorizer.vocabulary_)
#对于中文 使用jieba中文分词模块
import jieba
seg_list = jieba.cut('朋友，小红是我的')
corpus=[
'朋友，小红是我的',
'小明对小红说：“小红，我们还是不是朋友',
'小明与小红是朋友']

cut_corpus = ['/'.join(jieba.cut(x))for x in corpus]
#print(cut_corpus)
vectorizer=CountVectorizer(stop_words=['好的','是的'])
counts =vectorizer.fit_transform(cut_corpus).todense()
#print(counts)
#print(vectorizer.vocabulary_)
##如何衡量2个文档之间的距离（距离越小越相似）－
###可以用词频向量的欧式距离（L2范数）来衡量
from sklearn.metrics.pairwise import euclidean_distances
vectorizer=CountVectorizer()
for x,y in [[0,1],[0,2],[1,2]]:
    dist = euclidean_distances(counts[x],counts[y])
    #print('文档［｛｝］与文档［｛｝］的距离｛｝＊。format(corpus[x],corpus[y],dist))
##7.2.2 Tf=idf权重向量
###单词频率对文档意思有重要作用，但在对比长度不同的文档时，长度较长的文档的单词词频将明显变大。
###因此使用tf=idf将单词词频转化为权重是一个好主意
###此外，如果一些词语在需要分析的所有文档中都出现，那么可以认为这些词是文档中的常用词，对区分文件中的文档帮助不大。
###因此，可以把单词在文集中出现的频率考虑进来进行修正。＜－tf=idf、
#TfidfTransformer将CountVectorizer产生的词频向量 转换为 tf=id权重向量
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfTransformer
#TfidfVectorizer:集成CountVectorizer 和TfidfTransformer功能
from sklearn.feature_extraction.text import TfidfVectorizer
#counts=[[3,0,1],
#［2,0,0],
#［3,0,0],
#［4,0,0],
#［3,2,0],
#［3,0,2],
#］
transformer = TfidfTransformer(smooth_idf=False)
tfidf =transformer.fit_transform(counts)
#print(tfidf.toarray())
vectorizer = TfidfVectorizer()
#print(vectorizer.fit_transform(cut_corpus).toarray())
#print(vectorizer.vocabulary_)
##7.2.3 特征哈希向量
### 上面讲的词频相关的模型很好用，也很直接，但在有些场景下很难使用，比如分词后的词汇字典表非常大，达到100万＋，
### 此时如果直接使用词频向量或tf=id权重向量，将对应的样本特征导入内存，有可能将内存撑爆，在这种情况下，我们就需要使用哈希向量
### Hash函数可以将任意长度的字符串转换成固定长度的散列数字。MD5/MD4,SHA
from sklearn.feature_extraction.text import HashingVectorizer
corpus = ['smart boy', 'ate','bacon','a cat']
#HasingVectorizer是无状态的，因此需要fit
#n feature 向量维度
vectorizer = HashingVectorizer(n_features=6,stop_words='english')
#print(vectorizer.transform(corpus).todense())
corpus=[
'UNC played Duke in basketball',
'Duke lost the basketball game, game over',
'I ate a sandwich'
    ]
vectorizer =HashingVectorizer(n_features=10)
counts =vectorizer.transform(corpus).todense()
print(counts)
print(counts.shape)
##7.2.4 图片特征提取
###图片特征提取的基本方法是获取图片像素矩阵，并将其reshape拼接成一个行向量
import skimage.io as io
import matplotlib.pyplot as plt
imrgb = io.imread('123.jpg')
print('before reshape:',imrgb.shape)
##将imrgb 拼接成为一个行向量
imvec=imrgb.reshape(1,-1)
print('after reshape:',imvec.shape)
#plt.gray()
#fig,axes =plt.subplots(2,2,figsize=(12,10))
#ax0,ax1,ax2,ax3=axes.ravel()
#ax0.imshow(imrgb)
#ax0.set_title('original image')
#ax1.imshow(imrgb[:,:,0])##R通道 red
#ax1.set_title('red channel')
#ax2.imshow(imrgb[:,:,1])##
#ax2.set_title('green channel')
#ax3.imshow(imrgb[:,:,2])##
#ax3.set_title('blue channel')
#plt.show()
#转换成为黑白图像
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist
plt.gray()
imgray=equalize_hist(rgb2gray(imrgb))
io.imshow(imgray)
plt.show()
imvec=imgray.reshape(1,-1)
##print(imvec.shape)











