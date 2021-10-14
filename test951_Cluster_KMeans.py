'''sklearn中的聚类算法K-Means'''


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

'''3.2 重要参数 init & random_state & n_init ： 初始质心怎么放好？'''
plus = KMeans(n_clusters = 10).fit(x)
plus.n_iter_
random = KMeans(n_clusters = 10,init="random",random_state=420).fit(x)
random.n_iter_










