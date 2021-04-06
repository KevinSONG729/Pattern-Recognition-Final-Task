from 数据分析 import data, columns
import 绘制混淆矩阵
import  绘制ROC图
from 数据预处理 import X_train_3d_pca, X_train_2d_pca, X_train_3d_kpca, X_train_2d_kpca, Y_train, Y_validation, X_test_3d_kpca, X_train, X_validation, X_test_2d_kpca
import plotly
import chart_studio
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy.stats import pearsonr
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import log
import operator
import re
from collections import Counter

algothm_list = ['decison tree', 'decision tree kpca','knn', 'knn_zip', 'fisher', 'bayes', 'svm_rbf','svm_linear','svm_poly','svm_sigmoid']
accuracy_list = [0.972, 0.889, 0.917, 0.639, 0.833, 0.833, 0.917, 0.972, 0.917, 0.889]

fig = go.Figure()

trace = go.Bar(
    x = algothm_list,
    y = accuracy_list,
)
fig.add_trace(trace)

fig.update_layout(
    title=dict(
        x=0.5,
        xanchor='center',
        xref='paper',
        text='Accuracy Comparison'
    )
)
plotly.offline.plot(fig, filename='Accuracy Comparison.html')