from sklearn import svm
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


def train(x,y, kernel_name):
    clf = svm.SVC(C=0.8, kernel=kernel_name, gamma=20, decision_function_shape='ovo')
    clf.fit(x, y)
    return clf


def predict(clf, x, y, x_test, y_test):
    # print(clf.score(x, y))  # 精度
    y_pred1 = clf.predict(x)
    # print(y_pred1)
    accuracy1 = calc_accuracy(y_pred1, y)
    y_pred2 = clf.predict(x_test)
    accuracy2 = calc_accuracy(y_pred2, y_test)
    print('训练集精度：', end='')
    print(accuracy1)
    print('测试集准确率：', end='')
    print(accuracy2)
    return y_pred1, y_pred2, accuracy1, accuracy2


def calc_accuracy(y_test, y):
    return list(y == y_test).count(1)/len(y)


def different_SVM_kernel():
    kernel_list = ['rbf', 'linear', 'poly', 'sigmoid']
    clf_list = []
    pred_list = []
    for i in kernel_list:
        clf = train(X_train_3d_kpca, Y_train, kernel_name=i)
        clf_list.append(clf)
    for i in clf_list:
        y_pred1, y_pred2, acc1, acc2 = predict(i, X_train_3d_kpca, Y_train, X_test_3d_kpca, Y_validation)
        pred_list.append(acc2)
    print(pred_list)

    fig = go.Figure()
    trace = go.Bar(
        x=kernel_list,
        y=pred_list,
    )
    fig.add_trace(trace)
    fig.update_layout(
        title=dict(
            x=0.5,
            xanchor='center',
            xref='paper',
            text='Different SVM Kernel'
        )
    )
    plotly.offline.plot(fig, filename='Different SVM Kernel.html')


if __name__ == '__main__':
    print('svm_main')
    clf = train(X_train_3d_kpca, Y_train, kernel_name='rbf')
    y_pred1, y_pred2 = predict(clf, X_train_3d_kpca, Y_train, X_test_3d_kpca, Y_validation)
    # different_SVM_kernel()
    绘制ROC图.roc(list(Y_validation), list(y_pred1), 3, filename='SVM_Kpca_Roc.png', title='SVM_Kpca_Roc')