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

def knn_predict(X_train, Y_train, X_test, Y_test, k_num):
    # print(X_train.shape)
    # print(Y_train.shape)
    # print(X_test.shape)
    # print(Y_test.shape)
    X_train = np.c_[X_train, Y_train]
    X_test = np.c_[X_test, Y_test]
    y_pred = []
    for i in X_test:
        dist_list = []
        vote_list = []
        for j in X_train:
            dist = np.sqrt((i[0]-j[0])*(i[0]-j[0]) + (i[1]-j[1])*(i[1]-j[1]) + (i[2]-j[2])*(i[2]-j[2]))
            dict = {'dist':dist, 'label':j[3]}
            dist_list.append(dict)
        dist_list = sorted(dist_list, key= operator.itemgetter('dist'))
        vote_list = [dist_list[i]['label'] for i in range(0, k_num)]
        most = max(vote_list, key=vote_list.count)
        y_pred.append(most)
    # print(y_pred)
    # print(Y_test)
    # print(list(Y_test == y_pred))
    Y_test = np.array(Y_test)
    y_pred = np.array(y_pred)
    accuracy = list(Y_test == y_pred).count(1)/len(y_pred)
    # print("accuracy = ", end='')
    # print(accuracy)
    return Y_test, y_pred, accuracy


def knn_zip(X_train, Y_train, X_test, Y_test):  #剪辑knn
    X_train = np.c_[X_train, Y_train]
    X_test = np.c_[X_test, Y_test]
    store = []
    garbbag = []
    store.append(list(X_train[0]))
    store.append(list(X_train[1]))
    store.append(list(X_train[2]))
    for i in X_train[3:len(X_train)]:
        garbbag.append(list(i))
    # print(garbbag)
    while(garbbag !=[]):
        for i in garbbag:
            # print('i==')
            # print(i)
            if(garbbag!=[]):
                # print(garbbag)
                # print((np.array(store))[:,0:3])
                # print(np.array(i[0:3]))
                # print([np.array(i)[3]])
                Y_test, y_pred,accuracy = knn_predict((np.array(store))[:,0:3], (np.array(store))[:,3], np.array(i[0:3]).reshape(1,3), np.array([np.array(i)[3]]), 3)
                # print(Y_test)
                # print(y_pred)
                # Y_test = np.array(Y_test)
                # y_pred = np.array(y_pred)
                garbbag.remove(i)
                # print("remove!")
                # print(garbbag)
                if Y_test != y_pred:
                    # print("加入store！")
                    # print(Y_test == y_pred)
                    store.append(list(i))
            else:
                break

    # print(store)
    return store

def plot(store):
    X_train = np.c_[X_train_3d_kpca, Y_train]
    x1 = [float(i[0]) for i in store if i[3] == 1]
    y1 = [float(i[1]) for i in store if i[3] == 1]
    z1 = [float(i[2]) for i in store if i[3] == 1]
    x2 = [float(i[0]) for i in store if i[3] == 2]
    y2 = [float(i[1]) for i in store if i[3] == 2]
    z2 = [float(i[2]) for i in store if i[3] == 2]
    x3 = [float(i[0]) for i in store if i[3] == 3]
    y3 = [float(i[1]) for i in store if i[3] == 3]
    z3 = [float(i[2]) for i in store if i[3] == 3]
    x4 = [float(i[0]) for i in X_train if i[3] == 1]
    y4 = [float(i[1]) for i in X_train if i[3] == 1]
    z4 = [float(i[2]) for i in X_train if i[3] == 3]
    x5 = [float(i[0]) for i in X_train if i[3] == 2]
    y5 = [float(i[1]) for i in X_train if i[3] == 2]
    z5 = [float(i[2]) for i in X_train if i[3] == 3]
    x6 = [float(i[0]) for i in X_train if i[3] == 3]
    y6 = [float(i[1]) for i in X_train if i[3] == 3]
    z6 = [float(i[2]) for i in X_train if i[3] == 3]
    # print(X_train)
    # print(x4)
    fig = go.Figure()
    trace1 = go.Scatter3d(
        x = x4,
        y = y4,
        z = z4,
        marker=dict(
            size=10,
            color='red',
        ),
        mode='markers',
        name = '1类'
    )
    trace2 = go.Scatter3d(
        x=x5,
        y=y5,
        z=z5,
        marker=dict(
            size=10,
            color='green',
        ),
        mode='markers',
        name='2类'
    )
    trace3 = go.Scatter3d(
        x=x6,
        y=y6,
        z=z6,
        marker=dict(
            size=10,
            color='blue',
        ),
        mode='markers',
        name='3类'
    )
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.add_trace(trace3)
    fig.update_layout(
        title = dict(
            x = 0.5,
            xanchor = 'center',
            xref = 'paper',
            text='Knn Scatter'
        )
    )
    plotly.offline.plot(fig,filename='Knn_Scatter.html')


def knn_best_k(max_num):
    accuracy_list = []
    for i in range(1,max_num):
        Y_test, y_pred, accuracy = knn_predict(X_train_3d_kpca, Y_train, X_test_3d_kpca, Y_validation,i)
        accuracy_list.append(accuracy)
    print(accuracy_list)

    fig =go.Figure()
    trace = go.Scatter(
        x = list(range(0,max_num)),
        y = accuracy_list,
        mode = 'lines'
    )
    fig.add_trace(trace)
    fig.update_layout(
        title=dict(
            x=0.5,
            xanchor='center',
            xref='paper',
            text='Best K Value'
        )
    )
    plotly.offline.plot(fig, filename='Best_K_Value.html')


if __name__ == "__main__":
    Y_test, y_pred, accuracy = knn_predict(X_train_3d_kpca, Y_train, X_test_3d_kpca, Y_validation, 3)
    print(accuracy)
    # 绘制混淆矩阵.create_confusion_matrix_graph(['1','2','3'], list(Y_test), list(y_pred), "confusion_matrix_knn_kpca.png")
    # 绘制ROC图.roc(list(Y_test), list(y_pred),3,  filename='knn_kpca_roc.png', title='knn_kpca_roc')
    store = knn_zip(X_train_3d_kpca, Y_train, X_test_3d_kpca, Y_validation)
    y_store = [i[3] for i in store]
    store = [i[0:3] for i in store]
    store = np.array(store)
    print(store)
    Y_test1, y_pred1, accuracy1 = knn_predict(store, y_store, X_test_3d_kpca, Y_validation, 3)
    print(accuracy1)
    # plot(store)
    # knn_best_k(50)

