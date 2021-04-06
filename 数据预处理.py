from 数据分析 import columns, data
import plotly
import chart_studio
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy.stats import pearsonr
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def dropminimportFeature(data):
    data_drop=data.drop(['2Ash'], axis=1)
    columns.pop(2)
    # print(data.info())
    return data_drop


def split_out_dataset(data):
    data = data.values
    X = data[:,0:12].astype(float)
    Y = data[:,12]
    validation_size = 0.2
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    # print(X)
    # print(Y)
    # print(X_train)
    # print(Y_train)
    # print(X_train.shape)    (142, 12)
    return X_train, X_validation, Y_train, Y_validation


# PCA特征值排序后可视化
def PCA_Values_Sort_Visible(values):
    trace1 = go.Bar(
        y = ['特征值'+str(i) for i in range(1,13)],
        x = values,
        orientation = 'h',
        marker=dict(
            color=['rgb(34,139,34)', 'rgb(34,139,34)',
                'rgb(34,139,34)'],
            line=dict(
                color='rgb(0,0,0)',
                width=1.5,
            )),
        opacity=0.8,
        name='bar'
    )
    trace2 = go.Scatter(
        y=['特征值' + str(i) for i in range(1, 13)],
        x=values,
        name='line'
    )
    fig = go.Figure()
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    # plotly.offline.plot(fig, filename='feature_values_sort.html')


# 选取前三个特征进行降维
def PCA_Vector_3d_mapping(data, vectors, sorted_idx, y):
    vector_3d = vectors[sorted_idx[0]]
    for i in sorted_idx[1:3]:
        vector_3d = np.vstack((vector_3d, vectors[i]))
    # vector_3d = vector_3d.transpose()
    data_3d = np.dot(vector_3d, data)
    data_3d = data_3d.transpose()
    # print(data_3d)
    data_1, data_2, data_3 = [], [], []
    for i in range(len(y)):
        if y[i] == 1:
            data_1.append(data_3d[i][:])
        elif y[i] == 2:
            data_2.append(data_3d[i][:])
        else:
            data_3.append(data_3d[i][:])
    # print(data_1)
    trace1 = go.Scatter3d(
        x=[np.array(i)[0][0] for i in data_1],
        y=[np.array(i)[0][1] for i in data_1],
        z=[np.array(i)[0][2] for i in data_1],
        mode='markers',
        marker=dict(
            size=5,
            color='rgba(254, 67, 101, 1)',
        ),
        name='1类'
    )
    trace2 = go.Scatter3d(
        x=[np.array(i)[0][0] for i in data_2],
        y=[np.array(i)[0][1] for i in data_2],
        z=[np.array(i)[0][2] for i in data_2],
        mode='markers',
        marker=dict(
            size=5,
            color='rgba(220, 87, 18, 1)',
        ),
        name='2类'
    )
    trace3 = go.Scatter3d(
        x=[np.array(i)[0][0] for i in data_3],
        y=[np.array(i)[0][1] for i in data_3],
        z=[np.array(i)[0][2] for i in data_3],
        mode='markers',
        marker=dict(
            size=5,
            color='rgba(69, 137, 148, 1)',
        ),
        name='3类'
    )
    fig = go.Figure()
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.add_trace(trace3)
    # plotly.offline.plot(fig, filename='PCA_3d.html')
    return data_3d


# 选取前两个特征进行降维
def PCA_Vector_2d_mapping(data, vectors, sorted_idx, y):
    vector_2d = vectors[sorted_idx[0]]
    for i in sorted_idx[1:2]:
        vector_2d = np.vstack((vector_2d, vectors[i]))
    # vector_3d = vector_3d.transpose()
    data_2d = np.dot(vector_2d, data)
    data_2d = data_2d.transpose()
    # print(data_3d)
    data_1, data_2, data_3 = [], [], []
    for i in range(len(y)):
        if y[i] == 1:
            data_1.append(data_2d[i][:])
        elif y[i] == 2:
            data_2.append(data_2d[i][:])
        else:
            data_3.append(data_2d[i][:])
    # print(data_1)
    trace1 = go.Scatter(
        x=[np.array(i)[0][0] for i in data_1],
        y=[np.array(i)[0][1] for i in data_1],
        mode='markers',
        marker=dict(
            size=10,
            color='rgba(254, 67, 101, 1)',
        ),
        name='1类'
    )
    trace2 = go.Scatter(
        x=[np.array(i)[0][0] for i in data_2],
        y=[np.array(i)[0][1] for i in data_2],
        mode='markers',
        marker=dict(
            size=10,
            color='rgba(220, 87, 18, 1)',
        ),
        name = '2类'
    )
    trace3 = go.Scatter(
        x=[np.array(i)[0][0] for i in data_3],
        y=[np.array(i)[0][1] for i in data_3],
        mode='markers',
        marker=dict(
            size=10,
            color='rgba(69, 137, 148, 1)',
        ),
        name = '3类'
    )
    fig = go.Figure()
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.add_trace(trace3)
    # plotly.offline.plot(fig, filename='PCA_2d.html')
    return data_2d


def PCA_Vector_3d_mapping_data(data, y):
    data_1, data_2, data_3 = [], [], []
    for i in range(len(y)):
        if y[i] == 1:
            data_1.append(data[i][:])
        elif y[i] == 2:
            data_2.append(data[i][:])
        else:
            data_3.append(data[i][:])
    # print(data_1[0][0])
    trace1 = go.Scatter3d(
        x=[i[0] for i in data_1],
        y=[i[1] for i in data_1],
        z=[i[2] for i in data_1],
        mode='markers',
        marker=dict(
            size=5,
            color='rgba(254, 67, 101, 1)',
        ),
        name='1类'
    )
    trace2 = go.Scatter3d(
        x=[i[0] for i in data_2],
        y=[i[1] for i in data_2],
        z=[i[2] for i in data_2],
        mode='markers',
        marker=dict(
            size=5,
            color='rgba(220, 87, 18, 1)',
        ),
        name='2类'
    )
    trace3 = go.Scatter3d(
        x=[i[0] for i in data_3],
        y=[i[1] for i in data_3],
        z=[i[2] for i in data_3],
        mode='markers',
        marker=dict(
            size=5,
            color='rgba(69, 137, 148, 1)',
        ),
        name='3类'
    )
    fig = go.Figure()
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.add_trace(trace3)
    # plotly.offline.plot(fig, filename='KPCA_3d.html')


def PCA_Vector_2d_mapping_data(data, y):
    data_1, data_2, data_3 = [], [], []
    for i in range(len(y)):
        if y[i] == 1:
            data_1.append(data[i][:])
        elif y[i] == 2:
            data_2.append(data[i][:])
        else:
            data_3.append(data[i][:])
    # print(data_1[0][0])
    trace1 = go.Scatter(
        x=[i[0] for i in data_1],
        y=[i[1] for i in data_1],
        mode='markers',
        marker=dict(
            size=10,
            color='rgba(254, 67, 101, 1)',
        ),
        name='1类'
    )
    trace2 = go.Scatter(
        x=[i[0] for i in data_2],
        y=[i[1] for i in data_2],
        mode='markers',
        marker=dict(
            size=10,
            color='rgba(220, 87, 18, 1)',
        ),
        name='2类'
    )
    trace3 = go.Scatter(
        x=[i[0] for i in data_3],
        y=[i[1] for i in data_3],
        mode='markers',
        marker=dict(
            size=10,
            color='rgba(69, 137, 148, 1)',
        ),
        name='3类'
    )
    fig = go.Figure()
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.add_trace(trace3)
    # plotly.offline.plot(fig, filename='KPCA_2d.html')


def PCA(data, y, data_test):
    data = preprocessing.scale(data) # 对数据进行标准化
    data_test = preprocessing.scale(data_test)
    data = data.transpose()
    data_test = np.array(data_test).transpose()
    # print(data)
    # print(y)
    data_conMatrix = np.mat(np.dot(data,data.transpose()))
    values, vectors = np.linalg.eig(data_conMatrix)
    sorted_idx = np.argsort(values)[::-1]
    values = list(values)
    values.sort(reverse=True)
    # PCA_Values_Sort_Visible(values)
    data_3d = PCA_Vector_3d_mapping(data, vectors, sorted_idx, y)
    data_2d = PCA_Vector_2d_mapping(data, vectors, sorted_idx, y)
    data_3d_test = PCA_Vector_3d_mapping(data_test, vectors, sorted_idx, Y_validation)
    data_2d_test = PCA_Vector_2d_mapping(data_test, vectors, sorted_idx, Y_validation)
    return data_3d, data_2d, data_3d_test, data_2d_test


def KPCA(data, y, data_test):
    data = preprocessing.scale(data)  # 对数据进行标准化
    data_test = preprocessing.scale(data_test)  # 对数据进行标准化
    # print(data.shape)
    kpca_3d = KernelPCA(kernel='rbf', n_components=3)
    kpca_2d = KernelPCA(kernel='rbf', n_components=2)
    data_kpca_3d = kpca_3d.fit_transform(data)
    data_kpca_2d = kpca_2d.fit_transform(data)
    # data_kpca_3d_test = kpca_3d.fit(data_test)
    data_kpca_3d_test = kpca_3d.transform(data_test)
    # data_kpca_2d_test = kpca_2d.fit(data_test)
    data_kpca_2d_test = kpca_2d.transform(data_test)
    # print("data_kpca_2d_test")
    # print(data_kpca_2d_test)
    # print(data_kpca_3d)
    # print(data_kpca_2d)
    PCA_Vector_3d_mapping_data(data_kpca_3d, y)
    # PCA_Vector_2d_mapping_data(data_kpca_2d, y)
    return data_kpca_3d, data_kpca_2d, data_kpca_3d_test, data_kpca_2d_test


datadrop = dropminimportFeature(data=data)
X_train, X_validation, Y_train, Y_validation = split_out_dataset(datadrop)
X_train_3d_pca, X_train_2d_pca, X_test_3d_pca, X_test_2d_pca = PCA(data=X_train, y=Y_train, data_test=X_validation)
X_train_3d_kpca, X_train_2d_kpca, X_test_3d_kpca, X_test_2d_kpca = KPCA(data=X_train, y=Y_train, data_test=X_validation)

# print(X_test_2d_pca.shape)

if __name__ == '__main__':
    print('数据预处理_main')
    # datadrop = dropminimportFeature(data=data)
    # X_train, X_validation, Y_train, Y_validation = split_out_dataset(datadrop)
    # X_train_3d_pca, X_train_2d_pca = PCA(data=X_train, y=Y_train)
    # X_train_3d_kpca, X_train_2d_kpca = KPCA(data=X_train, y=Y_train)