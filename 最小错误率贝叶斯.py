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
import math
import numpy as np
import pandas as pd
from math import log
import operator
import re
from collections import Counter


def get_priority(Y_train):
    Y_train = list(Y_train)
    # print(Y_train)
    label_priority = []
    y_set = set(Y_train)
    for item in y_set:
        a = Y_train.count(item)/len(Y_train)
        label_priority.append(a)
    # print(label_priority)
    return  label_priority


def calc_norm(X_vector, miu_vector, cov, d):
    if d == 1:
        z = 1/(cov*np.sqrt(2*math.pi))*np.exp(-(X_vector - miu_vector)*(X_vector - miu_vector)/2*cov*cov)
    elif d == 2:
        z = 1 / 2 * math.pi * np.sqrt(np.linalg.det(cov)) * np.exp(
            -0.5 * np.transpose(X_vector - miu_vector) * np.linalg.inv(cov) * (
                    X_vector - miu_vector))
    else:
        z = 1 / math.pow(2 * math.pi, 0.5 * d) * np.sqrt(np.linalg.det(cov)) * np.exp(
            -0.5 * np.dot(np.dot(np.transpose(X_vector - miu_vector), np.linalg.inv(cov)), (
                    X_vector - miu_vector)))
    return z


def plot_feature_pdf(x, y, d, x_test):
    # print(x)
    # print(list(range(0, 3)))
    all_list = []
    for i in list(range(0, 3)):
        y_list = []
        feature_list = list(x[:,i])
        mean = np.mean(feature_list)
        std = np.std(feature_list)
        for j in list(x_test[:, i]):
            y = calc_norm(j, mean, std, 1)
            y_list.append(y)
        all_list.append(y_list)
    # print(all_list[0])
    data = []
    # print(len(all_list))
    fig = go.Figure()
    for i in list(range(0, len(all_list))):
        trace = go.Scatter(
            x = x[:, i],
            y = all_list[i],
            mode = 'markers',
        )
        fig.add_trace(trace)
    fig.show()


def get_cov_and_miu(x):
    mean = np.mean(x, axis = 0)
    cov = np.cov(x.T)
    # print(mean)
    # print(cov)
    return mean, cov


def classify_one_and_other(x, y, x_test, one):
    x = np.c_[x,y]
    x_1 = [list(i[0:3]) for i in x if i[3]==one]
    x_1 = np.mat(x_1)
    x_23 = [list(i[0:3]) for i in x if i[3]!=one]
    x_23 = np.mat(x_23)
    mean1, cov1 = get_cov_and_miu(x_1)
    mean23, cov23 = get_cov_and_miu(x_23)
    # print(np.array(mean1))
    # print(np.array(mean1).reshape(3,1))
    y_pred = []
    for i in x_test:
        y_1 = calc_norm(np.array(i).reshape(3,1), np.array(mean1).reshape(3,1), cov1, 3)
        y_23 = calc_norm(np.array(i).reshape(3,1), np.array(mean23).reshape(3,1), cov23, 3)
        pri_1 = len(x_1)/(len(x_1)+len(x_23))
        y_1 = float(y_1)*pri_1
        y_23 = float(y_23)*(1-pri_1)
        if(y_1>=y_23):
            y_pred.append(one)
        else:
            y_pred.append(0)
    # print(y_pred)
    return  y_pred


def predict(x_test, class_num, y_test):
    y_pred = np.zeros(shape=(1,len(y_test)))
    accuracy = 0
    for i in list(range(1,class_num+1)):
        class_list = classify_one_and_other(X_train_3d_kpca,Y_train, x_test, i)
        y_pred = y_pred + np.array(class_list)
        accuracy = accuracy + list(np.array(class_list) == np.array(y_test)).count(1)
    y_pred = y_pred.tolist()[0]
    for i in list(range(0,len(y_pred))):
        if y_pred[i]==0:
            y_pred[i]=1
    accuracy = accuracy / len(y_test)
    return accuracy, y_pred



if __name__ == '__main__':
    print('bayes_main')
    label_priority = get_priority(Y_train)
    accuracy, y_pred = predict(X_test_3d_kpca, 3, Y_validation)
    print('accuracy = ', end='')
    print(accuracy)
    print('y_pred = ', end='')
    print(y_pred)
    # 绘制ROC图.roc(list(Y_validation), list(y_pred), 3, filename='Bayes_Kpca_Roc.png', title='Bayes_Kpca_Roc')
    # 绘制混淆矩阵.create_confusion_matrix_graph(['1', '2', '3'], list(Y_validation), list(y_pred),"confusion_matrix_bayes_kpca.png")