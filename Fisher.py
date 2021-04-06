import os
import sys
from 数据预处理 import X_train_3d_pca, X_train_2d_pca, X_train_3d_kpca, X_train_2d_kpca, Y_train, Y_validation, X_test_3d_kpca, X_train, X_validation, X_test_2d_kpca
import 绘制ROC图
import 绘制混淆矩阵
import numpy as np
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt


def class_mean(samples):  # 求样本均值
    aver = np.mean(samples, axis=0) # 对各行求均值 返回m*1的矩阵
    return aver


def withclass_scatter(samples, mean):  # 求类内散度
    num, dim = samples.shape  # dim 维度  num 样本数
    samples_m = samples - mean
    s_with = 0
    for i in range(num):
        x = samples_m[i, :]
        s_with += np.dot(x, x.T)
    return s_with


def get_w(s_in1, s_in2, mean1, mean2):  # 得到权向量
    sw = s_in1 + s_in2
    w = np.dot(sw.I, (mean1 - mean2))
    return w


def classify(test, w, mean1, mean2):  # 分类算法
    # test = np.array(test)
    cen_1 = np.dot(w.T, mean1)
    cen_2 = np.dot(w.T, mean2)
    g = np.dot(w.T, test)
    return abs(g - cen_1) < abs(g - cen_2)


def predict(test, class_num, class_name):
    y_pred = []
    result = {}
    for k in np.array(test):
        for i in (range(class_num)):
            result[i] = 0
        # k = np.matrix(k)
        k = np.mat(k)
        # print('k')
        # print(k)
        for i in (range(class_num)):  # 第一次循环计算权向量的值
            for j in (range(class_num)):
                if i == j:
                    break
                mean1 = class_mean(class_name[i])
                mean2 = class_mean(class_name[j])
                s_in1 = withclass_scatter(class_name[i], mean1)
                s_in2 = withclass_scatter(class_name[j], mean2)
                w = get_w(s_in1, s_in2, mean1, mean2)
                w = np.array(w)
                # print(w)
                if any(classify(k, w, mean1, mean2)):
                    result[i] += 1
                else:
                    result[j] += 1
        # print(result)
        y_pred.append(max(result, key=result.get) + 1)
    y_pred = np.array(y_pred)
    accuracy = list(Y_validation == y_pred).count(1) / len(y_pred)
    return y_pred, accuracy


if __name__ == '__main__':
    # print(list(range(3)))
    X_train_3d_kpca = np.c_[X_train_3d_kpca, Y_train]
    X_train_0 = [i[0:3] for i in X_train_3d_kpca if i[3]==1]
    X_train_1 = [i[0:3] for i in X_train_3d_kpca if i[3] == 2]
    X_train_2 = [i[0:3] for i in X_train_3d_kpca if i[3] == 3]
    X_train_0 = np.matrix(X_train_0)
    X_train_1 = np.matrix(X_train_1)
    X_train_2 = np.matrix(X_train_2)
    # print(X_train_0)
    # print(X_train_1)
    # print(X_train_2)
    class_num = 3
    class_name = {0:X_train_0, 1:X_train_1, 2:X_train_2}
    test = X_test_3d_kpca
    # print('test')
    # print(test)
    y_pred, accuracy = predict(X_test_3d_kpca, class_num, class_name)
    print('accuracy:')
    print(accuracy)

    # 绘制ROC图.roc(list(Y_validation), list(y_pred), 3, filename='Fisher_Kpca_Roc.png', title='Fisher_Kpca_Roc')
    # 绘制混淆矩阵.create_confusion_matrix_graph(['1','2','3'], list(Y_validation), list(y_pred), "confusion_matrix_fisher_kpca.png")
