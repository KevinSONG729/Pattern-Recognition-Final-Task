from 数据分析 import data, columns
import 绘制混淆矩阵
import  绘制ROC图
from 数据预处理 import X_train_3d_pca, X_train_2d_pca, X_train_3d_kpca, X_train_2d_kpca, Y_train, Y_validation, X_test_3d_kpca, X_train, X_validation
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
# 获取数据集
def get_data():
    # 特征矩阵
    # print((data.values)[:, 0:12])
    # print((data.values)[:, 13])
    # print(columns[0:12])
    # print(X_train)
    feat_matrix = np.array(X_train_2d_kpca)
    # 类别标签
    labels = np.array(Y_train)
    # 特征名
    feat_names = np.array(columns[0:12])
    return feat_matrix,labels,feat_names

# 计算经验熵
def cal_entropy(x):
    x_set = set(x)
    x_size = x.shape[0]
    entropy = 0.0
    for label in x_set:
        p = np.count_nonzero(x == label) / x_size
        entropy -= p * log(p,2)
    return entropy

# 根据选定特征计算条件信息熵
def cal_conditionalentropy(feat_values,labels):
    values_set = set(feat_values)
    values_size = feat_values.shape[0]
    c_entropy = 0.0
    for value in values_set:
        p = np.count_nonzero(feat_values == value)/values_size
        c_entropy += p * cal_entropy(labels[feat_values == value])
    return c_entropy

# 计算连续特征的最小条件熵，以得到最大的信息增益,原理是上面的博客
# 大致原理:
'''
设某连续特征有n个取值，先将它们从小到大排序
使用二分法，尝试n-1种中间值(函数中的阈值就是中间值，通过前后两个元素的平均数计算得)
比较各自的条件熵，取最小的那个，最终得到最大的信息增益
返回值有二，一是最小的条件熵，二是中间值
'''
def cal_min_conditionalentropy(feat_values,labels):
    # print("cal_min_conditionalentropy labels:")
    # print(labels)
    # print("feat_values:")
    # print(feat_values)
    values_size = feat_values.shape[0]
    labels = list(labels)
    feat_values = list(feat_values)
    zip_feat_values_labels = list(zip(feat_values,labels)) # 将连续特征取值和对应的分类标签zip起来
    # print(zip_feat_values_labels)
    # 按照特征值升序排序,分类标签跟着一起排
    zip_feat_values_labels_sorted = sorted(zip_feat_values_labels, key=operator.itemgetter(0))
    feat_values_sorted = np.array([i[0] for i in zip_feat_values_labels_sorted]) # 排序过后的特征取值
    labels_sorted = np.array([i[1] for i in zip_feat_values_labels_sorted])  # 排序过后的分类标签
    thresholds = set([(feat_values_sorted[idx]+feat_values_sorted[idx+1])/2  # n个特征取值有n-1个缝隙,得n-1个阈值
                  for idx in range(feat_values_sorted.shape[0]-1)])
    min_c_entropy = float('inf')
    # print("feat values sorted")
    # print(feat_values_sorted)
    # print("labels_sorted")
    # print(labels_sorted)
    # print("thresholds")
    # print(thresholds)
    min_c_entropy_threshold = (feat_values_sorted[0] + feat_values_sorted[1])/2 # 初始化阈值是第一个缝隙中的
    for threshold in thresholds:
        filter_left = feat_values_sorted <= threshold  # 阈值左边的部分
        feat_values_left = feat_values_sorted[filter_left]
        labels_left = labels_sorted[filter_left]

        filter_right = feat_values_sorted > threshold  # 阈值右边的部分
        feat_values_right = feat_values_sorted[filter_right]
        labels_right = labels_sorted[filter_right]
        c_entropy = feat_values_left.shape[0]/values_size*cal_entropy(labels_left) +\
                    feat_values_right.shape[0]/values_size*cal_entropy(labels_right)
        if c_entropy <= min_c_entropy:
            min_c_entropy = c_entropy
            min_c_entropy_threshold = threshold
    return min_c_entropy,min_c_entropy_threshold  # 返回有二,最小的条件信息熵和对应的阈值

# 根据选定特征计算信息增益
def cal_info_gain(feat_values,labels):
    # # 如果是离散值
    # if feat_values[0].item()>=1:
    #     return cal_entropy(labels) - cal_conditionalentropy(feat_values,labels),'discrete'
    # # 如果是连续的
    # else:
    #     min_c_entropy, min_c_entropy_threshold = cal_min_conditionalentropy(feat_values,labels)
    #     return cal_entropy(labels) - min_c_entropy,min_c_entropy_threshold
    '''
    在葡萄酒数据集中各特征的数据都是连续的
    '''''
    min_c_entropy, min_c_entropy_threshold = cal_min_conditionalentropy(feat_values, labels)
    return cal_entropy(labels) - min_c_entropy,min_c_entropy_threshold



# 根据选定特征计算信息增益比
def cal_info_gain_ratio(feat_values,labels):
    return (cal_info_gain(feat_values,labels) + 0.01)/(cal_entropy(feat_values)+0.01)


# 生成决策树中的第二个终止条件满足时,返回实例数最大的类
def get_max_label(labels):
    return Counter(labels)[0][0]

# 选择信息增益、信息增益比最大的特征
def get_best_feat(feat_matrix,labels):
    feat_num = feat_matrix.shape[1]
    best_feat_idx = -1
    max_info_gain = 0.0
    ret_sign = 'discrete'  # 默认是离散的
    for feat_idx in range(feat_num):
        # print(feat_idx)
        feat_values = feat_matrix[:,feat_idx]
        info_gain,sign = cal_info_gain(feat_values,labels)
        if info_gain >= max_info_gain:
            max_info_gain = info_gain
            best_feat_idx = feat_idx
            ret_sign = sign
    return best_feat_idx,ret_sign

# 根据选定特征,划分得到子集
def get_subset(feat_matrix,labels,best_feat,sign):
    feat_values = feat_matrix[:,best_feat]
    if sign == 'discrete':
        values_set = set(feat_values)
        feat_matrix = np.delete(feat_matrix,best_feat,1)
        feat_matrixset = {}
        labelsset = {}
        for value in values_set:
            feat_matrixset[value] = feat_matrix[feat_values==value]
            labelsset[value] = labels[feat_values==value]
    # 连续值
    else:
        threshold = sign
        feat_matrixset = {}
        labelsset = {}
        # 左
        filter_left = feat_values <= threshold
        feat_matrixset['<={}'.format(threshold)] = feat_matrix[filter_left]
        labelsset['<={}'.format(threshold)] = labels[filter_left]
        # 右
        filter_right = feat_values > threshold
        feat_matrixset['>{}'.format(threshold)] = feat_matrix[filter_right]
        labelsset['>{}'.format(threshold)] = labels[filter_right]
    return feat_matrixset,labelsset

"""
introduction:
    生成一棵决策树
parameter:
    feat_matrix:特征矩阵
    labels:类别标签
    feat_names:特征名称
    method:选择方法(信息增益、信息增益比)
"""
def create_decision_tree(feat_matrix,labels,feat_names,method):
    # 首先考虑两种终止条件:1、类别标签只有一种类型 2、特征没有其他取值
    # print("create trees")
    # print("labels")
    # print(labels)
    if len(set(labels)) == 1:
        return labels[0] # 类型为numpy.int32
    if feat_matrix.shape[0] == 0:
        return get_max_label(labels)
    # 选择信息增益最大的特征，sign标志着是离散特征还是连续特征，若连续，sign为阈值
    best_feat,sign = get_best_feat(feat_matrix,labels)
    # print("best_feat")
    # print(best_feat)
    best_feat_name = feat_names[best_feat]
    # print(best_feat_name)
    # print("sign")
    # print(sign)
    # 初始化树
    decision_tree = {best_feat_name:{}}
    # 如果是离散的,则要删除该特征，否则不删。思考:连续特征何时删？当
    # if sign == 'discrete':
    #     feat_names = np.delete(feat_names,best_feat)
    # feat_names = np.delete(feat_names, best_feat)
    # 得到子集(得到字典类型的子集,键为选定特征的不同取值)
    feat_matrixset, labelsset = get_subset(feat_matrix,labels,best_feat,sign)
    # 递归构造子树
    for value in feat_matrixset.keys():
        # print("value = ")
        # print(value)
        # print(labelsset[value])
        decision_tree[best_feat_name][value] = create_decision_tree(feat_matrixset[value],labelsset[value],feat_names,method)
    return decision_tree


def predict(dt, x, y, feats):
    # print("predict")
    # print(feats)
    # print(float(re.findall(r"\d+\.?\d*","<=1.4")[0]))
    predict_list = []
    label_dict = dt
    for i in x:
        i = list(i)
        label_dict = dt
        while(type(label_dict)!=np.float64):
            # print(type(label_dict))
            # print(label_dict)
            label_key = list(label_dict.keys())[0]
            label_dict = list(label_dict.values())[0]
            label_key_idx = feats.index(label_key)
            if(i[label_key_idx]<=float(re.findall(r"\d+\.?\d*",list(label_dict.keys())[0])[0])):
                label_dict = list(label_dict.values())[0]
            else:
                label_dict = list(label_dict.values())[1]
        predict_list.append(label_dict)
    # print(predict_list)
    # print(y)
    predict_list = np.array(predict_list)
    y = np.array(y)
    accuracy = list(predict_list == y).count(1)/len(y)
    return accuracy, predict_list, y


if __name__ == "__main__":
   feat_matrix,labels,feat_names = get_data()
   # decision_tree = create_decision_tree(feat_matrix,labels,feat_names,method='ID3')
   decision_tree = create_decision_tree(feat_matrix, labels, ['feature_1','feature_2','feature_3'], method='ID3')
   # accuracy, y_pred, y_true = predict(decision_tree, X_test_3d_kpca, Y_validation, list(feat_names))
   accuracy, y_pred, y_true = predict(decision_tree, X_test_3d_kpca, Y_validation, ['feature_1','feature_2','feature_3'])
   # print([int(i) for i in y_true])
   # 绘制混淆矩阵.create_confusion_matrix_graph(['1','2','3'], [int(i) for i in y_true], [int(i) for i in y_pred], filename='confusion_matrix_decision_tree_12feature.png')
   # 绘制ROC图.roc([int(i) for i in y_true], [int(i) for i in y_pred], 3, filename='decision_tree_roc_kpca.png',title='decision tree roc_kpca')
   print(decision_tree)
   print("accuracy = ", end='')
   print(accuracy)