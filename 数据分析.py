import plotly
import chart_studio
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations
import numpy as np
import pandas as pd
import io
# chart_studio.tools.set_credentials_file(username='KevinSSS', api_key='6Nj84E5uu5s4CRqHXrT2')
# chart_studio.tools.set_config_file(world_readable=True, sharing='public')

columns=['0Alcohol', '1Malic acid ', '2Ash', '3Alcalinity of ash',
         '4Magnesium', '5Total phenols', '6Flavanoid',
         '7Nonflavanoid phenols', '8Proanthocyanins ', '9Color intensity ', '10Hue ', '11OD280/OD315 of diluted wines' , '12Proline ', '13category']

data = pd.read_csv("D:/专业课/模式识别/期末/项目代码/数据集/wine_data.csv", header=None, names=columns)


# ---------------特征类型表格----------
def datainfo(data):
    info = data.isnull().any()
    info = np.array(info.tolist()).reshape(-1,1)
    info1 = np.array(columns).reshape(-1,1)
    info2 = np.array((data.dtypes).tolist()).reshape(-1,1)
    info = np.hstack((info1, info, info2))
    info = np.row_stack((['cols','is_null','dtype'], info))
    fig = ff.create_table(info)
    plotly.offline.plot(fig, filename = 'datainfo.html')


#  --------------数据盒图--------------
def Feature_Box(data):
    data = [go.Box(
        x = data['13category'],
        y = data[i],
        name = i
    ) for i in columns]
    fig = make_subplots(rows=3, cols=5, subplot_titles=columns[0:13])
    for i in range(0, len(data)-1):
        row = int(i/5)+1
        col = int(i%5)+1
        fig.append_trace(data[i], row, col)
    plotly.offline.plot(fig, filename='Feature_Box.html')


# --------------样本特征直方图----------
def Feature_Histogram(data):
    data = [go.Histogram(
        x = data[i],
        histnorm='probability',
        opacity=0.75,
        name=i
    )for i in columns]
    fig = make_subplots(rows=3, cols=5, subplot_titles=columns[0:13])
    for i in range(0, len(data)-1):
        row = int(i/5)+1
        col = int(i%5)+1
        fig.append_trace(data[i], row, col)
    plotly.offline.plot(fig, filename='Feature_Histogram.html')


# -------计算各特征与类别的皮尔森相关系数---------
def pearsonar(data,y):
    pearson=[]
    for col in data.columns.values:
        pearson.append(abs(pearsonr(data[col].values,y)[0]))
    pearsonr_X = pd.DataFrame({'col':data.columns,'corr_value':pearson})
    pearsonr_X = pearsonr_X.sort_values(by='corr_value',ascending=False)
    fig = ff.create_table(pearsonr_X)
    plotly.offline.plot(fig, filename='Feature_Pearson.html')
# 2Ash的皮尔森系数最小，与类别的线性相关度最低


# ----------计算特征之间的皮尔森相关系数-----------
def pearson_eachother(data):
    c=list(combinations([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12],2))
    p=[]
    for i in range(len(c)):
        p.append(abs(pearsonr(data.iloc[:,c[i][0]],data.iloc[:,c[i][1]])[0]))
    pearsonr_ = pd.DataFrame({'col':c, 'corr_value':p})
    pearsonr_ = pearsonr_.sort_values(by='corr_value',ascending=False)
    fig = ff.create_table(pearsonr_)
    plotly.offline.plot(fig, filename='Feature_Pearson_eachother.html')
# 5 6 11 之间的数据相关性比较大，可能存在一定的数据冗余


# ---------通过随机森林特征重要性筛选特征-----------
def randomForest_importfeat(data, y):
    features_list=data.columns
    forest = RandomForestClassifier(oob_score=True, n_estimators=10000)
    forest.fit(data, y)
    feature_importance = forest.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    important_features = features_list[0:13]
    sorted_idx = np.argsort(feature_importance[0:13])[::-1]
    feature_importance = list(feature_importance)
    important_features = list(important_features)
    feature_importance.sort(reverse=True)
    data = [go.Bar(
        x=[important_features[i] for i in sorted_idx],
        y=feature_importance,
    )]
    plotly.offline.plot(data,filename='randomForest_importfeat.html')
# 2 7 特征对分类选择的影响程度相对较小

'''
    综上，可以看出特征2：Ash 对数据的分类影响很小，所以我们这里可以把特征2删去。
    同时，由于很多特征之间线性相关的程度比较高，所以需要对特征做降维处理，使得新的特征之间线性无关，减少计算量。
'''

if __name__ == '__main__':
    print('数据分析_main')
    # datainfo(data)
    # Feature_Box(data)
    # Feature_Histogram(data)
    # pearsonar(data.drop(['13category'], axis=1), data['13category'])
    # pearson_eachother(data)
    randomForest_importfeat(data.drop(['13category'], axis=1), data['13category'])