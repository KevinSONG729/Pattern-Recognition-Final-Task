from sklearn import tree
from 数据预处理 import X_train, X_validation, Y_train, Y_validation, X_train_3d_kpca, X_train_3d_pca, X_test_3d_kpca
from 数据分析 import columns
import matplotlib.pyplot as plt
import graphviz
print(X_train.shape)
def decision_tree(X_train, Y_train, X_validation, Y_validation):
    clf = tree.DecisionTreeClassifier(criterion="entropy")  # 载入决策树分类模型
    clf = clf.fit(X_train, Y_train)  # 决策树拟合，得到模型
    score = clf.score(X_validation, Y_validation)  # 返回预测的准确度
    print(clf)
    print(score)
    plot(clf)


def decision_tree_random_state_splitter(X_train, Y_train, X_validation, Y_validation):
    clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=30, splitter="random")
    clf = clf.fit(X_train, Y_train)
    score = clf.score(X_validation, Y_validation)
    print(score)

    test = []
    for i in range(10):
        clf = tree.DecisionTreeClassifier(max_depth=i + 1, criterion="entropy", random_state=30, splitter="random")
        clf = clf.fit(X_train, Y_train)
        score = clf.score(X_validation, Y_validation)
        test.append(score)
    plt.plot(range(1, 11), test, color="red", label="max_depth")
    plt.legend()
    plt.savefig
    plt.show()


def plot(clf):
    # feature_name = columns[0:3]
    feature_name = ['feature_1','feature_2','feature_3']
    dot_data = tree.export_graphviz(clf, feature_names=feature_name, class_names=["1", "2", "3"], filled=True,rounded=True)
    graph = graphviz.Source(dot_data)  # 画树
    print(graph)
    graph.view(filename="tree",directory="D:\专业课\模式识别\期末\项目代码")


if __name__ == '__main__':
    print('decision_tree_sklearn.main')
    decision_tree(X_train_3d_kpca, Y_train, X_test_3d_kpca, Y_validation)
    # decision_tree_random_state_splitter(X_train, Y_train, X_validation, Y_validation)