# -*- coding: UTF-8 -*- 
# 作者：hao.ren3
# 时间：2020/8/20 10:29
# IDE：PyCharm

from Tools.data import get_train_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
import numpy as np

if __name__ == '__main__':
    # 获取训练集数据
    data_train = get_train_data()

    # 统计各个词在各个帖子中出现的次数
    vectorizer = CountVectorizer(max_features=10000)
    vector_train = vectorizer.fit_transform(data_train["text"])
    print(np.shape(vector_train))

    # 使用逻辑回归模型
    clf = RidgeClassifier()
    clf.fit(vector_train[:10000], data_train['label'].values[:10000])

    # 使用训练好的逻辑回归进行预测
    val_pred = clf.predict(vector_train[10000:])

    # 0.7016762683786281
    print(f1_score(data_train['label'].values[10000:], val_pred, average='macro'))
