# -*- coding: UTF-8 -*- 
# 作者：hao.ren3
# 时间：2020/8/20 10:45
# IDE：PyCharm

from Tools.data import get_train_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from Tools.tf_idf import find_best_tf_idf_vectorizer

if __name__ == '__main__':
    # 获取训练集数据
    data_train = get_train_data()

    # 网格搜索
    result = find_best_tf_idf_vectorizer(data_train, "text", min_df = [1, 10, 20], max_df=[1.0, 0.9, 0.8], ngram_min=[1],
        ngram_max=[1,2,3], norm=["l1", "l2", None], max_features=[None, 1000, 2000, 3000])

    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=3000)
    train_test = tfidf.fit_transform(data_train['text'])

    clf = RidgeClassifier()
    clf.fit(train_test[:10000], data_train['label'].values[:10000])

    val_pred = clf.predict(train_test[10000:])
    # 0.8745508835598
    print(f1_score(data_train['label'].values[10000:], val_pred, average='macro'))
