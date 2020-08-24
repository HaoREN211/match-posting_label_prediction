# -*- coding: UTF-8 -*- 
# 作者：hao.ren3
# 时间：2020/8/20 10:45
# IDE：PyCharm

from Tools.data import get_train_data, get_test_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from Tools.tf_idf import find_best_tf_idf_vectorizer
from sklearn.model_selection import KFold
import pandas as pd
import datetime as dt

if __name__ == '__main__':
    # 获取训练集数据
    data_train = get_train_data()
    data_test = get_test_data()

    model_path = "Model/model_tf_idf.pickle"

    # 网格搜索
    result = find_best_tf_idf_vectorizer(data_train, "text", min_df = [1, 10, 20], max_df=[1.0, 0.9, 0.8], ngram_min=[1],
        ngram_max=[1,2,3], norm=["l1", "l2", None], max_features=[None, 1000, 2000, 3000])

    # 目前找到的最大的参数
    # min_df = 1, max_df = 0.9, norm = "l2"
    # ngram_min = 1, ngram_max=1

    # 0.8958231509657203
    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=3000)

    # 0.882819275358249
    tfidf = TfidfVectorizer(ngram_range=(1, 1), max_features=3000, norm = "l2", min_df = 1, max_df = 0.9)

    # 0.8961027866279293
    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=3000, norm="l2", min_df=1, max_df=0.9)

    # 0.8963417982056024
    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=3000, norm="l2", min_df=1, max_df=0.8)

    # 0.9052568996960401
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=4000, norm="l2", min_df=1, max_df=0.8)

    df_all_text = data_train[["text"]].copy()
    df_all_text = df_all_text.append(data_test[["text"]])

    train_test = tfidf.fit_transform(df_all_text['text'])
    vector_train = train_test[:len(data_train)]
    vector_test = train_test[len(data_train):]

    # 验证模型的好坏程度
    n_validation, current_validation = 10, 1
    folder = KFold(n_splits=n_validation, shuffle=True, random_state=0)
    df_validation = pd.DataFrame(columns=["label", "predict"])
    current_time = dt.datetime.now()
    clf = RidgeClassifier()
    for train_index, val_index in folder.split(vector_train):
        print("--- 当前完成" + str(round(current_validation/n_validation, 2)) + "，耗时" + str((dt.datetime.now() - current_time).seconds) + "秒。")
        c_train_data, c_val_data = vector_train[train_index], vector_train[val_index]
        clf.fit(c_train_data, data_train['label'].values[train_index])
        df_validation = df_validation.append(pd.DataFrame({
            "label": data_train['label'].values[val_index],
            "predict": clf.predict(c_val_data)
        }))
        current_validation += 1
    print("---模型准确率："+str(f1_score(list(df_validation["label"].values), list(df_validation["predict"].values), average="macro")))

    clf.fit(vector_train, data_train['label'].values)
    prediction = pd.DataFrame({"label": clf.predict(vector_test)})
    prediction.to_csv("submit_tf_idf.csv", index=False, encoding="utf_8_sig")
