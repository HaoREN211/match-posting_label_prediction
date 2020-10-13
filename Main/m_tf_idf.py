# -*- coding: UTF-8 -*-
# 作者：hao.ren3
# 时间：2020/8/20 10:45
# IDE：PyCharm

from Tools.data import get_train_data, get_test_data
from sklearn.feature_extraction.text import TfidfVectorizer
from Tools.tf_idf import find_best_tf_idf_vectorizer, calculate_score_and_predict
import pandas as pd

from xgboost import XGBClassifier
import xgboost as xgb


if __name__ == '__main__':
    # 获取训练集数据
    data_train = get_train_data()
    data_test = get_test_data()

    # 将训练集和测试集的数据放在一起分词
    df_all_text = pd.concat([data_train["text"], data_test["text"]])

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

    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=4000, norm="l2", min_df=10, max_df=0.7)


    f1_score, test_label_predict = calculate_score_and_predict(df_all_text, tfidf, data_train['label'].values)

    # 使用XGBClassifier分类器
    tfidf = TfidfVectorizer(ngram_range=(1, 1), max_features=10000, norm="l2", sublinear_tf=True)
    df_all_text_vectored = tfidf.fit_transform(df_all_text)
    # df_train = df_all_text_vectored[:len(data_train)]
    # df_test = df_all_text_vectored[len(data_train):]
    clf = XGBClassifier(learning_rate=0.05,
                        n_estimators=300,
                        max_depth=10,
                        min_child_weight=1,
                        gamma=0.5,
                        reg_alpha=0,
                        reg_lambda=2,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        scale_pos_weight=1,
                        objective='multi:softmax',
                        num_class=14,
                        nthread=20,
                        seed=1000)
    xgb_param = clf.get_xgb_params()
    xgb_train = xgb.DMatrix(df_all_text_vectored[:len(data_train)], label=list(data_train["label"].values))
    cvresult = xgb.cv(xgb_param, xgb_train, num_boost_round=500, nfold=5, metrics=['mlogloss'],
                      early_stopping_rounds=5, stratified=True, seed=1000)

    clf.fit(df_all_text_vectored[:len(data_train)], data_train["label"].values)
    prediction = pd.DataFrame({"label": clf.predict(df_all_text_vectored[len(data_train):])})

    prediction = pd.DataFrame({"label": test_label_predict})
    prediction.to_csv("submit_tf_idf.csv", index=False, encoding="utf_8_sig")

