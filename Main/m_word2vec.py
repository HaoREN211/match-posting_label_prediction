# -*- coding: UTF-8 -*- 
# 作者：hao.ren3
# 时间：2020/8/20 16:45
# IDE：PyCharm

# https://www.jianshu.com/p/6c8588d40d59
# https://zhuanlan.zhihu.com/p/50243702

import logging
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
import pickle
import os
import datetime as dt
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from copy import copy

# 对于每一篇文章，获取文章的每一个分词在word2vec模型的相关性向量。
# 然后把一篇文章的所有分词在word2vec模型中的相关性向量求和取平均数，即此篇文章在word2vec模型中的相关性向量。
# 实例化Word2Vec对象时，关键字参数size定义为200，则相关性矩阵都为200维。
# 定义getVector函数获取每个文章的词向量，传入2个参数，第1个参数是文章分词的结果，第2个参数是word2vec模型对象。
def get_contentVector(cutWords, word2vec_model):
    vector_list = [word2vec_model.wv[k] for k in cutWords if k in word2vec_model]
    contentVector = np.array(vector_list).mean(axis=0)
    return contentVector


# 将数组转换成字符串
def convert_array_to_string(list_array):
    string_array = [str(x) for x in list_array]
    return ",".join(string_array)

if __name__ == '__main__':
    # 20200821 尝试自己搭建word2vec模型
    model_path = "Model/model_word2vec.pickle"
    vector_train_file, vector_test_file = "Data/Word2Vec/train.csv", "Data/Word2Vec/test.csv"
    train_data = pd.read_csv('Data/train_set.csv', sep='\t', encoding='UTF-8')
    test_data = pd.read_csv('Data/test_set.csv', sep='\t', encoding='UTF-8')

    # word2vec训练模型
    num_features = 20  # Word vector dimensionality
    num_workers = 8  # Number of threads to run in parallel

    # 提取text部分
    list_text = [x.split() for x in train_data["text"].values]
    list_text.extend([x.split() for x in test_data["text"].values])

    # 如果之前保存了训练好了的模型，则从本地直接读取。否则重新训练模型。
    if os.path.exists(model_path):
        # 从本地读取模型
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
    else:
        model = Word2Vec(list_text, workers=num_workers, size=num_features)
        model.init_sims(replace=True)

        # 保存模型到本地
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)

    # 保存训练集词向量
    if not (os.path.exists(vector_train_file) and os.path.exists(vector_test_file)):
        # Word2Vec向量化
        list_vector = [get_contentVector(x, model) for x in list_text]

        # 保存训练集数据
        df_train_vector = pd.DataFrame({
            "label": train_data["label"].values,
            "vector": list_vector[:len(train_data)]
        })

        # 保存测试集数据
        df_test_vector = pd.DataFrame({
            "vector": list_vector[len(train_data):]
        })

        # 将Word2Vec得到的num_features向量进行OneHot编码，向量中的一个元素成为编码中的一个特征(1列)
        logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
        current_time = dt.datetime.now()
        train_vector, test_vector = copy(df_train_vector), copy(df_test_vector)
        for i in range(num_features):
            logging.info("--- 当前第"+str(i)+"步，耗时"+str((dt.datetime.now()-current_time).seconds)+"秒。")
            train_vector["c_" + str(i+1)] = train_vector.apply(lambda x: x["vector"][i], axis=1)
            test_vector["c_" + str(i + 1)] = test_vector.apply(lambda x: x["vector"][i], axis=1)
        # 删除向量列
        df_train_vector.drop(columns=["vector"], inplace=True)
        df_test_vector.drop(columns=["vector"], inplace=True)
        df_train_vector.to_csv(vector_train_file, encoding="utf_8_sig", index=False)
        df_test_vector.to_csv(vector_test_file, encoding="utf_8_sig", index=False)

    # 从文件中读取数据Word2Vec向量化之后的数据
    df_train_vector = pd.read_csv(vector_train_file, encoding="utf_8_sig")
    df_test_vector = pd.read_csv(vector_test_file, encoding="utf_8_sig")

    # 验证模型的好坏程度
    clf = RidgeClassifier()
    n_validation, current_validation = 10, 1
    folder = KFold(n_splits=n_validation, shuffle=True, random_state=0)
    df_validation = pd.DataFrame(columns=["label", "predict"])
    current_time = dt.datetime.now()
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
    for train_index, val_index in folder.split(df_train_vector):
        logging.info("--- 当前完成" + str(round(current_validation/n_validation, 2)) + "，耗时" + str((dt.datetime.now() - current_time).seconds) + "秒。")
        clf.fit([x for x in df_train_vector.loc[train_index, "vector"]],
                list(df_train_vector["label"][train_index]))
        df_validation = df_validation.append(pd.DataFrame({
            "label": list(df_train_vector["label"][val_index]),
            "predict": clf.predict([x for x in df_train_vector.loc[val_index, "vector"]])
        }))
        current_validation += 1
    print("---模型准确率："+str(f1_score(list(df_validation["label"].values), list(df_validation["predict"].values), average="macro")))

    # save model
    model.save("./word2vec.bin")

    # load model
    model = Word2Vec.load("./word2vec.bin")

    # convert format
    model.wv.save_word2vec_format('./word2vec.txt', binary=False)

    # 调用Word2Vec模型对象的wv.most_similar方法查看与“2693”含义最相近的词。
    model.wv.most_similar('2693')

    # wv.most_similar方法使用positive和negative这2个关键字参数的简单示例。查看'2693' + '2024' - '4231'的结果，代码如下：
    model.most_similar(positive=['2693', '2024'], negative=['4231'], topn=3)

    # 查看两个词的相关性，如所示：
    model.similarity("2693", "2024")

