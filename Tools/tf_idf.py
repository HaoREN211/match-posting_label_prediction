# -*- coding: UTF-8 -*-
# 作者：hao.ren3
# 时间：2020/8/20 11:20
# IDE：PyCharm
# https://blog.csdn.net/laobai1015/article/details/80451371

from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeClassifier
import pandas as pd
import logging
import datetime as dt

# ngram_range 同词袋模型
# 将 text 分成 n1,n1+1,……,n2个不同的词组。
# 比如比如'Python is useful'中ngram_range(1,3)之后可得到 'Python' ， 'is' ， 'useful' ， 'Python is' ， 'is useful' ， 'Python is useful'。
# 如果是ngram_range (1,1) 则只能得到单个单词'Python' ， 'is' ， 'useful'。

# norm: l1, l2 或 None。默认l2
# 归一化，我们计算TF-IDF的时候，是用TF*IDF，TF可以是归一化的，也可以是没有归一化的，一般都是采用归一化的方法，默认开启.
# L1正则化是指权值向量w中各个元素的绝对值之和，L1正则化可以产生稀疏权值矩阵，即产生一个稀疏模型，可以用于特征选择
# L2正则化是指权值向量w中各个元素的平方和然后再求平方根，L2正则化可以防止模型过拟合（overfitting）；一定程度上，L1也可以防止过拟合

# max_df: 0.0-1.0 默认1.0
# 有些词，他们的文档频率太高了（一个词如果每篇文档都出现，那还有必要用它来区分文本类别吗？当然不用了呀），
# 所以，我们可以设定一个阈值，比如float类型0.5（取值范围[0.0,1.0]）,
# 表示这个词如果在整个数据集中超过50%的文本都出现了，那么我们也把它列为临时停用词。
# 当然你也可以设定为int型，例如max_df=10,表示这个词如果在整个数据集中超过10的文本都出现了，那么我们也把它列为临时停用词。

# min_df：0.0-1.0 默认1
# 与max_df相反，虽然文档频率越低，似乎越能区分文本，可是如果太低，例如10000篇文本中只有1篇文本出现过这个词，
# 仅仅因为这1篇文本，就增加了词向量空间的维度，太不划算。

# smooth_idf：True 或 False 默认True
# 计算idf的时候log(分子/分母)分母有可能是0，smooth_idf会采用log(分子/(1+分母))的方式解决。默认已经开启，无需关心。

# sublinear_tf：True 或 False 默认False
# 计算tf值采用亚线性策略。比如，我们以前算tf是词频，现在用1+log(tf)来充当词频。

# stop_words 默认None
# 传入停用词，以后我们获得vocabulary_的时候，就会根据文本信息去掉停用词得到

# vocabulary 默认None
# 词典索引，例如    vocabulary={"我":0,"喜欢":1,"相国大人":2}

# analyzer : string, {‘word’, ‘char’} or callable 默认word
# 当analyzer='word'时，按照单词进行切词
# 当analyzer='char'时，按照char进行切分
# 该参数通常结合ngram_range来一起使用

# max_features  int or None, default=None
# 最大feature数量，即最多取多少个关键词，假设max_features=10, 就会根据整个corpus中的tf值，取top10的关键词

# tf_idf的网格搜索
def find_best_tf_idf_vectorizer(data, column, min_df=None, max_df=None, ngram_min=None, ngram_max=None, norm=None,
                                max_features=None, label="label"):
    # 如果没有给参数设置用于网格搜索的值列表，则使用tf_idf模型参数的默认值。
    min_df = [1] if min_df is None else min_df
    max_df = [1.0] if max_df is None else max_df
    ngram_min = [1] if ngram_min is None else ngram_min
    ngram_max = [1] if ngram_max is None else ngram_max
    norm = ["l2"] if norm is None else norm
    max_features = [None] if max_features is None else max_features

    # 循环参数列表，记录各参数组合时，对应的tf_idf模型的f1_score
    result = pd.DataFrame(columns=["min_df", "max_df", "ngram_min", "ngram_max", "norm", "max_features", "f1_score"])
    for row in itertools.product(ngram_min, ngram_max, min_df, max_df, norm, max_features):
        current_ngram_min = row[0]
        current_ngram_max = row[1]
        current_min_df = row[2]
        current_max_df = row[3]
        current_norm = row[4]
        current_max_features = row[5]

        # 当N-Gram的下限大于上限的时候属于异常情况，跳过此种情况。
        if current_ngram_min > current_ngram_max:
            continue

        print("----------------------------")
        print("ngram_range：("+str(current_ngram_min)+", "+str(current_ngram_max)+")")
        print("min_df："+str(current_min_df))
        print("max_df：" + str(current_max_df))
        print("norm："+str(current_norm))
        print("max_features：" + str(current_max_features))

        # 计算当前模型的f1分数
        current_f1_score, _ = verify_tf_idf_model_by_f1(data, column, current_ngram_min, current_ngram_max,
                                                        current_min_df, current_max_df, current_norm,
                                                        current_max_features, label=label)
        print("f1_score："+str(current_f1_score))
        result = result.append(pd.DataFrame({
            "min_df": [current_min_df],
            "max_df": [current_max_df],
            "ngram_min": [current_ngram_min],
            "ngram_max": [current_ngram_max],
            "norm": [current_norm],
            "max_features": [current_max_features],
            "f1_score": [current_f1_score]
        }))
    result.sort_values(by=["f1_score"], ascending=False, inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


# 通过F1-SCORE验证当前tf_idf模型的准确性
def verify_tf_idf_model_by_f1(data_train, column, ngram_min, ngram_max, min_df, max_df, norm, max_features, label="label"):
    # 重置索引，防止在KFold的时候造成因索引引起的问题
    data = data_train.reset_index(drop=True).copy()

    # 进行tf_idf的步骤
    tf_idf = TfidfVectorizer(ngram_range=(ngram_min, ngram_max), max_features=max_features,
                             min_df=min_df, max_df=max_df, norm=norm,
                             smooth_idf=True, sublinear_tf=False, stop_words=None, vocabulary=None, use_idf=True,
                             tokenizer=None, strip_accents=None, analyzer="word")
    transform_data = tf_idf.fit_transform(data[column])

    # K折交叉验证，根据F1-SCORE验证当前模型的准确性。
    k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
    result = pd.DataFrame(columns=["label", "predict"])
    for train_index, test_index in k_fold.split(data):
        # 划分训练集和验证集。
        c_train_data, c_test_data = transform_data[train_index], transform_data[test_index]
        c_train_label, c_test_label = data[label].values[train_index], data[label].values[test_index]

        # 使用桥回归分类模型来验证分类的结果
        clf = RidgeClassifier()
        clf.fit(c_train_data, c_train_label)
        result=result.append(pd.DataFrame({
            "label": c_test_label,
            "predict": clf.predict(c_test_data)
        }))
    return f1_score(list(result["label"].values), list(result["predict"].values), average='macro'), result.reset_index(drop=True)


# 验证模型的好坏程度
def verify_model_accuracy(inside_data, classifier, labels, inside_n_splits):
    i_validation = 1
    i_folder = KFold(n_splits=inside_n_splits, shuffle=True, random_state=0)
    i_df_validation = pd.DataFrame(columns=["label", "predict"])
    i_current_time = dt.datetime.now()
    logging.info("正在验证模型的准确程度")
    for i_train_index, i_val_index in i_folder.split(inside_data):
        logging.info("--- 当前完成" + str(round(i_validation / inside_n_splits, 2)) + "，耗时" + str(
            (dt.datetime.now() - i_current_time).seconds) + "秒。")
        i_train_data, i_val_data = inside_data[i_train_index], inside_data[i_val_index]
        classifier.fit(i_train_data, labels[i_train_index])
        i_df_validation = i_df_validation.append(pd.DataFrame({
            "label": labels[i_val_index],
            "predict": classifier.predict(i_val_data)
        }))
        i_validation += 1
    i_f1_score = f1_score(list(i_df_validation["label"].values), list(i_df_validation["predict"].values), average="macro")
    return i_f1_score


# 计算当前模型分词的好坏程度
def calculate_score_and_predict(inside_data, model, labels, text_column="text", nb_train_sample=200000,
                    inside_n_splits=10, classifier=RidgeClassifier()):
    # 设置日志打印的格式
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

    # 将数据进行tf-idf的转化
    logging.info("正在进行tf-idf分词")
    inside_transformed_data = model.fit_transform(inside_data[text_column])
    inside_vector_train = inside_transformed_data[:nb_train_sample]
    inside_vector_test = inside_transformed_data[nb_train_sample:]

    # 验证模型的好坏程度
    i_f1_score = verify_model_accuracy(inside_vector_train, classifier, labels, inside_n_splits)
    logging.info("模型准确率：" + str(i_f1_score))

    # 预测
    logging.info("开始预测测试集的类别")
    classifier.fit(inside_vector_train, labels)
    i_test_label_predict = classifier.predict(inside_vector_test)

    logging.basicConfig(level=logging.WARNING)
    return i_f1_score, i_test_label_predict
