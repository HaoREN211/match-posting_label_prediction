# @Time    : 2020/8/18 22:14
# @Author  : REN Hao
# @FileName: data_research.py
# @Software: PyCharm

from Tools.data import get_test_data, get_train_data
import pandas as pd
import os
import numpy as np
import logging
import datetime as dt
from copy import copy

# 计算基尼系数
def calculate_gini(x):
    i_result = 1
    for i_i in range(14):
        i_result -= pow(x["type_"+str(i_i)+"_ratio"], 2)
    return round(i_result, 3)


# 每篇文章出现的基尼系数最低的基尼系数的值
def calculate_posting_lower_gini(posting_text, word_gini_list):
    i_list_word = [int(x) for x in posting_text.split()]
    i_list_gini = [word_gini_list.loc[x, "gini"] for x in i_list_word]
    return min(i_list_gini)

if __name__ == '__main__':
    # 获取训练集数据，测试集数据
    data_train, data_test = get_train_data(), get_test_data()

    # 将帖子转化成词汇向量
    data_train["text"] = data_train.apply(lambda x: [int(x) for x in str(x["text"]).split()], axis=1)
    data_test["text"] = data_test.apply(lambda x: [int(x) for x in str(x["text"]).split()], axis=1)

    # 所有的词汇
    list_text = data_train[["text"]].append(data_test[["text"]]).copy()
    list_word = set([])

    # 抓取帖子中出现的单词
    # 遍历所有的帖子
    for text_row in list_text.itertuples():
        # 遍历当前帖子中出现的所有的单词
        for current_word in getattr(text_row, "text"):
            list_word.add(current_word)

    # 统计各类型的帖子数
    label_cnt_file_path = "Statistics/label_posting_cnt.csv"
    if not os.path.exists(label_cnt_file_path):
        label_cnt = {}
        for label, posting_cnt in data_train["label"].value_counts().items():
            label_cnt[label] = posting_cnt

        df_label_cnt = pd.DataFrame({
            "label": list(label_cnt.keys()),
            "posting_cnt": list(label_cnt.values())
        })
        df_label_cnt.to_csv(label_cnt_file_path, index=False, encoding="utf_8_sig")
        df_label_cnt.set_index("label", inplace=True)
    else:
        df_label_cnt = pd.read_csv(label_cnt_file_path).set_index("label")

    # 统计各个单词在各种类帖子中出现的次数
    word_type_cnt = pd.DataFrame({"word": list(list_word)})
    for current_type in np.arange(0, 14).tolist():
        word_type_cnt["type_"+str(current_type)] = [0] * len(word_type_cnt)
    word_type_cnt.set_index("word", inplace=True)

    for index, row in enumerate(data_train.itertuples()):
        current_label = getattr(row, "label")
        for current_word in list(set(getattr(row, "text"))):
            word_type_cnt.loc[int(current_word), "type_"+str(current_label)] = word_type_cnt.loc[int(current_word), "type_"+str(current_label)] + 1

    df_word_type_cnt = word_type_cnt.reset_index()
    df_word_type_cnt.to_csv("word_type_cnt.csv", index=False, encoding="utf_8_sig")

    # 统计各类单词在各种类帖子中出现的比例
    # 统计各类单词在各种类帖子中出现的次数
    df_word_type_cnt = pd.read_excel("Statistics/单词在各类型帖子出现的次数.xlsx", sheet_name=0)
    # 统计各类帖子的帖子数
    df_type_cnt = pd.read_excel("Statistics/单词在各类型帖子出现的次数.xlsx", sheet_name=1).set_index("label")
    df_word_type_ratio = pd.DataFrame({"word": df_word_type_cnt["word"].values})

    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
    current_time = dt.datetime.now()
    for current_type in range(14):
        df_word_type_ratio["type_"+str(current_type)+"_type"] = df_word_type_cnt.apply(
            lambda x: round(x["type_"+str(current_type)]/df_type_cnt.loc[current_type, "posting_cnt"], 3), axis=1)
        logging.info("--- 当前完成" + str(round(current_type+1 / 13, 2)) + "，耗时" + str(
            (dt.datetime.now() - current_time).seconds) + "秒。")
    df_word_type_ratio.to_excel("Statistics/ratio.xlsx", index=None)

    # 统计各单词在各类帖子中出现的占比
    df_word_type_cnt = pd.read_excel("Statistics/单词在各类型帖子出现的次数.xlsx", sheet_name=0)
    df_word_type_cnt["all_cnt"] = df_word_type_cnt.apply(lambda x: x.sum()-x["word"], axis=1)
    df_word_type_ratio = pd.DataFrame({"word": df_word_type_cnt["word"].values})
    for current_type in range(14):
        df_word_type_ratio["type_" + str(current_type) + "_ratio"] = df_word_type_cnt.apply(
            lambda x: round(x["type_"+str(current_type)]/x["all_cnt"], 3) if x["all_cnt"] > 0 else 0, axis=1)
        logging.info("--- 当前完成" + str(round((current_type + 1) / 14, 2)) + "，耗时" + str(
            (dt.datetime.now() - current_time).seconds) + "秒。")
    df_word_type_ratio["gini"] = df_word_type_ratio.apply(lambda x: calculate_gini(x), axis=1)
    df_word_type_ratio.to_excel("Statistics/ratio.xlsx", index=None)
    word_gini = df_word_type_ratio[["word", "gini"]].set_index("word").copy()

    # 每篇文章出现的基尼系数最低的基尼系数的值
    df_word_type_ratio = pd.read_excel("Statistics/单词在各类型帖子出现的次数.xlsx", sheet_name=2)
    word_gini = df_word_type_ratio[["word", "gini"]].copy()

    len(word_gini)


    temp_data_train = copy(data_train)
    temp_data_train["lower_gini"] = temp_data_train.apply(lambda x: calculate_posting_lower_gini(x["text"], word_gini), axis=1)
    # 最大值为0.858
    max(temp_data_train["lower_gini"].values)

    len(word_gini[word_gini["gini"]>0.858])
