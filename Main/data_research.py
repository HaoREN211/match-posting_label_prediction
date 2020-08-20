# @Time    : 2020/8/18 22:14
# @Author  : REN Hao
# @FileName: data_research.py
# @Software: PyCharm

from Tools.data import get_test_data, get_train_data
import pandas as pd
import os
import numpy as np

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
