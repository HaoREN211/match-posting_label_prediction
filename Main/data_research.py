# @Time    : 2020/8/18 22:14
# @Author  : REN Hao
# @FileName: data_research.py
# @Software: PyCharm

from Tools.data import get_test_data, get_train_data

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
    label_cnt = {}
    for label, posting_cnt in data_train["label"].value_counts().items():
        label_cnt[label] = posting_cnt
