# @Time    : 2020/8/18 22:03
# @Author  : REN Hao
# @FileName: data.py
# @Software: PyCharm

import pandas as pd


# 获取训练集数据
def get_train_data():
    return pd.read_csv("Data/train_set.csv", sep="\t", header=0)


# 获取测试集数据
def get_test_data():
    return pd.read_csv("Data/test_set.csv", sep="\t", header=0)

