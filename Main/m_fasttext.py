# -*- coding: UTF-8 -*- 
# 作者：hao.ren3
# 时间：2020/8/20 16:04
# IDE：PyCharm

import fasttext
from sklearn.metrics import f1_score
from Tools.data import get_train_data
import os
from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense

def generate_fasttext_train_data():
    temp_data = get_train_data()
    temp_data['label_ft'] = '__label__' + temp_data['label'].astype(str)
    temp_data[['text', 'label_ft']].iloc[:-5000].to_csv("Data/FastText/train.csv", index=None, header=None, sep="\t")
    temp_data[['text', 'label_ft']].iloc[-5000:].to_csv("Data/FastText/test.csv", index=None, header=None, sep="\t")

# fasttext原理
def build_fastTest():
    VOCAB_SIZE = 2000
    EMBEDDING_DIM = 100
    MAX_WORDS = 500
    CLASS_NUM = 5

    model = Sequential()
    # 通过embedding层，我们将词汇映射成Embedding_dim维向量
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_WORDS))

    # 通过GlobalAveragePooling1D层，平均文档中所有词的embedding
    model.add(GlobalAveragePooling1D())

    # 通过输出层SoftMax分类，得到类别概率分布。
    model.add(Dense(CLASS_NUM, activation="softmax"))

    # 定义损失函数、优化器、分类度量指标
    model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

    return model

if __name__ == '__main__':
    # fast_text的构建过程
    test_model = build_fastTest()
    test_model.summary()

    # 保存模型的地址
    model_path = "Model/model_fast_text.model"

    data = get_train_data()

    if not os.path.exists("Data/FastText/train.csv"):
        generate_fasttext_train_data()

    # 如果之前保存过模型，则不用再训练模型，直接从本地读取。否则则重新训练数据。
    if not os.path.exists(model_path):
        model = fasttext.train_supervised("Data/FastText/train.csv", lr=1.0, wordNgrams=2,
                                  verbose=2, minCount=1, epoch=25, loss="hs")
        model.save_model(model_path)
    else:
        model = fasttext.load_model(model_path)

    val_pred = [model.predict(x)[0][0].split('__')[-1] for x in data.iloc[-5000:]['text']]

    # 0.910261482839271
    print(f1_score(data['label'].values[-5000:].astype(str), val_pred, average='macro'))
