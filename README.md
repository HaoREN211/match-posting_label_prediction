# 文本挖掘-帖子类型分类

[比赛链接](https://tianchi.aliyun.com/competition/entrance/531810/introduction)
[GIT链接](https://github.com/HaoREN211/match-posting_label_prediction)

### 数据下载

1. [训练集数据](https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531810/train_set.csv.zip)
1. [测试集数据](https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531810/test_a.csv.zip)
1. [测试集提交样例](https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531810/test_a_sample_submit.csv)

### 数据说明

> 以新闻数据为目标数据，数据集见链接下载。数据为新闻文本，并按照字符级别进行匿名处理。整合划分出14个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐的文本数据。

在数据集中标签的对应的关系如下：

```python
{
    '科技': 0,
    '股票': 1,
    '体育': 2,
    '娱乐': 3,
    '时政': 4,
    '社会': 5,
    '教育': 6,
    '财经': 7,
    '家居': 8,
    '游戏': 9,
    '房产': 10,
    '时尚': 11,
    '彩票': 12,
    '星座': 13
}
```
