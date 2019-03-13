#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 3/13/19 9:15 AM
# @Author : YangXiangDong
# @File   : test_batch_data.py
import numpy as np


def batch_generator(x_input, y_input, batch_size, shuffle=True):
    """
    :param x_input: all_feature整个数据集
    :param y_input: all labels
    :param batch_size: batch_size表示每个batch的大小
    :param shuffle: 每次是否打乱顺序
    :return:
    """
    all_data = [x_input, y_input]
    all_data = [np.array(d) for d in all_data]
    data_size = all_data[0].shape[0]
    print("data size:{}, your choose batch:{}".format(data_size, batch_size))
    if shuffle:
        p = np.random.permutation(data_size)
        all_data = [d[p] for d in all_data]

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > data_size:
            batch_count = 0
            if shuffle:
                p = np.random.permutation(data_size)
                all_data = [d[p] for d in all_data]
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start: end] for d in all_data]


# 输入x表示有23个样本，每个样本有两个特征
# 输出y表示有23个标签，每个标签取值为0或1
x = [[[0, 9], [0, 768030], [0, 765486], [0, 760323], [0, 765663], [0, 767760], [0, 768404], [0, 768822], [0, 757348],
      [0, 761270], [0, 768762], [0, 2306432], [0, 2306439], [0, 2306458], [0, 3068203], [0, 3075221], [0, 2726515],
      [0, 3075143], [0, 3069324], [0, 3074964], [0, 3071132], [0, 3075263], [0, 3074663], [0, 3075080]],
     [[1, 10], [1, 767669], [1, 768613], [1, 768720], [1, 765217], [1, 768219], [1, 768812], [1, 768733], [1, 768132],
      [1, 768811], [1, 764118], [1, 2306432], [1, 2306449], [1, 2306458], [1, 3068203], [1, 3075221], [1, 2726515],
      [1, 3075143], [1, 3069324], [1, 3074964], [1, 3071132], [1, 3075263], [1, 3074663], [1, 3075080]],
     [[2, 9], [2, 767803], [2, 766353], [2, 765693], [2, 717864], [2, 767956], [2, 768413], [2, 768438], [2, 764267],
      [2, 765911], [2, 764349], [2, 1486667], [2, 2306432], [2, 2306439], [2, 2306458], [2, 3052447], [2, 3069613],
      [2, 3073895], [2, 3066980], [2, 3075261], [2, 3075131], [2, 3074173], [2, 3074764], [2, 3074307], [2, 3025723]],
     [[3, 9], [3, 768036], [3, 765663], [3, 768822], [3, 767392], [3, 571217], [3, 607957], [3, 651303], [3, 768200],
      [3, 768725], [3, 761502], [3, 1536839], [3, 2305716], [3, 2306197], [3, 2306432], [3, 2306439], [3, 2306458],
      [3, 3052447], [3, 3069613], [3, 3073895], [3, 3066980], [3, 3075261], [3, 3075131], [3, 3074173], [3, 3074764],
      [3, 3074307], [3, 3025723]],
     [[4, 10], [4, 767742], [4, 766587], [4, 764858], [4, 767737], [4, 768685], [4, 768437], [4, 738448], [4, 767873],
      [4, 758037], [4, 763055], [4, 2306432], [4, 2306449], [4, 2306458], [4, 3052447], [4, 3069613], [4, 3073895],
      [4, 3066980], [4, 3075261], [4, 3075131], [4, 3074173], [4, 3074764], [4, 3074307], [4, 3025723]]]
y = [0, 1, 1, 0, 0]

batch_data = batch_generator(x, y, 2)
def test_batch():
    batch_x, batch_y = next(batch_data)
    k = list()
    for i in batch_x:
        k.extend(i)
    return k, batch_y


#
# batch_size = 2
# batch_data = batch_generator(x, y, batch_size)
# for i in range(20):
#     print('ephch:{}'.format(i))
#     batch_x, batch_y = next(batch_data)
#     k = list()
#     for i in batch_x:
#         k.extend(i)
#     print(k, batch_y)
#     print('++++++++++++')

for i in range(20):
    print('ephch:{}'.format(i))
    batch_x, batch_y = test_batch()
    print(batch_x, batch_y)
    print('++++++++++++')
