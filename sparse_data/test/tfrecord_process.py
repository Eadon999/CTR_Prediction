#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 3/4/19 10:23 AM
# @Author : YangXiangDong
# @File   : tfrecord_process.py


import tensorflow as tf
import pandas as pd
import numpy as np

train = pd.read_csv('/home/mnt/rank_test_datasets/fm_test.csv')
print(train)

label = train['labels'].values
y_train = train.iloc[:, :-1].values

writer = tf.python_io.TFRecordWriter('train_csv.tfrecords')
print(y_train[1].shape)

for i in range(y_train.shape[0]):
    image_raw = y_train[i].tostring()
    example = tf.train.Example(
        # 需要主要此处是tf.train_model.Features，下面的是tf.train_model.Feature，差别在于一个's'
        features=tf.train.Features(
            feature={
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label[i]])),
            }
        )
    )
    writer.write(record=example.SerializeToString())
writer.close()
