#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 3/4/19 2:28 PM
# @Author : YangXiangDong
# @File   : categorical_column_with_hash_bucket.py

import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _LazyBuilder

def test_categorical_column_with_hash_bucket():
    color_data = {'color': [[2], [5], [-1], [0]]}  # 4行样本
    builder = _LazyBuilder(color_data)
    color_column = tf.feature_column.categorical_column_with_hash_bucket('color', 7, dtype=tf.int32)
    color_column_tensor = color_column._get_sparse_tensors(builder)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([color_column_tensor.id_tensor]))

    # 将稀疏的转换成dense，也就是one-hot形式，只是multi-hot
    color_column_identy = tf.feature_column.indicator_column(color_column)
    color_dense_tensor = tf.feature_column.input_layer(color_data, [color_column_identy])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('use input_layer' + '_' * 40)
        print(session.run([color_dense_tensor]))


test_categorical_column_with_hash_bucket()