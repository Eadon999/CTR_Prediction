#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 3/4/19 10:02 AM
# @Author : YangXiangDong
# @File   : saprse_placeholder.py
import tensorflow as tf
import numpy as np

x = tf.sparse_placeholder(tf.float32)
y = tf.sparse_reduce_sum(x)
index = [[0, 0], [0, 3], [0, 6], [1, 1], [1, 2], [1, 9], [2, 1], [2, 2], [2, 8], [3, 0], [3, 4], [3, 7], [4, 0], [4, 3],
         [4, 10], [5, 0], [5, 3], [5, 11], [6, 1], [6, 2], [6, 8], [7, 1], [7, 2], [7, 9], [8, 1], [8, 4], [8, 5]]

x = tf.sparse_placeholder(tf.float32)
# y = tf.sparse_reduce_sum(x)

with tf.Session() as sess:
    indices = np.array(index, dtype=np.int64)
    values = np.array([1] * len(index), dtype=np.float64)
    shape = np.array([9, 12], dtype=np.int64)
    sp_ten = sess.run(x, feed_dict={x: tf.SparseTensorValue(indices, values, shape)})
    print('tessor:\n', sess.run(tf.sparse_tensor_to_dense(sp_ten)))
