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
print(len(index))
with tf.Session() as sess:
    indices = np.array([[3, 2, 0], [4, 5, 1]], dtype=np.int64)
    values = np.array([1.0] * len(index), dtype=np.float64)
    shape = np.array([9, 12], dtype=np.int64)
    sparse_trensor_x = sess.run(x, feed_dict={x: tf.SparseTensorValue(indices, values, shape)})
    print(sess.run(x, feed_dict={x: tf.SparseTensorValue(indices, values, shape)}))  # Will succeed.)
    # print(sess.run(y, feed_dict={x: tf.SparseTensorValue(indices, values, shape)}))

    # dense_trensor_x = tf.sparse_to_dense(sparse_trensor_x, [9, 12], 1.0)
    # print(sess.run(sparse_trensor_x))
    # sess.run(tf.sparse_tensor_to_dense(sparse_trensor_x))

x = tf.sparse_placeholder(tf.float32)
y = tf.sparse_reduce_sum(x)

with tf.Session() as sess:
    indices = np.array(
        [[0, 0], [0, 3], [0, 6], [1, 1], [1, 2], [1, 9], [2, 1], [2, 2], [2, 8], [3, 0], [3, 4], [3, 7], [4, 0], [4, 3],
         [4, 10], [5, 0], [5, 3], [5, 11], [6, 1], [6, 2], [6, 8], [7, 1], [7, 2], [7, 9], [8, 1], [8, 4], [8, 5]],
        dtype=np.int64)
    values = np.array([1.0]*27, dtype=np.float32)
    shape = np.array([9, 12], dtype=np.int64)
    print(sess.run(y, feed_dict={
        x: tf.SparseTensorValue(indices, values, shape)}))  # Will succeed.
    print(sess.run(y, feed_dict={
        x: (indices, values, shape)}))  # Will succeed.

    sp = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    sp_value = sp.eval(session=sess)
    print(sess.run(y, feed_dict={x: sp_value}))  # Will succeed
