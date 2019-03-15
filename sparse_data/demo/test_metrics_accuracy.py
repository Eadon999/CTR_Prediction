#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 3/15/19 11:06 AM
# @Author : YangXiangDong
# @File   : test_metrics_accuracy.py

import os

# supress tensorflow logging other than errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

print(tf.__version__)
# 1.1.0

x = tf.placeholder(tf.int32, [5])
y = tf.placeholder(tf.int32, [5])
acc, acc_op = tf.metrics.accuracy(labels=x, predictions=y)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

v = sess.run([acc, acc_op], feed_dict={x: [1, 0, 0, 0, 0],
                                       y: [1, 0, 0, 0, 1]})
print(v)
# acc 与acc op都 运行，得到的都是正确率，不过acc是用来更新的，当前的feed_dict不计入
# 另外，acc_op会维护历史数据，acc只是从历史数据中获得结果
# 如果不运行acc_op, 历史数据不会更新
# 所以acc输出是0， acc_op是0.8
# [0.0, 0.8]

v = sess.run(acc)
print(v)
# 这里单独运行acc，对历史数据做统计，输出0.8
# 0.8

v = sess.run([acc_op], feed_dict={x: [1, 0, 0, 0, 0],
                                  y: [0, 1, 1, 1, 1]})
print(v)
# 这里单独运行acc_op, 历史正确率是0.4
# 0.4


v = sess.run([acc], feed_dict={x: [0, 1, 1, 1, 1],
                               y: [0, 1, 1, 1, 1]})
print(v)
# 这里单独运行acc，输出历史正确率0.4，历史数据得不到更新
# 实际的历史正确率是0.6
# 0.4

sess.close()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
v = sess.run([acc, acc_op], feed_dict={x: [1, 0, 0, 0, 0],
                                       y: [1, 0, 0, 0, 1]})
print(v)
# 重新开始一个session，重新计算正确率
# [0.0, 0.8]
