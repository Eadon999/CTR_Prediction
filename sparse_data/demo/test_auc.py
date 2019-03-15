#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 3/15/19 9:25 AM
# @Author : YangXiangDong
# @File   : test_auc.py


import tensorflow as tf

a = tf.Variable([0.2, 0.5])
b = tf.Variable([0.2, 0.6])

auc = tf.metrics.auc(a, b)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())  # try commenting this line and you'll get the error
train_auc = sess.run(auc)

print(train_auc)
