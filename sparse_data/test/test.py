#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 3/4/19 2:28 PM
# @Author : YangXiangDong
# @File   : test.py


import numpy as np
import tensorflow as tf

CLASS = 2

b = tf.one_hot([0, 1, 1, 1, 0, 0, 0, 1, 1], CLASS, 1, 0)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    k = sess.run(b)
    print(k)
    print('after one_hot', k[0:2])
