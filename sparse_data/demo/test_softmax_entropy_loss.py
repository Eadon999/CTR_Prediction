#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 3/11/19 3:18 PM
# @Author : YangXiangDong
# @File   : test_softmax_entropy_loss.py

import tensorflow as tf
import numpy as np

logits = np.array([[11, 22], [33, 44], [55, 66]])

labels = np.array([1, 0, 1])
"""
tf.nn.sparse_softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, name=None)
这个函数与上一个函数十分类似，唯一的区别在于labels.
注意：
对于此操作,labels的每一行为真实类别的索引
logits这个操作的输入logits同样是是未经softmax缩放的，该操作内部会对logits使用softmax操作
警告：
1. 这个操作的输入logits同样是是未经缩放的，该操作内部会对logits使用softmax操作
2. 参数logits的形状 [batch_size, num_classes] 和labels的形状[batch_size]
返回值：长度为batch_size的一维Tensor, 和label的形状相同，和logits的类型相同
"""

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)  # make sure you do this!
print(sess.run(cross_entropy))

"""
tf.nn.softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, dim=-1, name=None)
Computes softmax cross entropy between logits and labels.
注意：
labels的每一行是one-hot表示，也就是只有一个地方为1，其他地方为0
logits这个操作的输入logits是未经缩放的，该操作内部会对logits使用softmax操作
警告：
1. 这个操作的输入logits是未经缩放的，该操作内部会对logits使用softmax操作
2. 参数labels,logits必须有相同的形状 [batch_size, num_classes] 和相同的类型(float16, float32, float64)中的一种
参数：_sentinel: 一般不使用
labels: labels的每一行labels[i]必须为一个概率分布
logits: 未缩放的对数概率
dims: 类的维度，默认-1，也就是最后一维
name: 该操作的名称
返回值：长度为batch_size的一维Tensor
"""