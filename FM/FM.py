# coding:utf-8
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import tensorflow as tf
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class FM:
    """
    Factorization Machine with FTRL optimization
    """

    def __init__(self, param):
        """
        :param config: configuration of hyperparameters
        type of dict
        """
        # number of latent factors
        self.k = param['k']
        self.lr = param['lr']
        self.batch_size = param['batch_size']
        self.reg_l1 = param['reg_l1']
        self.reg_l2 = param['reg_l2']
        # num of features
        self.feature_dim = param['feature_dim']

    def add_placeholders(self):
        self.X = tf.sparse_placeholder('float32', [None, self.feature_dim])
        self.y = tf.placeholder('float32', [None, 2])
        self.keep_prob = tf.placeholder('float32')

    def inference(self):
        """
        forward propagation
        :return: labels for each sample
        """
        with tf.variable_scope('linear_layer'):
            b = tf.get_variable('bias', shape=[2],
                                initializer=tf.zeros_initializer())
            w1 = tf.get_variable('w1', shape=[self.feature_dim, 2],
                                 initializer=tf.truncated_normal_initializer(mean=0, stddev=1e-2))

            self.linear_terms = tf.add(tf.sparse_tensor_dense_matmul(self.X, w1), b)

        with tf.variable_scope('interaction_layer'):
            v = tf.get_variable('v', shape=[self.feature_dim, self.k],
                                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))

            # self.interaction_terms = tf.multiply(0.5,
            #                                      tf.reduce_mean(
            #                                          tf.subtract(
            #                                              tf.pow(tf.sparse_tensor_dense_matmul(self.X, v), 2),
            #                                              tf.sparse_tensor_dense_matmul(self.X, tf.pow(v, 2))),
            #                                          1, keep_dims=True))
            self.interaction_terms = tf.multiply(0.5,
                                                 tf.reduce_sum(
                                                     tf.subtract(
                                                         tf.pow(tf.sparse_tensor_dense_matmul(self.X, v), 2),
                                                         tf.sparse_tensor_dense_matmul(self.X, tf.pow(v, 2))),
                                                     1, keep_dims=True))

        self.y_out = tf.add(self.linear_terms, self.interaction_terms)
        self.y_out_prob = tf.nn.softmax(self.y_out)

    def add_loss(self, is_y_onehot=False):
        # row is the index of labels is actual classes , logits is not yet softmax original output value
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y_out,
                                                                       labels=tf.argmax(self.y, 1))

        self.cross_entropy = cross_entropy
        mean_loss = tf.reduce_mean(self.cross_entropy)
        self.loss = mean_loss
        tf.summary.scalar('loss', self.loss)

    def add_accuracy(self):
        # accuracy
        self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.y_out_prob, 1), tf.float32),
                                           tf.cast(tf.argmax(self.y, 1), tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        # add summary to accuracy
        tf.summary.scalar('accuracy', self.accuracy)

    def train(self):
        # Applies exponential decay to learning rate
        self.global_step = tf.Variable(0, trainable=False)
        # define optimizer
        optimizer = tf.train.FtrlOptimizer(self.lr, l1_regularization_strength=self.reg_l1,
                                           l2_regularization_strength=self.reg_l2)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    """
    def gd_train(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.train_op = tf.train_model.GradientDescentOptimizer(0.001).minimize(self.loss)
    """

    def build_graph(self):
        """build graph for model"""
        self.add_placeholders()
        self.inference()
        self.add_loss()
        self.add_accuracy()
        self.train()
        # self.gd_train()
