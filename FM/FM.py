# coding:utf-8
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import tensorflow as tf
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import numpy as np
import argparse
from sparse_data.generate_data import DatePreprocess


class FM(object):
    """
    Factorization Machine with FTRL optimization
    """

    def __init__(self, config):
        """
        :param config: configuration of hyperparameters
        type of dict
        """
        # number of latent factors
        self.k = config['k']
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.reg_l1 = config['reg_l1']
        self.reg_l2 = config['reg_l2']
        # num of features
        self.p = config['feature_len']

    def add_placeholders(self):
        self.X = tf.sparse_placeholder('float32', [None, self.p])
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
            w1 = tf.get_variable('w1', shape=[self.p, 2],
                                 initializer=tf.truncated_normal_initializer(mean=0, stddev=1e-2))
            # shape of [None, 2]
            self.linear_terms = tf.add(tf.sparse_tensor_dense_matmul(self.X, w1), b)

        with tf.variable_scope('interaction_layer'):
            v = tf.get_variable('v', shape=[self.p, self.k],
                                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            # shape of [None, 1]
            self.interaction_terms = tf.multiply(0.5,
                                                 tf.reduce_mean(
                                                     tf.subtract(
                                                         tf.pow(tf.sparse_tensor_dense_matmul(self.X, v), 2),
                                                         tf.sparse_tensor_dense_matmul(self.X, tf.pow(v, 2))),
                                                     1, keep_dims=True))
        # shape of [None, 2]
        self.y_out = tf.add(self.linear_terms, self.interaction_terms)
        self.y_out_prob = tf.nn.softmax(self.y_out)

    def add_loss(self):
        # labels的每一行为真实类别的索引
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model.y_out,
                                                                       labels=tf.argmax(self.y, 1))
        self.cross_entropy = cross_entropy
        mean_loss = tf.reduce_mean(self.cross_entropy)
        self.loss = mean_loss
        tf.summary.scalar('loss', self.loss)

    def add_accuracy(self):
        # accuracy
        self.correct_prediction = tf.equal(tf.cast(tf.argmax(model.y_out, 1), tf.float32),
                                           tf.cast(tf.argmax(model.y, 1), tf.float32))
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

    def build_graph(self):
        """build graph for model"""
        self.add_placeholders()
        self.inference()
        self.add_loss()
        self.add_accuracy()
        self.train()


def check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state("checkpoints")
    if ckpt and ckpt.model_checkpoint_path:
        logging.info("Loading parameters for the my CNN architectures...")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        logging.info("Initializing fresh parameters for the my Factorization Machine")


def train_model(sess, model, epochs=10, print_every=50):
    """training model"""
    data_pro = DatePreprocess()
    # Merge all the summaries and write them out to train_logs
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train_logs', sess.graph)
    # get sparse training data
    x, y = data_pro.run_generate()
    # get number of batches
    num_batches = len(y)
    indices = np.array(x, dtype=np.int64)
    values = np.array([1] * len(x), dtype=np.float32)
    shape = np.array([9, 12], dtype=np.int64)
    for e in range(epochs):
        num_samples = 0
        losses = []
        # batch_size data
        batch_y = y
        # create a feed dictionary for this batch
        feed_dict = {model.X: tf.SparseTensorValue(indices, values, shape),
                     model.y: batch_y,
                     model.keep_prob: 1.0}

        loss, accuracy, summary, global_step, _ = sess.run([model.loss, model.accuracy,
                                                            merged, model.global_step,
                                                            model.train_op], feed_dict=feed_dict)
        print("Epoch:{}, loss:{}, accuracy:{}".format(e, loss, accuracy))

        # Record summaries and train.csv-set accuracy
        train_writer.add_summary(summary, global_step=global_step)
        # print training loss and accuracy
        if global_step % print_every == 0:
            logging.info("Iteration {0}: with minibatch training loss = {1} and accuracy of {2}"
                         .format(global_step, loss, accuracy))
            saver.save(sess, "checkpoints/model", global_step=global_step)


def test_model(sess, model, print_every=50):
    """training model"""
    # get testing data, iterable
    with open('../avazu_CTR/test_sparse_data_frac_0.01.pkl', 'rb') as f:
        test_sparse_data_fraction = pickle.load(f)
    all_ids = []
    all_clicks = []
    # get number of batches
    num_batches = len(test_sparse_data_fraction)

    for ibatch in range(num_batches):
        # batch_size data
        batch_ids = test_sparse_data_fraction[ibatch]['id']
        batch_indexes = test_sparse_data_fraction[ibatch]['indexes']
        actual_batch_size = len(batch_indexes) // 21
        batch_shape = np.array([actual_batch_size, feature_length], dtype=np.int64)
        batch_values = np.ones(len(batch_indexes), dtype=np.float32)
        # create a feed dictionary for this15162 batch
        feed_dict = {model.X: (batch_indexes, batch_values, batch_shape),
                     model.keep_prob: 1}
        # shape of [None,2]
        y_out_prob = sess.run([model.y_out_prob], feed_dict=feed_dict)
        batch_clicks = y_out_prob[0][:, -1]

        all_ids.extend(batch_ids)
        all_clicks.extend(batch_clicks)

        ibatch += 1
        if ibatch % print_every == 0:
            logging.info("Iteration {0} has finished".format(ibatch))

    pd.DataFrame(np.array([all_ids, all_clicks]).T, columns=['id', 'click']).to_csv('result_regl1_.csv', index=False)


if __name__ == '__main__':
    '''launching TensorBoard: tensorboard --logdir=path/to/log-directory'''
    # get mode (train or test)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='train or test', type=str)
    args = parser.parse_args()
    mode = args.mode
    mode = 'train'
    # original fields
    fields = ['hour', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
              'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_domain',
              'app_id', 'app_category', 'device_model', 'device_type', 'device_id',
              'device_conn_type', 'click']
    # initialize the model
    config = {}
    config['lr'] = 0.001
    config['batch_size'] = 512
    config['reg_l1'] = 2e-2
    config['reg_l2'] = 0
    config['k'] = 40
    config['feature_len'] = 12
    # get feature length
    # initialize FM model
    model = FM(config)
    # build graph for model
    model.build_graph()

    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        # TODO: with every epoches, print training accuracy and validation accuracy
        sess.run(tf.global_variables_initializer())
        # restore trained parameters
        check_restore_parameters(sess, saver)
        if mode == 'train':
            print('start training...')
            train_model(sess, model, epochs=20, print_every=500)
        if mode == 'test':
            print('start testing...')
            test_model(sess, model)
