#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import tensorflow as tf
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import numpy as np
from FM.FM import FM
from sparse_data.generate_data import DataPreprocess

TRAIN_LOG_PATH = '/home/mnt/model_file'


def check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state("checkpoints")
    if ckpt and ckpt.model_checkpoint_path:
        logging.info("Loading parameters for the my CNN architectures...")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        logging.info("Initializing fresh parameters for the my Factorization Machine")


def train_model(sess, fm_model, data_processor, epochs=10, print_every=50):
    """training model"""
    # Merge all the summaries and write them out to train_logs
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(TRAIN_LOG_PATH, 'train_logs'), sess.graph)
    for e in range(epochs):
        # batch_size data
        # create a feed dictionary for this batch
        batch_x, batch_y = data_processor.next_batch()
        indices, values, shape = data_processor.get_sparse_value_param(input_x=batch_x, batch_size=fm_model.batch_size,
                                                                       feature_dim=fm_model.feature_dim)

        feed_dict = {fm_model.X: tf.SparseTensorValue(indices, values, shape),
                     fm_model.y: batch_y,
                     fm_model.keep_prob: 1.0}

        loss, accuracy, summary, global_step, _ = sess.run([fm_model.loss, fm_model.accuracy,
                                                            merged, fm_model.global_step,
                                                            fm_model.train_op], feed_dict=feed_dict)
        print("Epoch:{}, loss:{}, accuracy:{}".format(e, loss, accuracy))

        # Record summaries and train.csv-set accuracy
        train_writer.add_summary(summary, global_step=global_step)
        # print training loss and accuracy
        if global_step % print_every == 0:
            logging.info("Iteration {0}: with minibatch training loss = {1} and accuracy of {2}"
                         .format(global_step, loss, accuracy))
            saver.save(sess, os.path.join(TRAIN_LOG_PATH, "checkpoints/model"), global_step=global_step)


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
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='train or test', type=str)
    args = parser.parse_args()
    mode = args.mode
    """
    mode = 'train'

    # set parameter
    config = {}
    config['lr'] = 0.01
    config['batch_size'] = 4
    config['reg_l1'] = 2e-2
    config['reg_l2'] = 0
    config['k'] = 40

    # data processer
    csv_data_path = '/home/mnt/test_datasets/test.csv'

    data_processor = DataPreprocess(batch_size=5)

    # initialize fm_model model
    fm_model = FM(config)
    # build graph for fm_model model
    fm_model.build_graph()

    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # restore trained parameters
        check_restore_parameters(sess, saver)
        if mode == 'train':
            print('start training...')
            train_model(sess, fm_model, data_processor, epochs=20, print_every=500)
        if mode == 'test':
            print('start testing...')
            test_model(sess, fm_model)
