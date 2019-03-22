#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf


class DataPreprocess:
    def __init__(self, batch_size):
        self.columns_name = ['feature_1', 'feature_2', 'feature_3']
        self.df = self.read_csv('fm_test.csv')
        self.gender_values_set = set(self.df['feature_1'])
        self.channel_category_set = set(self.df['feature_2'])
        self.key_words_set = set(self.df['feature_3'])
        # generate map dict
        self.gender_map = self.generate_mapping_relation(self.gender_values_set)
        self.category_map = self.generate_mapping_relation(self.channel_category_set)
        self.key_words_map = self.generate_mapping_relation(self.key_words_set)
        self.feature_len = {'feature_1': len(self.gender_map), 'feature_2': len(self.category_map),
                            'feature_3': len(self.key_words_map)}
        self.feature_map = {'feature_1': self.gender_map, 'feature_2': self.category_map,
                            'feature_3': self.key_words_map}
        self.batch_data_iter = self.get_batch_iter(batch_size)

    def get_data(self):
        pass

    def generate_sparse_feature_map(self, df, col_name):
        pass

    def get_feature_length(self):
        length_dict = {}
        pass

    def read_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        return df

    def generate_mapping_relation(self, values_list):
        map_dict = {label: idx for idx, label in enumerate(values_list)}
        return map_dict

    def generate_sparse_data(self, df, columns):
        feature_indices_list = list()
        labels = list()
        for index, row in df.iterrows():
            row_feature_indices = list()
            labels.append(row['labels'])
            feature_len = 0
            for col in columns:
                feature_index = self.feature_map.get(col).get(row[col])
                row_feature_indices.append([index, feature_len + feature_index])
                feature_len += self.feature_len.get(col)
            feature_indices_list.append(row_feature_indices)
        return feature_indices_list, labels

    def batch_generator(self, all_data, batch_size, shuffle=True):
        """
        :param all_data : all_data整个数据集
        :param batch_size: batch_size表示每个batch的大小
        :param shuffle: 每次是否打乱顺序
        :return:
        """
        all_data = [np.array(d) for d in all_data]
        data_size = all_data[0].shape[0]
        print("data size:{}, your choose batch:{}".format(data_size, batch_size))
        if shuffle:
            p = np.random.permutation(data_size)
            all_data = [d[p] for d in all_data]

        batch_count = 0
        while True:
            if batch_count * batch_size + batch_size > data_size:
                batch_count = 0
                if shuffle:
                    p = np.random.permutation(data_size)
                    all_data = [d[p] for d in all_data]
            start = batch_count * batch_size
            end = start + batch_size
            batch_count += 1
            yield [d[start: end] for d in all_data]

    def run_generate(self):
        features, labels = self.generate_sparse_data(self.df, self.columns_name)
        CLASS = 2
        onehot_func = tf.one_hot(labels, CLASS, 1.0, 0.0, dtype=tf.float32)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            labels_onehot = sess.run(onehot_func)
        return features, labels, labels_onehot

    def get_batch_iter(self, batch_size):
        feature_indice, label, one_hot_label = self.run_generate()
        self.feature_indice_value = feature_indice
        self.label_value = label
        self.onehot_label_value = one_hot_label
        batch_data_iter = self.batch_generator(feature_indice, label, batch_size)
        return batch_data_iter

    def reorder_feature_indices(self, original_data):
        """
        reoeder feature one hot index values
        :param original_data: before reorder data
        :return:
        """
        batch_row = 0
        reorder_value = list()  #
        for sub_list in original_data:
            sub_list.sort()
            batch_index = list()
            for row, col in sub_list:
                batch_index.append([batch_row, col])
            reorder_value.extend(batch_index)
            batch_row += 1
        return reorder_value

    def next_batch(self):
        original_batch_x, batch_y = next(self.batch_data_iter)
        """alter batch_x format to feed tf.SparseTensorValue(indices, values, shape)"""
        """the indices value of tf.SparseTensorValue的SparseValue must be ordered"""
        actual_batch_x = self.reorder_feature_indices(original_batch_x)
        return actual_batch_x, batch_y


    def multi_process_target(self, df, labels_list, feature_indices_list):
        for index, row in df.iterrows():
            if not pd.isnull(row['label']):

                feature_indices = self.generate_feature_indices(index, row)
                feature_indices_list.append(feature_indices)
                labels_list.append(row['label'])
            else:
                print('row:{}, labels is Nan, drop the data'.format(index))
                logging.info('row:{}, labels is Nan, drop the data'.format(index))

    def generate_sparse_data(self, df):

        process_list = list()
        manager = multiprocessing.Manager()
        feature_indices_list = manager.list()
        labels_list = manager.list()
        sub_size = len(df) // self.n_processes + 1

        for idx in range(self.n_processes):
            proc = multiprocessing.Process(target=self.multi_process_target,
                                           args=(df[idx * sub_size: (idx + 1) * sub_size],
                                                 labels_list, feature_indices_list,))
            process_list.append(proc)

        for p in process_list:
            p.start()

        for p in process_list:
            p.join()


    def run(self):
        self.n_processes = 3
        self.map_dict = {}
        # if self.is_new_data:
        #     self.df_iteration = self.read_csv(csv_path, batch_size)
        all_time = time.time()
        feature_f = open('.txt', 'w')
        label_f = open('.txt', 'w')
        for iter in self.df_iteration:
            # x, y = self.generate_sparse_data(iter)
            #
            # print(len(x), len(y))
            batch_time = time.time()
            batch_data = self.get_sparse_data(iter)
            feature_list = [json.dumps(i) + '\n' for i in batch_data.feature_train]
            label_list = [json.dumps(i) + '\n' for i in batch_data.lable_train]
            feature_f.writelines(feature_list)
            label_f.writelines(label_list)
            print('batch write using {}'.format(time.time() - batch_time))
        print('all write using {}'.format(time.time() - all_time))
        feature_f.close()
        label_f.close()
        print('close end')


if __name__ == '__main__':
    batch_size = 3
    handler = DataPreprocess(batch_size)
    columns_name = ['feature_1', 'feature_2', 'feature_3']
    df = handler.read_csv('/home/mnt/rank_test_datasets/fm_test.csv')
    gender_values_set = set(df['feature_1'])
    channel_category_set = set(df['feature_2'])
    key_words_set = set(df['feature_3'])
    # generate map dict
    gender_map = handler.generate_mapping_relation(gender_values_set)
    category_map = handler.generate_mapping_relation(channel_category_set)
    key_words_map = handler.generate_mapping_relation(key_words_set)
    handler.generate_sparse_data(df, columns_name)

    x, y = handler.next_batch()
    print(x)
    print(y)
