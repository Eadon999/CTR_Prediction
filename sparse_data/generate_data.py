#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 3/4/19 11:03 AM
# @Author : YangXiangDong
# @File   : generate_data.py

import pandas as pd
import numpy as np
import tensorflow as tf

train = pd.read_csv('/home/mnt/rank_test_datasets/fm_test.csv')


class DatePreprocess:
    def __init__(self):
        self.columns_name = ['feature_1', 'feature_2', 'feature_3']
        self.df = self.read_csv('/home/mnt/rank_test_datasets/fm_test.csv')
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
            labels.append(row['labels'])
            feature_len = 0
            for col in columns:
                feature_index = self.feature_map.get(col).get(row[col])
                feature_indices_list.append([index, feature_len + feature_index])
                feature_len += self.feature_len.get(col)
        return feature_indices_list, labels

    def run_generate(self):
        features, labels = self.generate_sparse_data(self.df, self.columns_name)
        CLASS = 2
        onehot_func = tf.one_hot(labels, CLASS, 1, 0, dtype=tf.float32)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            labels_onehot = sess.run(onehot_func)
        return features, labels_onehot


if __name__ == '__main__':
    handler = DatePreprocess()
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

    x, y = handler.run_generate()
    print(x)
    print(y)
