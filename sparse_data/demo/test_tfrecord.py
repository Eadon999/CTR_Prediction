import tensorflow as tf
import numpy as np
import json

# def _parse_function(filename, label):
#     image_string = tf.read_file(filename)
#     image_decoded = tf.image.decode_image(image_string)
#     image_resized = tf.image.resize_images(image_decoded, [28, 28])
#     return image_resized, label


# 图片文件的列表
filenames = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]

# label[i]就是图片filenames[i]的label
labels = [0, 37]
'''
# 此时dataset中的一个元素是(filename, label)
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

# 此时dataset中的一个元素是(image_resized_batch, label_batch)
dataset = dataset.shuffle(buffer_size=2).batch(1).repeat(2)

iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
'''
# with tf.Session() as sess:
#     try:
#         while True:
#             print('++++++++')
#             x, y = sess.run(one_element)
#             print(x, y)
#     except tf.errors.OutOfRangeError:
#         print("end!")

csv_data_path = 'E:/virtualboxshare/rank_test_datasets/tf_record_txt.txt'

data_set = tf.data.TextLineDataset(
    '/home/mnt/gitlab/personal_project/CTR_Prediction/sparse_data/demo/test.txt')
# data_set_y = tf.data.TextLineDataset(
#     r'E:\virtualboxshare\gitlab\personal_project\CTR_Prediction\sparse_data\demo\test_y.txt')
dataset = data_set.batch(2).repeat(4)
# dataset_y = data_set_y.batch(1).repeat(3)
iterator = dataset.make_one_shot_iterator()
# iterator_y = dataset_y.make_one_shot_iterator()
one_element = iterator.get_next()

with tf.Session() as sess:
    i = 0
    try:
        while True:

            batch_data = sess.run(one_element)


            batch_row = 0
            reorder_value = list()  #
            for sub_list in batch_data:
                sub_list = json.loads(sub_list.decode())
                batch_index = list()
                for row, col in sub_list:
                    batch_index.append([batch_row, col])
                reorder_value.extend(batch_index)
                batch_row += 1
            i += 1
            print('==========================', i)
            print(batch_data)
            print('x', reorder_value)


    except:
        print(i)
'''

with tf.Session() as sess:
    i = 0
    for i in range(9):

        batch_data = sess.run(one_element)

        batch_row = 0
        reorder_value = list()  #
        for sub_list in batch_data:
            sub_list = json.loads(sub_list.decode())
            batch_index = list()
            for row, col in sub_list:
                batch_index.append([batch_row, col])
            reorder_value.extend(batch_index)
            batch_row += 1
        i += 1
        print('==========================', i)
        print(batch_data)
        print('x', reorder_value)
'''