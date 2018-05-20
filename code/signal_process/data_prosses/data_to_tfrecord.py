import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
def normalization(data, axis=0):
    m = np.mean(data, axis=axis)
    s = np.std(data, axis=axis)
    data = (data - m) / s
    return data

def encode_to_tfrecords(rootdir_path, tfsave_path):
    #rootdir_path = r'F:\data2\train.txt'
    inputs = np.loadtxt(rootdir_path, str)
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    inputs = inputs[indices]
    writer = tf.python_io.TFRecordWriter(tfsave_path)

    for each_path, label in inputs:
        print(each_path, label)
        data = np.loadtxt(each_path).astype(np.float32)
        data = normalization(data, axis=0)
        label = np.uint8(label)
        data_string = data.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'txtdata': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_string]))}))
        writer.write(example.SerializeToString())
    writer.close()

def read_and_decode(tf_filepath):
    filename_queue = tf.train.string_input_producer([tf_filepath], shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           "txtdata": tf.FixedLenFeature([], tf.string), })
    data = tf.decode_raw(features['txtdata'], tf.float32)
    data = tf.reshape(data, [4096, 4])
    label = tf.cast(features['label'], tf.uint8)
    print(data.shape)
    return data, label



def get_batch(data, label, batch_size, is_train=True):
    capacity = 1000 + 3 * batch_size
    if is_train:
        data_batch, label_batch = tf.train.shuffle_batch([data, label],
                                                         batch_size=batch_size,
                                                         capacity=capacity,
                                                         min_after_dequeue=1000)
    else:
        data_batch, label_batch = tf.train.batch([data, label],
                                                 batch_size=batch_size,
                                                 capacity=capacity)
    return data_batch, label_batch


if __name__ == '__main__':
    encode_to_tfrecords(r'F:\data2\train_data_16.txt', r'F:\data2\data_train_normal_16.tfrecords')
    # data, label = read_and_decode(r'F:\data2\data_train.tfrecords')
    # data = data[:, 0:1]
    # data_batch, label_batch = get_batch(data=data, label=label, batch_size=32, is_train=False)
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #     for i in range(1):
    #         print('run.........')
    #         data_batchss, label_batchss = sess.run([data_batch, label_batch])
    #         print('end.........')
    #         print(data_batchss.shape)
    #         print(data_batchss.shape)
    #
    # # print(data_batch)
    #         print(i, label_batchss)
    #
    # coord.request_stop()
    # coord.join(threads)
    #
    # for num , data in enumerate(data_batchss):
    #     plt.figure(num)
    #     plt.plot(data)
    # plt.show()
