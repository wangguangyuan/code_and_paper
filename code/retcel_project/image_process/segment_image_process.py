# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os
import pandas as pd
from skimage import io
import random
import cv2
import tensorlayer as tl


try:
    from retcel_project.image_process import image_process as img_pro
except:
    from image_process import image_process as img_pro

try:
    from retcel_project.read_file_process import read_files
except:
    from read_file_process import read_files

try:
    from retcel_project.image_process import data_agument as dag
except:
    from image_process import data_agument as dag

def segment_fill_zeros(dicm_data, shape, bound_box, max_box, maskers):
    '''
    :param dicm_data:
    :param shape:
    :param bound_box:
    :param max_box:
    :return:
    '''
    dept = np.max([shape[0], max_box[0]])
    weight = np.max([shape[1], max_box[1]])
    height = np.max([shape[2], max_box[2]])

    dept_mid = dept / 2
    weight_mid = weight / 2
    height_mid = height / 2

    temp_mid = np.array([dept_mid, weight_mid, height_mid], np.float32)
    bound_lent = bound_box[:, 1] - bound_box[:, 0]

    bound_mid = bound_lent / 2

    #    bound_mid = max_box / 2
    low_bound = temp_mid - bound_mid
    up_bound = temp_mid + bound_mid

    low_bound = np.ceil(low_bound)
    up_bound = np.ceil(up_bound)

    temp_dicm = np.zeros([dept, weight, height], np.uint16)
    temp_masker = np.zeros([dept, weight, height], np.uint16)

    dicm_array = dicm_data[bound_box[0, 0]:bound_box[0, 1] + 1,
                 bound_box[1, 0]:bound_box[1, 1] + 1,
                 bound_box[2, 0]:bound_box[2, 1] + 1]

    dicm_masker = maskers[bound_box[0, 0]:bound_box[0, 1] + 1,
                  bound_box[1, 0]:bound_box[1, 1] + 1,
                  bound_box[2, 0]:bound_box[2, 1] + 1]

    temp_dicm[int(low_bound[0]):int(up_bound[0]) + 1,
    int(low_bound[1]):int(up_bound[1]) + 1,
    int(low_bound[2]):int(up_bound[2]) + 1] = dicm_array

    temp_masker[int(low_bound[0]):int(up_bound[0]) + 1,
    int(low_bound[1]):int(up_bound[1]) + 1,
    int(low_bound[2]):int(up_bound[2]) + 1] = dicm_masker

    max_box_mid = max_box / 2
    crop_low_bound = temp_mid - max_box_mid
    crop_up_bound = temp_mid + max_box_mid

    crop_low_bound = np.ceil(crop_low_bound)
    crop_up_bound = np.ceil(crop_up_bound)

    temp_array = temp_dicm[int(crop_low_bound[0]):int(crop_up_bound[0]),
                 int(crop_low_bound[1]):int(crop_up_bound[1]),
                 int(crop_low_bound[2]):int(crop_up_bound[2])]

    masker = temp_masker[int(crop_low_bound[0]):int(crop_up_bound[0]),
             int(crop_low_bound[1]):int(crop_up_bound[1]),
             int(crop_low_bound[2]):int(crop_up_bound[2])]

    return temp_array, masker


def _adjust_to_mid(bound_box, max_box, shape):
    if np.any(max_box > shape):
        raise ValueError('max_box.shape must less shape')
    bound_box_mid = np.mean(bound_box, 1)
    max_box_mid = max_box / 2
    low_bound = bound_box_mid - max_box_mid
    up_bound = bound_box_mid + max_box_mid

    low_bound = np.ceil(low_bound)
    up_bound = np.ceil(up_bound)
    low_bound_flag = (low_bound >= 0)
    up_bound_flag = (up_bound <= shape)

    if low_bound_flag[0] == False and up_bound_flag[0] == True:
        low_bound[0] = 0
        up_bound[0] = max_box[0]
    elif low_bound_flag[0] == True and up_bound_flag[0] == False:
        low_bound[0] = shape[0] - max_box[0]
        up_bound[0] = shape[0]
    # else:
    #        raise ValueError('low_bound_flag[0] is %s and up_bound_flag[0] is %s' %(low_bound_flag[0],up_bound_flag[0]))

    if low_bound_flag[1] == False and up_bound_flag[1] == True:
        low_bound[1] = 0
        up_bound[1] = max_box[1]
    elif low_bound_flag[1] == True and up_bound_flag[1] == False:
        low_bound[1] = shape[1] - max_box[1]
        up_bound[1] = shape[1]
    # else:
    #        raise ValueError('low_bound_flag[1] is %s and up_bound_flag[1] is %s' %(low_bound_flag[1],up_bound_flag[1]))

    if low_bound_flag[2] == False and up_bound_flag[2] == True:
        low_bound[2] = 0
        up_bound[2] = max_box[2]
    elif low_bound_flag[2] == True and up_bound_flag[2] == False:
        low_bound[2] = shape[2] - max_box[2]
        up_bound[2] = shape[2]
    # else:
    #        raise ValueError('low_bound_flag[2] is %s and up_bound_flag[2] is %s' %(low_bound_flag[2],up_bound_flag[2]))
    return low_bound, up_bound


def segment_crop_dicm(dicm_data, bound_box, max_box, maskers):
    '''
    :param dicm_data: 3D数据，每个患者的所有断层图片
    :param bound_box: 3D数据的癌症区域边界
    :param max_box:   最大边界
    :return:
    '''
    shape = np.array(dicm_data.shape, np.int16)
    bound_box_mid = np.mean(bound_box, 1)
    max_box_mid = max_box / 2
    low_bound = bound_box_mid - max_box_mid
    up_bound = bound_box_mid + max_box_mid

    low_bound = np.ceil(low_bound)
    up_bound = np.ceil(up_bound)

    if dicm_data.shape != maskers.shape:
        print('dcim_data shape is {}'.format(dicm_data.shape))
        print('maskers shape is {}'.format(maskers.shape))
        raise ValueError('dicm_data and maskers have not same shape .')

    maskers[maskers > 0] = 1
    dicm_array = dicm_data
    dicm_masker = maskers

    if np.all(low_bound >= 0) and np.all(up_bound <= shape):
        crop_dicm = dicm_array[int(low_bound[0]):int(up_bound[0]),
                    int(low_bound[1]):int(up_bound[1]),
                    int(low_bound[2]):int(up_bound[2])]

        crop_masker = dicm_masker[int(low_bound[0]):int(up_bound[0]),
                      int(low_bound[1]):int(up_bound[1]),
                      int(low_bound[2]):int(up_bound[2])]
    else:
        low_bound, up_bound = _adjust_to_mid(bound_box, max_box, shape)
        crop_dicm = dicm_array[int(low_bound[0]):int(up_bound[0]),
                    int(low_bound[1]):int(up_bound[1]),
                    int(low_bound[2]):int(up_bound[2])]

        crop_masker = dicm_masker[int(low_bound[0]):int(up_bound[0]),
                      int(low_bound[1]):int(up_bound[1]),
                      int(low_bound[2]):int(up_bound[2])]

    return crop_dicm, crop_masker


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def data_and_masker_to_tfrecord(path_lists, save_tfrecord_path, tfrecord_filename,
                                max_box, jilu_path='./julu_train.csv'):
    '''
    Save data into TFRecord
    param:
        path_list:每个病人的数据路径,类型list
    '''

    filename_path = os.path.join(save_tfrecord_path, tfrecord_filename)
    print(filename_path)

    if not os.path.isdir(save_tfrecord_path):
        os.makedirs(save_tfrecord_path)
    else:
        if os.path.isfile(filename_path):
            print("%s exists" % filename_path)
            return
    try:
        print("Converting data into %s ..." % filename_path)
        writer = tf.python_io.TFRecordWriter(filename_path)
    except Exception as e:
        print(e)
        return

    jilu = []
    for path in path_lists:
        try:

            data, label, bounding_box = img_pro.read_pkl(path)
            maskers = img_pro.read_masker_data(path)

            crop_data, crop_masker = segment_crop_dicm(data, bounding_box, max_box, maskers)

            crop_data = np.array(crop_data, np.float32)
            crop_masker = np.array(crop_masker, np.float32)

            print('crop_data shape is {}'.format(crop_data.shape))
            print('crop_masker shape is {}'.format(crop_masker.shape))

            temp = []
            j = 0
            for num, each_masker in enumerate(crop_masker):
                if np.count_nonzero(each_masker) > 0:
                    j += 1
                    dicm_raw = crop_data[num].tobytes()
                    dicm_masker = each_masker.tobytes()
                    feature = {'dicm_raw': _bytes_feature(dicm_raw),
                               'dicm_masker': _bytes_feature(dicm_masker)}
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
            temp.append(path)
            temp.append(j)
            temp.append(num + 1)
            jilu.append(temp)
            print(temp)

        except Exception as e:
            print(e)

    if jilu_path is not None:
        jilu = pd.DataFrame(jilu)
        jilu.to_csv(jilu_path)

    writer.close()


def data_and_masker_not_crop_to_tfrecord(path_lists, save_tfrecord_path, tfrecord_filename,
                                         jilu_path='./julu_train.csv'):

    '''
    Save data into TFRecord
    param:
        path_list:每个病人的数据路径,类型list
    '''

    filename_path = os.path.join(save_tfrecord_path, tfrecord_filename)
    print(filename_path)

    if not os.path.isdir(save_tfrecord_path):
        os.makedirs(save_tfrecord_path)
    else:
        if os.path.isfile(filename_path):
            print("%s exists" % filename_path)
            return
    try:
        print("Converting data into %s ..." % filename_path)
        writer = tf.python_io.TFRecordWriter(filename_path)
    except Exception as e:
        print(e)
        return

    jilu = []
    for path in path_lists:
        try:

            data, label, bounding_box = img_pro.read_pkl(path)
            maskers = img_pro.read_masker_data(path)
            maskers[maskers > 0] = 1
            # crop_data,crop_masker = segment_crop_dicm(data, bounding_box, max_box, maskers)

            crop_data = np.array(data, np.float32)
            crop_masker = np.array(maskers, np.float32)

            if crop_data.shape != crop_masker.shape:
                print('dcim_data shape is {}'.format(crop_data.shape))
                print('maskers shape is {}'.format(crop_masker.shape))
                raise ValueError('dicm_data and maskers have not same shape .')

            print('crop_data shape is {}'.format(crop_data.shape))
            print('crop_masker shape is {}'.format(crop_masker.shape))

            temp = []
            j = 0
            for num, each_masker in enumerate(crop_masker):
                if np.count_nonzero(each_masker) > 0:
                    j += 1
                    dicm_raw = crop_data[num].tobytes()
                    dicm_masker = each_masker.tobytes()
                    feature = {'dicm_raw': _bytes_feature(dicm_raw),
                               'dicm_masker': _bytes_feature(dicm_masker)}
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

            temp.append(path)
            temp.append(j)
            temp.append(num + 1)
            jilu.append(temp)
            print(temp)

        except Exception as e:
            print(e)

    if jilu_path is not None:
        jilu = pd.DataFrame(jilu)
        jilu.to_csv(jilu_path)

    writer.close()


def data_and_masker_decode(filename, d2_max_box):
    '''
    读取TFrecode文件里的数据并解码
    param:
        filename:文件路径
    return:
        ...
    '''
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = {'dicm_raw': tf.FixedLenFeature([], tf.string),
                'dicm_masker': tf.FixedLenFeature([], tf.string)}

    features = tf.parse_single_example(serialized_example, features=features)

    dicm_masker_raw = features['dicm_masker']
    dicm_masker_raw = tf.decode_raw(dicm_masker_raw, tf.float32)
    dicm_masker = tf.reshape(dicm_masker_raw, d2_max_box)

    dicm_raw = features['dicm_raw']
    dicm_raw = tf.decode_raw(dicm_raw, tf.float32)
    dicm_array = tf.reshape(dicm_raw, d2_max_box)
    return dicm_array, dicm_masker


def distored_inputs(filename, d2_max_box, batch_size, input_size, random_scale, random_mirror,standardiztion,  ignore_label):

    img, label = data_and_masker_decode(filename, d2_max_box)

    img = tf.expand_dims(img, 2)
    label = tf.expand_dims(label, 2)

    if input_size is not None:
        h, w = input_size
        # Randomly scale the images and labels.
        if random_scale:
            img, label = dag.image_scaling(img, label)

        # Randomly mirror the images and labels.
        if random_mirror:
            img, label = dag.image_mirroring(img, label)

        # Randomly crops the images and labels.
        img, label = dag.random_crop_and_pad_image_and_labels(img, label, h, w, ignore_label)

        if standardiztion is not None:
            img = tf.image.per_image_standardization(img)
            # 数据组合成batch_size
        img_batch, label_batch = tf.train.batch([img, label],
                                                batch_size=batch_size,
                                                capacity=2000,
                                                min_after_dequeue=1000,
                                                num_threads=10)
        return img_batch, label_batch

    else:
        if standardiztion is not None:
            img = tf.image.per_image_standardization(img)
        # 数据组合成batch_size
        img_batch, label_batch = tf.train.batch([img, label],
                                                        batch_size=batch_size,
                                                        capacity=2000,
                                                        min_after_dequeue=1000,
                                                        num_threads=10)
        return img_batch, label_batch


def distored_inputs2(filename,
                     d2_max_box,
                     batch_size,
                     resize_sape=None,
                     samplewise_norm=None,
                     maxmin_nor=None,
                     rotation=None):

    def samplewiseNorm(image):
        return tl.prepro.samplewise_norm(image,
                                  samplewise_center=True,
                                  samplewise_std_normalization=True)

    def maxminNorm(image, max, min):
        return MaxMinNormalization(image, Max=max, Min=min)

    def random_rotate_image_func(image, angle):
        # 旋转角度范围
        image = tl.prepro.rotation(image, rg=angle, is_random=False)
        return image

    img, label = data_and_masker_decode(filename, d2_max_box)
    img = tf.expand_dims(img, 2)
    label = tf.expand_dims(label, 2)

    if samplewise_norm is not None:
        img = tf.py_func(samplewiseNorm, [img], tf.float32)

    if maxmin_nor is not None:
        img = tf.py_func(maxminNorm, [img, 1000, 0], tf.float32)

    if rotation is not None:
        angle = np.random.uniform(low=-30, high=30)
        img = tf.py_func(random_rotate_image_func, [img, angle], tf.float32)
        label = tf.py_func(random_rotate_image_func, [label, angle], tf.float32)

    if resize_sape is not None:
        img = tf.image.resize_images(img, resize_sape)
        label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), resize_sape)
        label = tf.squeeze(label, squeeze_dims=[0])

    if resize_sape is not None:
        inputs_shape = [resize_sape+[1], [resize_sape+[1]]]
    else:
        inputs_shape = [[d2_max_box+[1]], [d2_max_box+[10]]]
    # 数据组合成batch_size
    img_batch, label_batch = tf.train.batch([img, label],
                                            batch_size=batch_size,
                                            capacity=2000,
                                            num_threads=10,
                                            shapes=inputs_shape)
    return img_batch, label_batch

def data_prosess(path_lists, positive_path, negative_path, positive_dir, negative_dir, jilu_path='./julu_train.csv'):
    '''
    :param path_lists:      从原图生成的pkl数据
    :param positive_path:   保存正样本数据的路径，数据都进行了归一化
    :param negative_path:   保存负样本的数
    :param positive_dir:    正样本统计路径 txt文件
    :param negative_dir:
    :param jilu_path:       记录样本情况路径
    :return:
    '''

    jilu = []
    positive_file_mapping = []
    negative_file_mapping = []
    for path in path_lists:
        try:
            #读取pkl里的数据
            data, label, bounding_box = img_pro.read_pkl(path)
            maskers = img_pro.read_masker_data(path)
            maskers[maskers > 0] = 1

            #转换数据类型为float32
            dicm_data = np.array(data, np.float32)
            dicm_masker = np.array(maskers, np.uint8)

            #如果原始数据和masker大小不一样，抛出异常
            if dicm_data.shape != dicm_masker.shape:
                print('dcim_data shape is {}'.format(dicm_data.shape))
                print('maskers shape is {}'.format(dicm_masker.shape))
                raise ValueError('dicm_data and maskers have not same shape .')

            print('dicm_data shape is {}'.format(dicm_data.shape))
            print('dicm_masker shape is {}'.format(dicm_masker.shape))

            temp = []
            positive_samples_len = 0
            negative_samples_len = 0
            for num, each_masker in enumerate(dicm_masker):
                if np.count_nonzero(each_masker) > 0:
                    positive_samples_len += 1
                    norma = (dicm_data[num] - dicm_data[num].min()) / (dicm_data[num].max() - dicm_data[num].min())
                    positive_data_file = path.replace('/', '_')+'num'+'_raw.jpg'
                    positive_masker_file = positive_data_file.replace('_raw.jpg','_raw.png')
                    io.imsave(os.path.join(positive_path, positive_data_file), norma)
                    io.imsave(os.path.join(positive_path, positive_masker_file), each_masker)
                    positive_file_mapping.append((positive_data_file, positive_masker_file))
                else:
                    negative_samples_len += 1
                    norma = (dicm_data[num] - dicm_data[num].min()) / (dicm_data[num].max() - dicm_data[num].min())
                    negative_data_file = path.replace('/', '_') + '_' + str(num) + '_raw.jpg'
                    negative_masker_file = negative_data_file.replace('_raw.jpg', '_raw.png')
                    io.imsave(os.path.join(negative_path, negative_data_file), norma)
                    io.imsave(os.path.join(negative_path, negative_masker_file), each_masker)
                    negative_file_mapping.append((negative_data_file, negative_masker_file))

            temp.append(path)
            temp.append(positive_samples_len)
            temp.append(negative_samples_len)
            jilu.append(temp)
            print(temp)

        except Exception as e:
            print(e)

    if jilu_path is not None:
        jilu = pd.DataFrame(jilu)
        jilu.to_csv(jilu_path)

    with open(os.path.join(positive_dir, 'train_positive.txt'), 'w') as f:
        for image_file, mask_file in positive_file_mapping:
            f.write('{} {}\n'.format(image_file, mask_file))

    with open(os.path.join(negative_dir, 'train_negative.txt'), 'w') as f:
        for image_file, mask_file in negative_file_mapping:
            f.write('{} {}\n'.format(image_file, mask_file))


def generate_data_masker_tfrecord(path_lists, save_tfrecord_path, tfrecord_filename,
                                  jilu_path='./julu_train.csv', save_path=None, lenn=-3,default_shape=(512,512)):

    #tfrecord文件名字及保存的路径
    filename_path = os.path.join(save_tfrecord_path, tfrecord_filename)
    print(filename_path)

    if not os.path.isdir(save_tfrecord_path):
        os.makedirs(save_tfrecord_path)
    else:
        if os.path.isfile(filename_path):
            print("%s exists" % filename_path)
            return
    try:
        print("Converting data into %s ..." % filename_path)
        writer = tf.python_io.TFRecordWriter(filename_path)
    except Exception as e:
        print(e)
        return
    h, w = default_shape
    jilu = []
    for path in path_lists:
        try:
            data, label, bounding_box = img_pro.read_pkl(path)
            maskers = img_pro.read_masker_data(path)
            maskers[maskers > 0] = 1
            # crop_data,crop_masker = segment_crop_dicm(data, bounding_box, max_box, maskers)

            dicm_data = np.array(data, np.float32)
            dicm_masker = np.array(maskers, np.float32)
            temp = []
            temp.append(dicm_data.shape)
            temp.append(dicm_masker.shape)

            if dicm_data.shape[1] != h or dicm_data.shape[2] != w:
                dicm_data = [cv2.resize(dat, default_shape) for dat in dicm_data]
                dicm_masker = [cv2.resize(mask, default_shape) for mask in dicm_masker]
                dicm_data = np.stack(dicm_data)
                dicm_masker = np.stack(dicm_masker)
                print('dicm_data shape is {}'.format(dicm_data.shape))
                print('dicm_masker shape is {}'.format(dicm_masker.shape))

            if dicm_data.shape != dicm_masker.shape:
                print('dcim_data shape is {}'.format(dicm_data.shape))
                print('maskers shape is {}'.format(dicm_masker.shape))
                raise ValueError('dicm_data and maskers have not same shape .')

            print('dicm_data shape is {}'.format(dicm_data.shape))
            print('dicm_masker shape is {}'.format(dicm_masker.shape))
            positive_samples_len = 0
            positive_samples = []
            negative_samples_len = 0
            negative_samples = []

            for num, each_masker in enumerate(dicm_masker):
                if np.count_nonzero(each_masker) > 0:
                    positive_samples_len += 1
                    positive_samples.append((dicm_data[num],each_masker,num))
                else:
                    negative_samples_len += 1
                    negative_samples.append((dicm_data[num],each_masker,num))

            lenth = min(negative_samples_len, positive_samples_len)
            #把负样本打乱，便于随机抽样
            random.shuffle(negative_samples)
            train_samples = positive_samples + negative_samples[:lenth]  #通过此处可以改变正负样本的比例
            random.shuffle(train_samples)

            for raw, masker, numer in train_samples:
                dicm_raw = raw.tobytes()
                dicm_masker = masker.tobytes()
                feature = {'dicm_raw': _bytes_feature(dicm_raw),
                           'dicm_masker': _bytes_feature(dicm_masker)}
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

            if save_path is not None:
                save_data(train_samples, save_path, path, lenn)

            temp.append(path)
            temp.append(positive_samples_len)
            temp.append(negative_samples_len)
            temp.append(len(train_samples))
            jilu.append(temp)
            print(temp)

        except Exception as e:
            print(e)

    if jilu_path is not None:
        jilu = pd.DataFrame(jilu)
        jilu.to_csv(jilu_path)

    writer.close()

def save_data(dicm_data, save_path, path,len):

    '''
    dicm_data,为了检查扣图正确
    '''

    path_list = path.split('/')

    temp_path = save_path
    for pat in path_list[len:]:
        temp_path = os.path.join(temp_path,pat)

    if not os.path.isdir(temp_path):
        os.makedirs(temp_path)

    for dicm, masker, num in dicm_data:
        image_path = os.path.join(temp_path, str(num)+'_dicm.bmp')
        masker_path = os.path.join(temp_path,str(num)+'_masker.bmp')
        dicm_array = MaxMinNormalization(dicm) * 255
        dicm_masker = masker*255
        cv2.imwrite(image_path, dicm_array)
        cv2.imwrite(masker_path, dicm_masker)

def MaxMinNormalization(x, Max=None, Min=None):

    '''
    (0,1)标准化
    param:
        x:输入数据，类型为numpy.array类型
        Max：x的最大值，或设定的值
        Min:x的最小值，或设定的值
    '''
    if Max == None:
        Max = x.max()
    if Min == None:
        Min = x.min()

    x = (x - Min) / (Max - Min)
    x[x > 1] = 1
    return x

if __name__ == '__main__':
    # _,seconde_path,third_path=read_files.get_files_path('/data/b/wanbbuanbyuan/rectal_data')
    max_box = np.array([20, 224, 224])
    d2_max_box = np.array([256, 256])
    # data_and_masker_to_tfrecord(['/data/b/wanbbuanbyuan/rectal_data/5/PA39/DCE'],'./','test.tfrecord',max_box)

    with tf.device('/cpu:0'):
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        # 读取数据
        x_train, y_train = data_and_masker_decode(r'D:\after_process_data\liver\tfrecord\dwi_train_not_crop.tfrecord',
                                                  d2_max_box)
        x_t = sess.run(x_train)
        # 初始化线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        coord.request_stop()
        coord.join(threads)
        sess.close()


