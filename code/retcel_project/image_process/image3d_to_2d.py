# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 15:21:43 2017

@author: admin
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os 
import pandas as pd
from sklearn.cross_validation import train_test_split
try:
    from retcel_project.image_process import image_process as img_pro
except :
    import image_process as img_pro
    
try:
    from retcel_project.read_file_process import read_files
except:
    from read_file_process import read_files

    
#生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    

def data3d_to_2d_to_tfrecord(path_lists,save_tfrecord_path,tfrecord_filename,max_box,jilu_path='./julu_train.csv',masker_flag=True):
    
    '''
    Save data into TFRecord
    param:
        path_list:每个病人的数据路径,类型list
        labels:   和path_list对应的病人的标签，类型list
        str_list:["BMP","BMP_Marker","DCM"]
        filename:生成的TFRecord文件保存路径
        bound_box_path:要保存的bounding_box路径，如果给了路径就把它保存到给的路径下，没有就不保存
    '''

    filename_path = os.path.join(save_tfrecord_path,tfrecord_filename)
    print(filename_path)

    if not os.path.isdir(save_tfrecord_path):
        os.makedirs(save_tfrecord_path)
    else:
        if os.path.isfile(filename_path):
            print("%s exists" %filename_path)
            return
    try:
        print("Converting data into %s ..." % filename_path)
        writer = tf.python_io.TFRecordWriter(filename_path)
    except Exception as e:
        print(e)
        return

    jilu=[]
    for path in path_lists:
        try:
            data, label, bounding_box = img_pro.read_pkl(path)
            if label >=4:
                label=1
            else:
                label=0
            if masker_flag is False:
                crop_data, _ = img_pro.crop_dicm(data, bounding_box, max_box)
            else:
                maskers = img_pro.read_masker_data(path)
                crop_data, _ = img_pro.crop_dicm(data, bounding_box, max_box, maskers)
            crop_data = np.array(crop_data, np.float32)
            print(crop_data.shape)
            temp=[]
            j=0
            for num,each_data in enumerate(crop_data):
                if np.count_nonzero(each_data)>0:
                    j +=1
                    dicm_raw = each_data.tobytes()
                    feature = {'dicm_raw': _bytes_feature(dicm_raw),
                               'label': _int64_feature(label)}
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
            temp.append(label)
            temp.append(path)                    
            temp.append(j)
            temp.append(num+1)
            jilu.append(temp)
            print(temp)
            
        except Exception as e:
            print(e)

    if jilu_path is not None:
        jilu=pd.DataFrame(jilu)
        jilu.to_csv(jilu_path)
    writer.close()

    
def data2d_and_label_decode(filename,d2_max_box):
    
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
                'label': tf.FixedLenFeature([], tf.int64)}

    features = tf.parse_single_example(serialized_example, features=features)

    label = tf.cast(features['label'], tf.int32)

    dicm_raw = features['dicm_raw']
    dicm_raw = tf.decode_raw(dicm_raw, tf.float32)
    dicm_array = tf.reshape(dicm_raw, d2_max_box)
    return dicm_array, label


if __name__=='__main__':
    max_box=np.array([20,100,100])
    
    _,seconde_path,third_path=read_files.get_files_path('/data/b/wanbbuanbyuan/rectal_data')
    
    # train_ss,test_ss,train_labels,test_labels=train_test_split(third_path,[1]*410,test_size=0.05,random_state=0)
    
    
    
    #data3d_to_2d_to_tfrecord(train_ss, './data/tfrecord/','dce2d_train_tfrecord.tfrecord',max_box,'./jilu_train.csv')
    #data3d_to_2d_to_tfrecord(test_ss, './data/tfrecord/','dce2d_test_tfrecord.tfrecord',max_box,'./jilu_train.csv')
    
    
    #data,label,bound_box=img_pro.read_pkl(third_path[10])
    #crop_data,_=img_pro.crop_dicm(data,bound_box,max_box)
    #img_pro.save_maskers(third_path[10],crop_data,'data','datt')
    #plt.imshow(crop_data[10],plt.cm.gray)
