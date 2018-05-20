from retcel_project.image_process import image_process as img_pro
from retcel_project.read_file_process import read_files
from sklearn.cross_validation import train_test_split
import numpy as np

# 分析bounding_box 程序
# import os
# import numpy as np
# import pandas as pd
#
# def mid_3d(box_all):
#     measure_all = np.zeros((len(box_all), 2, 3))
#     for i in range(len(box_all)):
#         measure_all[i, 0, :] = box_all[i, :, 1] - box_all[i, :, 0] + 1
#         measure_all[i, 1, :] = np.mean(box_all[i, :, :], axis=1)
#     return measure_all
#
# bound_path =os.path.join('D:/','dce_bounding_box.csv')
# bounding_box_all = np.loadtxt(bound_path, delimiter=',').reshape([-1, 3, 2])
# measure_all = mid_3d(bounding_box_all)
#
# dat=[measure_all[i][0] for i in range(len(measure_all))]
# df=pd.DataFrame(dat)

max_box = np.array([20,100,100])
_, seconde_path, third_path = read_files.get_files_path('/data/b/wangguangyuan/rectal_data/rectel_pkl_data')

train_ss, test_ss, train_labels, test_labels = train_test_split(third_path,[1]*410,test_size=0.15,random_state=0)


#生成有背景的tfrecord
# img_pro.data_and_label_to_tfrecord(train_ss,
#                                    '/data/b/wangguangyuan/rectal_data/tfrecord/jiabeijing/DCE',        #tfrecord 路径
#                                    'dce_train_tfrecord.tfrecord',                                      #文件名
#                                    max_box=max_box,                                                    #切割大小
#                                    augment_tfrecord_filename=None,                                     #是否进行数据增强
#                                    masker_flag=False,                                                  #是否加背景 Fasle加背景
#                                    dicm_show_path=True,                                                #数据保存，用于查看提取的数据是否正确
#                                    show_str1='/data/b/wangguangyuan/rectal_data/rectel_pkl_data',
#                                    show_str2='/data/b/wangguangyuan/rectal_data/tfrecord/jiabeijing/show_data',
#                                    jilu_path='/data/b/wangguangyuan/rectal_data/tfrecord/jiabeijing/jilu/dce_train_jilu.csv'
#                                    )
#
# img_pro.data_and_label_to_tfrecord(test_ss,
#                                    '/data/b/wangguangyuan/rectal_data/tfrecord/jiabeijing/DCE',
#                                    'dce_test_tfrecord.tfrecord',
#                                    max_box=max_box,
#                                    augment_tfrecord_filename=None,
#                                    masker_flag=False,
#                                    dicm_show_path=False,
#                                    # show_str1='/data/b/wangguangyuan/rectal_data/rectel_pkl_data',
#                                    # show_str2='/data/b/wangguangyuan/rectal_data/tfrecord/show_data',
#                                    jilu_path='/data/b/wangguangyuan/rectal_data/tfrecord/jiabeijing/jilu/dce_test_jilu.csv'
#                                    )


#生成无背景的tfrecord
img_pro.data_and_label_to_tfrecord(train_ss,
                                   '/data/b/wangguangyuan/rectal_data/tfrecord/wubeijing/DCE',        #tfrecord 路径
                                   'dce_train_tfrecord.tfrecord',                                      #文件名
                                   max_box=max_box,                                                    #切割大小
                                   augment_tfrecord_filename=None,                                     #是否进行数据增强
                                   masker_flag=True,                                                  #是否加背景 Fasle加背景，True为不加背景
                                   dicm_show_path=True,                                                #数据保存，用于查看提取的数据是否正确
                                   show_str1='/data/b/wangguangyuan/rectal_data/rectel_pkl_data',
                                   show_str2='/data/b/wangguangyuan/rectal_data/tfrecord/wubeijing/show_data',
                                   jilu_path='/data/b/wangguangyuan/rectal_data/tfrecord/wubeijing/jilu/dce_train_jilu.csv'
                                   )


img_pro.data_and_label_to_tfrecord(test_ss,
                                   '/data/b/wangguangyuan/rectal_data/tfrecord/wubeijing/DCE',
                                   'dce_test_tfrecord.tfrecord',
                                   max_box=max_box,
                                   augment_tfrecord_filename=None,
                                   masker_flag=True,
                                   dicm_show_path=False,
                                   # show_str1='/data/b/wangguangyuan/rectal_data/rectel_pkl_data',
                                   # show_str2='/data/b/wangguangyuan/rectal_data/tfrecord/show_data',
                                   jilu_path='/data/b/wangguangyuan/rectal_data/tfrecord/wubeijing/jilu/dce_test_jilu.csv'
                                   )
