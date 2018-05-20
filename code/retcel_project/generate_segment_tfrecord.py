# -*- coding: utf-8 -*-
try:
    from retcel_project.image_process import segment_image_process as img_seg
except:
    from image_process import segment_image_process as img_seg
try:
    from retcel_project.read_file_process import read_files
except:
    from read_file_process import read_files

_, seconde_path, third_path = read_files.get_files_path('/data/b/wangguangyuan/rectal_data/rectel_pkl_data')
path_lists, labels = read_files.get_path_type(third_path, [1]*len(third_path), types='T2_FS')

img_seg.generate_data_masker_tfrecord(path_lists,
                                      '/data/b/wangguangyuan/rectal_segment/DCE/tfrecord',
                                      'ts_fs_train.tfrecord',
                                      '/data/b/wangguangyuan/rectal_segment/DCE/tfrecord/jilu/t2_fs_jilu_train.csv',
                                      '/data/b/wangguangyuan/rectal_segment/DCE/tfrecord/show_data_t2',
                                      )