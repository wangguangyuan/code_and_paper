from retcel_project.image_process import segment_image_process as img_seg
from retcel_project.read_file_process import read_files
_, seconde_path, third_path = read_files.get_files_path('/data/b/wangguangyuan/rectal_data/rectel_pkl_data')
path_lists, labels = read_files.get_path_type(third_path, [1]*len(third_path))

img_seg.data_prosess(path_lists,
                     '/data/b/wangguangyuan/rectal_segment/DCE/positive_data',
                     '/data/b/wangguangyuan/rectal_segment/DCE/negative_data',
                     '/data/b/wangguangyuan/rectal_segment/DCE/train_text',
                     '/data/b/wangguangyuan/rectal_segment/DCE/train_text',
                     '/data/b/wangguangyuan/rectal_segment/DCE/jilu')