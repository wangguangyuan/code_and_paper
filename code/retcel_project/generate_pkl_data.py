try:
    from retcel_project.read_file_process import read_files
except:
    from read_file_process import read_files
try:
    from retcel_project.image_process import image_process as img_pro
except:
    from image_process import image_process as img_pro

label_path = '/data/b/wangguangyuan/rectal_data/rectel_label/data.xls'
data_path = '/data/g/wangguangyuan/rectal_data'

data = read_files.read_xls(label_path)
path_lists, label_lists = read_files.get_data_path(data,data_path,'DCE')

img_pro.process_data(path_lists,label_lists,
                     '/g/wangguangyuan/rectal_data/',
                     '/b/wangguangyuan/rectal_data/rectel_pkl_one_period/',
                     str_list=["BMP", "BMP_Marker", "DCM"],
                     bound_box_path='/data/b/wangguangyuan/rectal_data/bound_box_path',
                     bound_box_filename='dce_bounding_box.csv',
                     jilu_path_and_file='/data/b/wangguangyuan/rectal_data/jilu_file/dce_jilu.csv',
                     check_masker_flag=True,
                     mask_str1='rectel_pkl_one_period',
                     masker_str2='dce_masker'
                     )

