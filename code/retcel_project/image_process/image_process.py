# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 18:03:32 2017
@author: wgy
"""
import tensorflow as tf
from skimage import measure
import matplotlib.pyplot as plt
import skimage.morphology as sm
import numpy as np
import cv2
import dicom
import os
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pickle
import pandas as pd
from sklearn.cross_validation import train_test_split

try:
    from retcel_project.read_file_process import read_files
except:
    from read_file_process import read_files

def mode(l):
    # 统计list中各个数值出现的次数
    count_dict = {}
    for i in l:
        if i in count_dict:
            count_dict[i] += 1
        else:
            count_dict[i] = 1
            # 求出现次数的最大值
    max_appear = 0
    for v in count_dict.values():
        if v > max_appear:
            max_appear = v
    if max_appear == 1:
        return
    mode_list = []
    for k, v in count_dict.items():
        if v == max_appear:
            mode_list.append(k)
    return mode_list, count_dict


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

    x = (x - Min) / (Max - Min);
    return x


def fillholse(im_th):
    '''
    空洞填充
    param:
        im_th:二值图像
    return:
        im_out:填充好的图像
        success:填充是否成功
    '''

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    retval, image, mask, rect = cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    success = ~np.all(im_out == im_th)
    return im_out, success


def read_one_patient_image(path, str_list):

    '''
    提取每个病人的扫描图像里的癌变区域
    param:
        path:每个病人的每种扫描方式路径
        str_list:["BMP","BMP_Marker","DCM"]
    return: bmp_slices,bmp_masker_slices,dicm_slices
                类型为array of list
    '''

    # 路径
    image_bmp_path = os.path.join(path, str_list[0])
    image_bmp_masker_path = os.path.join(path, str_list[1])
    image_dicm_path = os.path.join(path, str_list[2])
    # 提取该路径下的所有文件
    image_bmp_list = os.listdir(image_bmp_path)
    image_bmp_masker_list = os.listdir(image_bmp_masker_path)
    image_dicm_list = os.listdir(image_dicm_path)
    # 排序
    image_bmp_list = sorted(image_bmp_list)
    image_bmp_masker_list = sorted(image_bmp_masker_list)
    image_dicm_list = sorted(image_dicm_list)

    len_bmp = len(image_bmp_list)
    len_bmp_masker = len(image_bmp_masker_list)
    len_dicm = len(image_dicm_list)

    # 判断三个文件下的图片数量是否相等
    if len_bmp == len_bmp_masker == len_dicm:
        bmp_slices = [cv2.imread(os.path.join(image_bmp_path, s)) for s in image_bmp_list]
        bmp_masker_slices = [cv2.imread(os.path.join(image_bmp_masker_path, s)) for s in image_bmp_masker_list]
        dicm_slices = [dicom.read_file(os.path.join(image_dicm_path, s)).pixel_array for s in image_dicm_list]
    else:
        raise TypeError('三个文件夹下的图片数量不同')

    return bmp_slices, bmp_masker_slices, dicm_slices

def find_one_period(path, str_list):
    '''
        提取每个病人的扫描图像里的癌变区域
        param:
            path:每个病人的每种扫描方式路径
            str_list:["BMP","BMP_Marker","DCM"]
        return: bmp_slices,bmp_masker_slices,dicm_slices
                    类型为array of list
        '''

    # 路径
    image_bmp_path = os.path.join(path, str_list[0])
    image_bmp_masker_path = os.path.join(path, str_list[1])
    image_dicm_path = os.path.join(path, str_list[2])
    # 提取该路径下的所有文件
    image_bmp_list = os.listdir(image_bmp_path)
    image_bmp_masker_list = os.listdir(image_bmp_masker_path)
    image_dicm_list = os.listdir(image_dicm_path)

    if len(image_dicm_list) == 0 or len(image_bmp_list) == 0 or len(image_bmp_masker_list) == 0:
        raise ValueError(path, 'No file')

    # 排序
    image_bmp_list = sorted(image_bmp_list)
    image_bmp_masker_list = sorted(image_bmp_masker_list)
    image_dicm_list = sorted(image_dicm_list)

    len_bmp = len(image_bmp_list)
    len_bmp_masker = len(image_bmp_masker_list)
    len_dicm = len(image_dicm_list)

    image_bmp_files = [os.path.join(image_bmp_path, s) for s in image_bmp_list]
    image_bmp_masker_files = [os.path.join(image_bmp_masker_path, s) for s in image_bmp_masker_list]
    image_dicm_files = [os.path.join(image_dicm_path, s) for s in image_dicm_list]

    # 判断三个文件下的图片数量是否相等
    if len_bmp == len_bmp_masker == len_dicm:
        paths = [pat for pat in zip(image_dicm_files, image_bmp_masker_files, image_bmp_files)]
        paths.sort(key=lambda x: int(dicom.read_file(x[0]).InstanceNumber), reverse=False)
        # print(paths)
        slice_lactin = [dicom.read_file(po[0]).SliceLocation for po in paths]
        # print(slice_lactin)

        # print(len(slice_lactin))

        loctions = [round(float(slice_lactin[i]) - float(slice_lactin[i + 1])) for i in range(len(slice_lactin) - 1)]
        # print(loctions)
        zo, ls = mode(loctions)
        print('befor:', ls)

        for i in range(len(loctions)):
            if loctions[i] > 0 and loctions[i] < 15:
                loctions[i] = zo[0]
            elif loctions[i] < 0 and loctions[i] > -15:
                loctions[i] = zo[0]

        zo, ls = mode(loctions)
        print('after:', ls)
        # print(zo, ls)

        # bmp_slices = []
        # bmp_masker_slices = []
        # dicm_slices = []

        bmp_slices = [cv2.imread(bmp_slice) for _, _, bmp_slice in paths]
        bmp_masker_slices = [cv2.imread(bmp_masker_slice) for _, bmp_masker_slice, _ in paths]
        dicm_slices = [dicom.read_file(dicm_slice).pixel_array for dicm_slice, _, _ in paths]

        if len(ls) == 1:
            return bmp_slices, bmp_masker_slices, dicm_slices

        elif len(ls) == 2:
            #读文件

            # for dicm_slice, bmp_masker_slice, bmp_slice in paths:
            #     bmp_slices.append(cv2.imread(bmp_slice))
            #     bmp_masker_slices.append(cv2.imread(bmp_masker_slice))
            #     dicm_slices.append(dicom.read_file(dicm_slice).pixel_array)

            # 把list转换成array类型
            bmp_array = np.stack(bmp_slices)
            bmp_masker_array = np.stack(bmp_masker_slices)
            image_maskers = bmp_masker_array - bmp_array

            image_maskers[image_maskers > 0] = 255
            maskers_shape = image_maskers.shape

            for each_masker_num in range(maskers_shape[0]):
                if np.count_nonzero(image_maskers[each_masker_num]) > 0:
                    length = each_masker_num + 1
                    break

            # print(length)

            if len(ls) == 2:
                for k, value in ls.items():
                    if k != zo[0]:
                        period = value+1
                        # print(period)

            one_period_nums = np.int16(len_dicm / period)

            peri = length // one_period_nums
            # print(peri)

            # print(one_period_nums)
            start = peri*one_period_nums
            end = (peri+1)*one_period_nums
            print(start, end, zo, ls, one_period_nums, peri, period)
            return bmp_slices[start:end], bmp_masker_slices[start:end], dicm_slices[start:end]
        else:
            print('\n------------------------------------\n')
            print(path, len(bmp_slices), 'SliceLaction error \n')


def judge_no_empty(path, bmp_slices, bmp_masker_slices, dicm_slices):
    '''
    判断某个病人的某种扫描方式是否不为空
    '''
    #    path=os.path.split(path)[0]
    #    filedir=os.path.split(path)[1]
    if len(bmp_slices) == 0 or len(bmp_masker_slices) == 0 or len(dicm_slices) == 0:
        #        print('%s is a empty dir!' %filedir)
        return False
    else:
        return True


def extract_one_patient_roi(path, str_list, threshold=0, structural_elements=sm.square(5), types='BMP', channel=2):
    '''
    提取每个扫描图像有癌症的区域
    param:
        path:每个病人的每种扫描方式路径
        str_list:["BMP","BMP_Marker","DCM"]
        threshold:阈值设置
        structural_elements:膨胀的结构元素设置
        types: 'JPG' or 'BMP'
        channel: 1 for 'JPG',2 for 'BMP'
    return:
        dicm_array:读出来的每个病人的dicom序列，类型array shape=[z,x,y]
        maskers_out:癌症区域的masker 类型array, shape=[z,x,y]
        not_fill_image:记录没有填充上的图像信息
        after_dilation_not_fill_images:记录膨胀后仍然没有填充上的图像
    '''

    not_fill_image = []
    after_dilation_not_fill_images = []
    maskers_out = []

    # bmp_slices, bmp_masker_slices, dicm_slices = read_one_patient_image(path, str_list)
    bmp_slices, bmp_masker_slices, dicm_slices = find_one_period(path, str_list)
    # 如果文件夹为空抛出异常
    if not judge_no_empty(path, bmp_slices, bmp_masker_slices, dicm_slices):
        raise ValueError('%s is a empty dir!' % path)

    # 把list转换成array类型
    bmp_array = np.stack(bmp_slices)
    bmp_masker_array = np.stack(bmp_masker_slices)
    dicm_array = np.stack(dicm_slices)

    if types == 'BMP':
        image_maskers = bmp_masker_array - bmp_array
    else:
        image_maskers = bmp_masker_array.astype(np.int16) - bmp_array.astype(np.int16)
        image_maskers[image_maskers < threshold] = 0
        image_maskers = image_maskers.astype(np.uint8)

    image_maskers[image_maskers > 0] = 255
    maskers_shape = image_maskers.shape

    for each_masker_num in range(maskers_shape[0]):
        # 判断是否为零，不为零进行填充后保存，否则直接保存
        if np.count_nonzero(image_maskers[each_masker_num]) > 0:
            if image_maskers[each_masker_num].ndim == 2:
                masker, success = fillholse(image_maskers[each_masker_num])
                if ~success:
                    not_fill_image.append(each_masker_num + 1)
                    # 填充未成功，需要进行膨胀
                    image_dilation = sm.dilation(image_maskers[each_masker_num], structural_elements)

                    masker, success = fillholse(image_dilation)
                    if ~success:
                        after_dilation_not_fill_images.append(each_masker_num + 1)
                maskers_out.append(masker)
            elif image_maskers[each_masker_num].ndim == 3:
                masker, success = fillholse(image_maskers[each_masker_num][:, :, channel])
                if ~success:
                    not_fill_image.append(each_masker_num + 1)
                    # 填充未成功，需要进行膨胀
                    image_dilation = sm.dilation(image_maskers[each_masker_num][:, :, channel], structural_elements)
                    masker, success = fillholse(image_dilation)
                    if ~success:
                        after_dilation_not_fill_images.append(each_masker_num + 1)
                maskers_out.append(masker)
            else:
                raise TypeError("your input array must 2 or 3 nidm")
        else:

            if image_maskers[each_masker_num].ndim == 2:
                maskers_out.append(image_maskers[each_masker_num])
            elif image_maskers[each_masker_num].ndim == 3:
                maskers_out.append(image_maskers[each_masker_num][:, :, channel])
            else:
                raise TypeError("your input array must 2 or 3 nidm")

    maskers_out = np.stack(maskers_out)

    return dicm_array, maskers_out, not_fill_image, after_dilation_not_fill_images


def find_border(maskers):
    '''
    找到2d或3d图像的边界
    param:
        maskers:每个病人标画的区域标记,array,三维，[slices，row,col]
    return:
        bounding_box: 三维边界 array
    '''

    indexs = np.stack(np.nonzero(maskers))
    indexs_max = indexs.max(axis=1)
    indexs_min = indexs.min(axis=1)
    bounding_box = [[indexs_min[0], indexs_max[0]],
                    [indexs_min[1], indexs_max[1]],
                    [indexs_min[2], indexs_max[2]]]

    bounding_box = np.array(bounding_box, np.uint16)

    #    dicom_roi=dicoms[bounding_box[0,0]:bounding_box[0,1]+1,
    #                     bounding_box[1,0]:bounding_box[1,1]+1,
    #                     bounding_box[2,0]:bounding_box[2,1]+1]

    return bounding_box


def save_maskers(path, maskers, str1=None, str2=None):
    '''
    保存maskers,为了检查扣图正确
    '''

    if str1 !=None and str2 !=None:
        save_path = path.replace(str1, str2)
    else:
        save_path=path

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for num, each_masker in enumerate(maskers):
        if np.count_nonzero(each_masker) > 0:
            temp_path = os.path.join(save_path, ("%s.bmp" % num))
            cv2.imwrite(temp_path, each_masker)


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def process_data(path_lists, labels, str1, str2, str_list=["BMP", "BMP_Marker", "DCM"],
                 bound_box_path=None, bound_box_filename='bounding_box.csv', jilu_path_and_file=None, threshold=0,
                 structural_elements=sm.square(5), types='BMP', channel=2,
                 check_masker_flag=None,mask_str1=None,masker_str2=None):

    #记录数据
    bounding_box_all = []  # 所有path的原始Boxx
    not_fill_sucess = []  # 没有提取成功的文件记录

    for index, path in enumerate(path_lists):
        temp = []

        #要保存的数据路径
        save_path = path.replace(str1, str2)
        #如果路径替换不成功，跳出本次执行
        if save_path == path:
            print(path, str1, str2, '路径替换不成功')
            continue

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        data_path = os.path.join(save_path, 'data.pkl')
        masker_path = os.path.join(save_path, 'masker.pkl')
        label_path = os.path.join(save_path, 'label.pkl')
        bound_path = os.path.join(save_path, 'bound_box.pkl')

        # 如果文件都存在，跳过
        if os.path.isfile(data_path) and os.path.isfile(masker_path) \
                and os.path.isfile(label_path) and os.path.isfile(bound_path):
            continue

        print(data_path)

        try:
            dicm_array, maskers_out, a, b = extract_one_patient_roi(path, str_list, threshold,
                                                                    structural_elements, types, channel)
            if check_masker_flag is True:
                #查看数据是否正确
                save_maskers(save_path, maskers_out,mask_str1,masker_str2)

            bounding_box = find_border(maskers_out)
            bounding_box_all.append(bounding_box)

            with open(data_path, 'wb') as f:
                pickle.dump(dicm_array, f, pickle.HIGHEST_PROTOCOL)

            with open(masker_path, 'wb') as f:
                pickle.dump(maskers_out, f, pickle.HIGHEST_PROTOCOL)

            label = int(labels[index])
            with open(label_path, 'wb') as f:
                pickle.dump(label, f, pickle.HIGHEST_PROTOCOL)

            with open(bound_path, 'wb') as f:
                pickle.dump(bounding_box, f, pickle.HIGHEST_PROTOCOL)

            # shape = []
            # shape.append(dicm_array.shape[1])
            # shape.append(dicm_array.shape[2])

            print(dicm_array.shape)
            temp.append(path)
            temp.append(a)
            temp.append(b)
            temp.append(dicm_array.shape)
            if len(a) > 0 or len(b) > 0:
                print(temp)
            not_fill_sucess.append(temp)

            # 50次保存一次参数
            if index % 50 == 0:

                if bound_box_path != None:
                    if not os.path.isdir(bound_box_path):
                        os.makedirs(bound_box_path)

                    temp_box_all = np.stack(bounding_box_all)

                    # 将变换前后的box均写入csv
                    bounding_box_all_reshape = temp_box_all.reshape(-1, 6, 1)
                    np.savetxt(os.path.join(bound_box_path, bound_box_filename), bounding_box_all_reshape, fmt='%d',
                               delimiter=',')
                    print('Successfully saved bounding_box to %s' % bound_box_path)

        except Exception as e:
            print(path,e)

    bounding_box_all = np.stack(bounding_box_all)
    measure_all = mid_3d(bounding_box_all)
    max_box = find_max_box(measure_all)

    if bound_box_path != None:
        if not os.path.isdir(bound_box_path):
            os.makedirs(bound_box_path)

        # 将变换前后的box均写入csv
        bounding_box_all_reshape = bounding_box_all.reshape(-1, 6, 1)
        np.savetxt(os.path.join(bound_box_path, bound_box_filename), bounding_box_all_reshape, fmt='%d', delimiter=',')

        print('Successfully saved bounding_box to %s' % bound_box_path)
    if jilu_path_and_file is not None:
        not_fill_sucess = pd.DataFrame(not_fill_sucess)
        not_fill_sucess.to_csv(jilu_path_and_file)
    return max_box

def read_pkl(path):
    data_path = os.path.join(path, 'data.pkl')
    label_path = os.path.join(path, 'label.pkl')
    bound_path = os.path.join(path, 'bound_box.pkl')

    with open(data_path, 'rb') as f:
        dicm_data = pickle.load(f)
    with open(label_path, 'rb') as f:
        label = pickle.load(f)

    with open(bound_path, 'rb') as f:
        bound_box = pickle.load(f)

    return dicm_data, label, bound_box


def read_masker_data(path):
    masker = []
    masker_path = os.path.join(path, 'masker.pkl')
    with open(masker_path, 'rb') as f:
        masker = pickle.load(f)
    return masker


def read_max_box(path=r'../data/dce_bound', filename='bounding_box.csv'):
    bound_path = os.path.join(path, filename)
    bounding_box_all = np.loadtxt(bound_path, delimiter=',').reshape([-1, 3, 2])
    measure_all = mid_3d(bounding_box_all)
    max_box = find_max_box(measure_all)
    return max_box


def _fill_zeros(dicm_data, shape, bound_box, max_box, maskers=None):
    '''

    :param dicm_data: dicom数据
    :param shape:     dicom数据的大小
    :param bound_box: 癌症区域的边界[z,x,y]
    :param max_box:   需要抠出区域大小[z,x,y]
    :param maskers:
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

    low_bound = temp_mid - bound_mid
    up_bound = temp_mid + bound_mid

    low_bound = np.ceil(low_bound)
    up_bound = np.ceil(up_bound)

    temp_masker = np.zeros([dept, weight, height], np.uint16)

    if maskers is None:
        dicm_array = dicm_data[bound_box[0, 0]:bound_box[0, 1] + 1,
                               bound_box[1, 0]:bound_box[1, 1],
                               bound_box[2, 0]:bound_box[2, 1]]

        temp_masker[int(low_bound[0]):int(up_bound[0]) + 1,
                    int(low_bound[1]):int(up_bound[1]),
                    int(low_bound[2]):int(up_bound[2])] = dicm_array

    else:
        dicm_masker = dicm_data * maskers
        dicm_array = dicm_masker[bound_box[0, 0]:bound_box[0, 1] + 1,
                                 bound_box[1, 0]:bound_box[1, 1] + 1,
                                 bound_box[2, 0]:bound_box[2, 1] + 1]

        temp_masker[int(low_bound[0]):int(up_bound[0]) + 1,
                    int(low_bound[1]):int(up_bound[1]) + 1,
                    int(low_bound[2]):int(up_bound[2]) + 1] = dicm_array

    max_box_mid = max_box / 2
    crop_low_bound = temp_mid - max_box_mid
    crop_up_bound = temp_mid + max_box_mid

    crop_low_bound = np.ceil(crop_low_bound)
    crop_up_bound = np.ceil(crop_up_bound)

    temp_array = temp_masker[int(crop_low_bound[0]):int(crop_up_bound[0]),
                             int(crop_low_bound[1]):int(crop_up_bound[1]),
                             int(crop_low_bound[2]):int(crop_up_bound[2])]

    return temp_array


def crop_dicm(dicm_data, bound_box, max_box, maskers=None):

    '''
    :param dicm_data: dicom数据
    :param bound_box: 癌症区域的边界[z,x,y]
    :param max_box:   需要抠出区域大小[z,x,y]
    :param maskers:
    :return:
    '''

    shape = np.array(dicm_data.shape, np.int16)
    bound_box_mid = np.mean(bound_box, 1)
    max_box_mid = max_box / 2
    low_bound = bound_box_mid - max_box_mid
    up_bound = bound_box_mid + max_box_mid

    low_bound = np.ceil(low_bound)
    up_bound = np.ceil(up_bound)

    if maskers is None:
        if np.all(low_bound >= 0) and np.all(up_bound <= shape):
            dicm_masker = np.zeros(shape, np.uint16)

            dicm_masker[bound_box[0, 0]:bound_box[0, 1]+1,
                        bound_box[1, 0]:bound_box[1, 1],
                        bound_box[2, 0]:bound_box[2, 1]] = 1

            dicm_array = dicm_data * dicm_masker
            crop_dicm = dicm_array[int(low_bound[0]):int(up_bound[0]),
                                   int(low_bound[1]):int(up_bound[1]),
                                   int(low_bound[2]):int(up_bound[2])]
        else:
            crop_dicm = _fill_zeros(dicm_data, shape, bound_box, max_box)
    else:

        if np.all(low_bound >= 0) and np.all(up_bound <= shape):
            maskers[maskers > 0] = 1
            dicm_array = dicm_data * maskers
            crop_dicm = dicm_array[int(low_bound[0]):int(up_bound[0]),
                                   int(low_bound[1]):int(up_bound[1]),
                                   int(low_bound[2]):int(up_bound[2])]
        else:
            maskers[maskers > 0] = 1
            crop_dicm = _fill_zeros(dicm_data, shape, bound_box, max_box, maskers)

    return crop_dicm, max_box




def data_and_label_to_tfrecord(path_lists, save_tfrecord_path, tfrecord_filename, max_box,
                               augment_tfrecord_filename=None,
                               masker_flag=True,
                               dicm_show_path=True,
                               show_str1=None,
                               show_str2=None,
                               jilu_path=None):

    '''
    Save data into TFRecord
    param:
        path_list:每个病人的数据路径,类型list
        labels:   和path_list对应的病人的标签，类型list
        str_list:["BMP","BMP_Marker","DCM"]
        filename:生成的TFRecord文件保存路径
        bound_box_path:要保存的bounding_box路径，如果给了路径就把它保存到给的路径下，没有就不保存
    '''

    #生成tfrecord文件路径及文件名
    filename_path = os.path.join(save_tfrecord_path, tfrecord_filename)
    print(filename_path)
    if augment_tfrecord_filename is not None:
        augment_file_path = os.path.join(save_tfrecord_path,augment_tfrecord_filename)
        print(augment_file_path)

    if not os.path.isdir(save_tfrecord_path):
        os.makedirs(save_tfrecord_path)
    else:
        if os.path.isfile(filename_path):
            print("%s exists" % filename_path)
            return

    try:
        print("Converting data into %s ..." % filename_path)
        writer = tf.python_io.TFRecordWriter(filename_path)

        if augment_tfrecord_filename is not None:
            print("Converting augment_data into %s ..." % augment_file_path)
            augment_writer=tf.python_io.TFRecordWriter(augment_file_path)

    except Exception as e:
        print(e)
        return

    #记录处理过程信息
    jilu = []
    num = 0
    for path in path_lists:
        temp = []
        print(path)
        try:
            #读取pkl信息
            data, label, bounding_box = read_pkl(path)
            print(label)
            if label >= 4:
                label = 1
            else:
                label = 0
            print(label)

            #切割图片，分两种情况，一是加入背景信息，二是不加入背景信息，通过masker_flag设置
            if masker_flag is False:
                crop_data, _ = crop_dicm(data, bounding_box, max_box)
            else:
                maskers = read_masker_data(path)
                crop_data, _ = crop_dicm(data, bounding_box, max_box, maskers)

            #保存数据，用于可视化
            if dicm_show_path is True:
                save_dicm(path,crop_data,show_str1,show_str2)

            #数据转换成tfrecord
            crop_datas = np.array(crop_data, np.float32)
            dicm_raw = crop_datas.tobytes()
            feature = {'dicm_raw': _bytes_feature(dicm_raw),
                       'label': _int64_feature(label)}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

            #数据增强
            if augment_tfrecord_filename is not None:
                pass

            print(crop_data.shape)
            temp.append(path)
            temp.append(label)
            print(temp)
            jilu.append(temp)
            num = num+1
        except Exception as e:
            print(e)

    if jilu_path is not None:
        jilu = pd.DataFrame(jilu)
        jilu.to_csv(jilu_path)
        print('总样本数',num)

    writer.close()
    if augment_tfrecord_filename is not None:
        augment_writer.close()


def read_and_label_decode(filename, max_box):
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
    dicm_array = tf.reshape(dicm_raw, max_box)
    return dicm_array, label


def data_to_tfrecord(path_lists, save_tfrecord_path, tfrecord_filename, max_box_path=r'../data/dce_bound',
                     bound_filename='bounding_box.csv', masker_flag=True):
    '''
    Save data into TFRecord
    param:
        path_list:每个病人的数据路径,类型list
        labels:   和path_list对应的病人的标签，类型list
        str_list:["BMP","BMP_Marker","DCM"]
        filename:生成的TFRecord文件保存路径
        bound_box_path:要保存的bounding_box路径，如果给了路径就把它保存到给的路径下，没有就不保存
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
    max_box = read_max_box(max_box_path, bound_filename)
    for path in path_lists:
        print(path)
        try:
            data, label, bounding_box = read_pkl(path)
            if masker_flag is False:
                crop_data, _ = crop_dicm(data, bounding_box, max_box)
            else:
                maskers = read_masker_data(path)
                crop_data, _ = crop_dicm(data, bounding_box, max_box, maskers)
            crop_data = np.array(crop_data, np.float32)

            dicm_raw = crop_data.tobytes()

            feature = {'dicm_raw': _bytes_feature(dicm_raw),
                       'label': _int64_feature(label)}

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            print(crop_data.shape)

        except Exception as e:
            print(e)

    writer.close()

    return max_box


def read_and_decode(filename, max_box_path='/home/wangguangyuan/demo/retcel_project/data/dce_bound',
                    bound_filename='bounding_box.csv'):
    '''
    读取TFrecode文件里的数据并解码
    param:
        filename:文件路径
    return:
        ...
    '''

    max_box = read_max_box(max_box_path, bound_filename)
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = {'dicm_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)}

    features = tf.parse_single_example(serialized_example, features=features)

    label = tf.cast(features['label'], tf.int32)

    dicm_raw = features['dicm_raw']
    dicm_raw = tf.decode_raw(dicm_raw, tf.float32)
    dicm_array = tf.reshape(dicm_raw, max_box)
    return dicm_array, label


def mid_3d(box_all):
    '''
    找到每个3dbox的中心
    para:
        box:3*2的范围
    return:
        每个维度的长度和中心（2*3）
        长度为奇数说明，存在中心（整数），反之中心为x.5
    '''
    measure_all = np.zeros((len(box_all), 2, 3))
    for i in range(len(box_all)):
        measure_all[i, 0, :] = box_all[i, :, 1] - box_all[i, :, 0] + 1
        measure_all[i, 1, :] = np.mean(box_all[i, :, :], axis=1)
    return measure_all


def find_max_box(measure_all):
    '''
    找到所有用户Box长宽高的最大值
    '''
    max_3d = np.max(measure_all, axis=0)[0, :]
    return np.array(max_3d, dtype=np.int16)


def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces, _, _ = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.9)
    face_color = [128 / 255, 128 / 255, 128 / 255]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()


if __name__ == '__main__':
    # path = r'../data/PA0/DWI'
    # tfrecord_path = r'../data/'
    # bounding_box_path = r'../data'
    # tfrecord_path_path = r'../data/'
    #
    # data_path = r'D:\User\wangguangyuan\data\rectel orignal\rectel_data'
    #
    # datt_path = r'D:\User\wangguangyuan\datt\rectel orignal\rectel_datt'
    #
    # labels = [1] * 14
    # _, second_path, _ = read_files.get_files_path(data_path)
    #
    # pp = read_files.get_path_type(second_path, 'DCE')
    #
    # _, ss, _ = read_files.get_files_path(datt_path)
    #
    # train_ss, test_ss, train_labels, test_labels = train_test_split(ss, labels, test_size=0.3, random_state=0)

    '''
    测试process_data函数
    '''

    #    process_data(pp,labels,'data','datt',bound_box_path=r'../data/dce_bound')




    '''
    测试data_to_tfrecord函数代码段
    '''

    #    train_max_3d = data_to_tfrecord(train_ss,tfrecord_path,'train.tfrecord',masker_flag=False)
    #    test_max_3d = data_to_tfrecord(test_ss,tfrecord_path,'test.tfrecord',masker_flag=False)


    '''
    测试read_pkl函数
    '''

    #    data,label,bounding_box=read_pkl(r'D:\User\wangguangyuan\datt\rectel orignal\rectel_datt\333027\DCE')
    #    masker=read_masker_data(r'D:\User\wangguangyuan\datt\rectel orignal\rectel_datt\333027\DCE')

    '''
    测试crop_dicm函数
    '''

    #    max_box=read_max_box()
    #    crop_data,max_box=crop_dicm(data,bounding_box,max_box,masker)

    '''
    测试save_dicm函数
    '''

    #    _,datt_path,_=read_files.get_files_path(r'D:\User\wangguangyuan\datt\rectel orignal\rectel_datt')
    #    max_box=read_max_box()
    #
    #    for each_path in datt_path:
    #        try:
    #            data,label,bounding_box=read_pkl(each_path)
    #            masker=read_masker_data(each_path)
    #            crop_data,_=crop_dicm(data,bounding_box,max_box,masker)
    #            save_dicm(each_path,crop_data,'datt','dicm_image')
    #
    #        except Exception as e:
    #            print(e)


    '''
    测试extract_one_patient_roi函数代码段
    '''

    #    dicm_array,maskers_out,a,b=extract_one_patient_roi(path,["BMP","BMP_Marker","DCM"])
    #    dicoms=dicm_array

    #    save_maskers(path,maskers_out,'data','dat')
    #
    #    plot_3d(maskers_out,0)



    '''
    测试生的output.tfrecord文件里的数据是否正确
    '''

    #    for serialized_example in tf.python_io.tf_record_iterator(tfrecord_path):
    #        example=tf.train.Example()
    #        example.ParseFromString(serialized_example)
    #        dicm_raw=example.features.feature['dicm_raw'].bytes_list.value
    #        label=example.features.feature['label'].int64_list.value
    #        print(label)



    '''
    测试函数read_and_decode()
    '''

    # dicm_array, label = read_and_decode('../data/train.tfrecord')
    #
    # with tf.Session() as sess:
    #     init_op = tf.global_variables_initializer()
    #
    #     sess.run(init_op)
    #
    #     coord = tf.train.Coordinator()
    #
    #     x_train_batch, y_train_batch = tf.train.shuffle_batch([dicm_array, label],
    #                                                           batch_size=3, capacity=20, min_after_dequeue=10,
    #                                                           num_threads=2)
    #
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #
    #     data = sess.run(x_train_batch)
    #     label = sess.run(y_train_batch)
    #
    #     coord.request_stop()
    #     coord.join(threads)
    path = '/data/g/wangguangyuan/rectal_data/7/PA45/DCE'
    a,b,c = find_one_period(path, ["BMP","BMP_Marker","DCM"])
    print(len(a), len(b), len(c))
    for num, bb in enumerate(b):
        cv2.imwrite('/data/b/result/{}.png'.format(num), bb)



