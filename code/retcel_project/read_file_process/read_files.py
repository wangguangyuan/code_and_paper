# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 09:06:58 2017

@author: wangguangyuan
"""
import os
import pandas as pd
from sklearn.cross_validation import train_test_split

def get_files_path(path):
    
    '''
    获取文件的路径，该函数获取到第三层的路径
    param:
        path:起始路径
    return:
        first_level_path:第一层路径,类型list
        second_level_path:第二层路径,类型list
        third_level_path:第三层路径,类型list
    '''
    
    first_level_path=[]
    second_level_path=[]
    third_level_path=[]
    
    #读取第一层文件夹下的文件或文件夹,并拼接成路径
    first_level_path=[os.path.join(path, s) for s in os.listdir(path)]
    #读取第二层文件夹下的文件或文件夹,并拼接成路径
    for each_path in first_level_path:
        second_level_path += [os.path.join(each_path, s) for s in os.listdir(each_path)]
        
    for each_path in second_level_path:
        third_level_path += [os.path.join(each_path, s) for s in os.listdir(each_path)]
    return first_level_path, second_level_path, third_level_path


def get_liver_label(path):
    '''
    获得肝癌的label
    parm:
        path:原数据的path,如：'D:\orignal data\liver orignal\orignal\MRI_H_1\PA32_H_1\DCE'
    return：
        time_label:
        states_label:
        对应label不存在时均返回-1
    '''
    label_path = r'D:\tanxiaofeng\cox\new_data_end.csv'
    data_end = pd.read_csv(label_path)
    find_path = path.replace(r'D:\orignal data\liver orignal\orignal','G:\liuhuan\liver\orignal')
    
    find_index = data_end[(data_end['path1']==find_path)|(data_end['path2']==find_path)|
            (data_end['path3']==find_path)|(data_end['path4']==find_path)]
    try:
        find_index.index = [0]
        time_label = find_index['delta'][0]
        states_label = int(find_index['end_states'][0])
    except ValueError:
        time_label = -1
        states_label = -1
    return time_label,states_label

def get_path_type(path_list,label_list,types='DCE'):
    path_choose = []
    label_choose = []

    for index in range(len(path_list)):
        if os.path.split(path_list[index])[1] == types:
            path_choose.append(path_list[index])
            label_choose.append(label_list[index])
    return path_choose,label_choose



def convert_age(age_str):
    if age_str == u'1':
        return 0
    elif age_str == u'2':
        return 1
    elif age_str == u'2-3':
        return 2
    elif age_str == u'3':
        return 3
    elif age_str == u'4':
        return 4
    elif age_str == u'5':
        return 5
    elif age_str == u'3-4':
        return 3
    else:
        return -1



def read_xls(path):
    '''
    读取Excel表格里的数据
    '''
    data=pd.read_excel(path)
    result_data=data[['MRI_ID','Result','DirName']]
    drop_nan=result_data.dropna(axis=0)
    drop_nan['Result']=drop_nan['Result'].map(convert_age)
    data_value=drop_nan.values
    return data_value

def get_data_path(data,path,str1):
    '''

    :param data:
    :param path:
    :param str1:
    :return:
    example:
    label_path='/data/b/wanbbuanbyuan/rectel_label/data.xls'
    data_path='/data/g/wangguangyuan/rectal_data'
    data=read_files.read_xls(label_path)
    path_lists,label_lists=read_files.get_data_path(data,data_path,'DCE')
    '''
    path_list=[]
    label_list=[]
    for each in data:
        temp=each[2].replace('-','/')
        #lxs_path=os.path.join(path,temp,'DCE','*.mat')
        lxs_path=os.path.join(path,temp,str1)
        path_list.append(lxs_path)
        label_list.append(each[1])
    return path_list,label_list