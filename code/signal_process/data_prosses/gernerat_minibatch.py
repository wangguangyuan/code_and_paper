import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

def get_data_path(path, save_path):
    # path = r'F:\data'
    #train_path = r'F:\data2'
    filelist = os.listdir(path)
    path_list = [os.path.join(path, s) for s in filelist]
    print(path_list)

    all_files = []
    for num, each_path in enumerate(path_list):
        files = os.listdir(each_path)
        each_path = [(os.path.join(each_path, s), num) for s in files]
        all_files.extend(each_path)
        # all_files.extend(each_path)

    with open(save_path, 'w') as f:
        for image_file, label in all_files:
            f.write('{} {}\n'.format(image_file, label))

def get_data_path2(path, train_path, test_path, lenth=20000):
    # path = r'F:\data'
    #train_path = r'F:\data2'
    filelist = os.listdir(path)
    path_list = [os.path.join(path, s) for s in filelist]
    print(path_list)

    train_files = []
    test_files = []
    for num, each_path in enumerate(path_list):
        files = os.listdir(each_path)
        each_path = [(os.path.join(each_path, s), num) for s in files]
        end_len = len(each_path)
        random.shuffle(each_path)
        train_files.extend(each_path[:lenth])
        test_files.extend(each_path[lenth:end_len])

    with open(train_path, 'w') as f:
        for image_file, label in train_files:
            f.write('{} {}\n'.format(image_file, label))
    with open(test_path, 'w') as f:
        for test_file, test_label in test_files:
            f.write('{} {}\n'.format(test_file, test_label))

def data_augument(inputs):
    # data_batch, label_batch = [], []
    data_batch = [np.loadtxt(each_path) for each_path, _ in inputs]
    label_batch = [int(label) for _, label in inputs]
    # for each_path, label in inputs:
    #     data = np.loadtxt(each_path).astype(np.float32)
    #     label = int(label)
    #     data_batch.append(data)
    #     label_batch.append(label)
    return np.stack(data_batch), np.stack(label_batch)

def minibatches(inputs, batch_size, f, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield f(inputs[excerpt])

def get_one_minibatch(inputs, batch_size, f, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)

        return f(inputs[excerpt])

if __name__ == '__main__':

    # data = np.loadtxt(r'F:\data2\train.txt', str)
    # batch_data, batch_label = get_one_minibatch(data, 32, f=data_augument, shuffle=True)
    # dd = batch_data[:, :, 0:1]
    # for num , data in enumerate(dd):
    #     plt.figure(num)
    #     plt.plot(data)
    # plt.show()

    # for batch_data, batch_label in minibatches(data, 32, f=data_augument, shuffle=True):
    #     dd = batch_data[:, :, 0:1]
    #     print('ddd')

    # with open(r'F:\data\16FSK\16FSK_1006.txt','r') as f:
    #     datas = pickle.load(f)
    #     print('dfd')

    #gernate test set
    # get_data_path(r'F:\data1', r'F:\data2\test.txt')

    #gernate train and test data
    get_data_path2(r'F:\data',
                   train_path=r'F:\data2\train_data_16.txt',
                   test_path=r'F:\data2\test_data_16.txt',
                   lenth=20000)