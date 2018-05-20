from multiprocessing import Process, JoinableQueue, Queue
import numpy as np
import sys
import time
import scipy.misc as misc
def report(message='', error=False):
    if len(message) >= 70 and not error:
        message = message[:67] + '...'
        sys.stdout.write('\r{:70}{}'.format(message, "\n" if error else ""))
        sys.stdout.flush()

class Multiprocess_Data(object):
    def __init__(self, data_path, batch_size,  capacity, num_threads=4, all_samples=None):
        self.jobs = JoinableQueue(capacity)
        self.results = Queue(capacity)
        self.path = data_path
        self.threads = []
        self.num_theads = num_threads
        self.batch_size = batch_size
        self.all_samples = all_samples
        self.create_processes()

    # 读取数据路径， 根据自己的数据可以重载该函数
    def read_path(self, path):
        datas = np.loadtxt(path, str)
        if self.all_samples == None:
            self.all_samples = datas.shape[0]
            print('train sets of numbers is: %d' % self.all_samples)
        return datas

    #处理数据函数，如旋转，缩放大小等， 根据自己的需求重载此函数
    def process_data(self, each_data_path):
        image_path, maskee_path = each_data_path
        imge = misc.imread(image_path)
        masker = misc.imread(maskee_path)
        img_re = misc.imresize(imge, (256, 256))
        masker_re = misc.imresize(masker, (256, 256))
        return (img_re, masker_re)

    def worker(self):
        while True:
            try:
                data_path = self.jobs.get()
                result = self.process_data(data_path)
                self.results.put(result)
            except Exception as err:
                report(err, True)
            finally:
                self.jobs.task_done()

    def add_jobs(self):
        temp = 0
        datas = self.read_path(self.path)
        while(True):
            temp += 1
            indices = np.arange(len(datas))
            np.random.shuffle(indices)
            datas = datas[indices]
            print('temp: %d' % temp)

            if not self.jobs.full():
                for item in datas:
                    self.jobs.put(item)

    def create_processes(self):
        add_job_process = Process(target=self.add_jobs)
        add_job_process.daemon = True
        add_job_process.start()

        self.threads.append(add_job_process)
        for _ in range(self.num_theads):
            worker_process = Process(target=self.worker)
            worker_process.daemon = True
            worker_process.start()
            self.threads.append(worker_process)

        print('进程数目: %s' % len(self.threads))

    def shuffle_batch(self):
        image_batch = []
        masker_batch = []
        for _ in range(self.batch_size):
            image, masker = self.results.get()
            masker = np.expand_dims(masker, axis=2)
            image_batch.append(image)
            masker_batch.append(masker)
        return np.stack(image_batch), np.stack(masker_batch)

    def end_processers(self):
        for process in self.threads:
            process.terminate()
            process.join()

def get_one_minibatch(path, batch_size, f, shuffle=False):
    inputs = np.loadtxt(path, str)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        return f(inputs[excerpt])

def data_augument(inputs):
    data_batch, label_batch, shapes = [], [], []
    for image_path, maskee_path in inputs:
        imge = misc.imread(image_path)
        masker = misc.imread(maskee_path)
        shapes.append(imge.shape)
        img_re = misc.imresize(imge, (256, 256))
        masker_re = misc.imresize(masker, (256, 256))
        masker_re = np.expand_dims(masker_re, axis=2)
        data_batch.append(img_re)
        label_batch.append(masker_re)
    return np.stack(data_batch), np.stack(label_batch), shapes

def test():
    import matplotlib.pyplot as plt
    import scipy.misc as misc
    path = r'D:\AIchanger\data\train\train.txt'
    datas = np.loadtxt(path, str)
    image_path, maskee_path = datas[0]
    imge = misc.imread(image_path)
    masker = misc.imread(maskee_path)

    plt.figure()
    plt.imshow(imge)
    plt.figure()
    plt.imshow(masker)

    img_re = misc.imresize(imge, (512, 512))
    masker_re = misc.imresize(masker, (512, 512))
    plt.figure()
    plt.imshow(img_re)

    plt.figure()
    plt.imshow(masker_re)
    im_re = misc.imresize(img_re, imge.shape)
    plt.figure()
    plt.imshow(im_re)
    plt.show()
def test_multiprocess():
    temp = 0
    path = r'D:\AIchanger\data\train\train.txt'
    multi_process = Multiprocess_Data(path, 16, 32, 2)
    time.sleep(20)

    start = time.time()
    image_batch, masker_batch = multi_process.shuffle_batch()
    print(image_batch.shape)
    print(masker_batch.shape)
    end = time.time()
    print('take time: %f' % (end - start))

    for num, masker in enumerate(masker_batch):
        masker = np.squeeze(masker)
        misc.imsave(r'D:\tmp\{}_imge.jpg'.format(temp), image_batch[num])
        misc.imsave(r'D:\tmp\{}_masker.jpg'.format(temp), masker * 255)
        temp += 1

    start = time.time()
    image_batch, masker_batch = multi_process.shuffle_batch()
    print(image_batch.shape)
    print(masker_batch.shape)
    end = time.time()
    print('take time: %f' % (end - start))

    for num, masker in enumerate(masker_batch):
        masker = np.squeeze(masker)
        misc.imsave(r'D:\tmp\{}_imge.jpg'.format(temp), image_batch[num])
        misc.imsave(r'D:\tmp\{}_masker.jpg'.format(temp), masker * 255)
        temp += 1

    start = time.time()
    image_batch, masker_batch = multi_process.shuffle_batch()
    print(image_batch.shape)
    print(masker_batch.shape)
    end = time.time()
    print('take time: %f' % (end - start))

    for num, masker in enumerate(masker_batch):
        masker = np.squeeze(masker)
        misc.imsave(r'D:\tmp\{}_imge.jpg'.format(temp), image_batch[num])
        misc.imsave(r'D:\tmp\{}_masker.jpg'.format(temp), masker * 255)
        temp += 1

    start = time.time()
    image_batch, masker_batch = multi_process.shuffle_batch()
    print(image_batch.shape)
    print(masker_batch.shape)
    end = time.time()
    print('take time: %f' % (end - start))

    for num, masker in enumerate(masker_batch):
        masker = np.squeeze(masker)
        misc.imsave(r'D:\tmp\{}_imge.jpg'.format(temp), image_batch[num])
        misc.imsave(r'D:\tmp\{}_masker.jpg'.format(temp), masker * 255)
        temp += 1

if __name__ == '__main__':
    path = r'D:\AIchanger\data\train\train.txt'
    images, maskers = get_one_minibatch(path, 4, data_augument)
    pass
