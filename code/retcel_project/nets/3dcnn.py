
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import os
from retcel_project.image_process import image_process as img_pro
import time

def my3dcnn(x, y_,is_train=True,reuse=False,spatial_squeeze = True,dropout_keep_prob=0.5,fc_conv_padding='VALID',n_out=2):
    with tf.variable_scope("3dcnn",reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x,name='inputs')

        '''conv1'''
        network=Conv3dLayer(inputs,act = tf.nn.relu,shape = [3, 3, 3, 1, 32],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv1_1')

        network=Conv3dLayer(network,act = tf.nn.relu,shape = [3, 3, 3, 32, 32],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv1_2')

        network=PoolLayer(network,ksize=[1, 2, 2, 2, 1],strides=[1, 2, 2, 2, 1],
                    padding='SAME',pool = tf.nn.max_pool3d,name ='pool1')

        '''conv2'''
        network=Conv3dLayer(network,act = tf.nn.relu,shape = [3, 3, 3, 32,64],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv2_1')

        network=Conv3dLayer(network,act = tf.nn.relu,shape = [3, 3, 3, 64, 64],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv2_2')

        network=PoolLayer(network,ksize=[1, 2, 2, 2, 1],strides=[1, 2, 2, 2, 1],
                    padding='SAME',pool = tf.nn.max_pool3d,name='pool2')

        '''conv3'''
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 64, 128],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv3_1')

        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 128, 128],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv3_2')

        network = Conv3dLayer(network, act=tf.nn.relu, shape=[3, 3, 3, 128, 128],
                              strides=[1, 1, 1, 1, 1], padding='SAME', name='conv3_3')

        network=PoolLayer(network,ksize=[1, 2, 2, 2, 1],strides=[1, 2, 2, 2, 1],
                    padding='SAME',pool=tf.nn.max_pool3d,name='pool3')

        '''conv4'''
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 128, 256],
                    strides=[1, 1, 1, 1, 1],padding='SAME', name='conv4_1')

        network=Conv3dLayer(network, act=tf.nn.relu,shape=[3, 3, 3, 256, 256],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv4_2')

        network = Conv3dLayer(network, act=tf.nn.relu, shape=[3, 3, 3, 256, 256],
                    strides=[1, 1, 1, 1, 1], padding='SAME', name='conv4_3')
        network=PoolLayer(network,ksize=[1, 2, 2, 2,1],strides=[1, 2, 2, 2, 1],
                    padding='SAME',pool=tf.nn.max_pool3d,name='pool4')

        # Use Conv3d instead of fully_connected layers.
        network = Conv3dLayer(network, act=tf.nn.relu, shape=[2, 7, 7, 256, 2048],
                              strides=[1, 1, 1, 1, 1], padding=fc_conv_padding, name='fc5')
        network=DropoutLayer(network,keep=dropout_keep_prob,is_fix=True,is_train=is_train,name='drop1')

        network = Conv3dLayer(network, act=tf.nn.relu, shape=[1, 1, 1, 2048, 2048],
                              strides=[1, 1, 1, 1, 1], padding=fc_conv_padding, name='fc6')

        network = DropoutLayer(network, keep=dropout_keep_prob, is_fix=True, is_train=is_train, name='drop2')

        network = Conv3dLayer(network, act=None, shape=[1, 1, 1, 2048, n_out],
                              strides=[1, 1, 1, 1, 1], padding=fc_conv_padding, name='fc7')

        if spatial_squeeze :
            network = tf.squeeze(network,[1,2,3],name='output')

        y=network.outputs
        cost=tl.cost.cross_entropy(y,y_,name='cost')

        # L2 = 0
        # for p in tl.layers.get_variables_with_name('relu/W', True, True):
        #     L2 += tf.contrib.layers.l2_regularizer(0.004)(p)
        # cost = ce + L2
        correct_prediction = tf.equal(tf.cast(tf.argmax(y,1), tf.int32), y_)
        acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        return network, cost, acc

batch_szie = 3
n_epoch=10
learning_rate=0.001
n_step_epoch=int(13/batch_szie)
n_step=n_epoch*n_step_epoch
print_freq = 1

max_box=np.array([20,190,140])
with tf.device('/cpu:0'):
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        #读取数据
        # x_train,y_train = img_pro.read_and_decode('/home/wangguangyuan/demo/retcel_project/data/train.tfrecord')
        x_train, y_train = img_pro.read_and_label_decode('/data/b/tfrecord/dce_train_tfrecord.tfrecord', max_box)
        x_tr=tf.expand_dims(x_train,-1)
        
        #数据组合成batch_size
        x_train_batch,y_train_batch = tf.train.shuffle_batch([x_tr,y_train],
                                                             batch_size=batch_szie,
                                                             capacity=2000,
                                                             min_after_dequeue=1000,
                                                             num_threads=3)
        with tf.device('/gpu:0'):
            #定义训练模型
            network,cost,acc=my3dcnn(x_train_batch,y_train_batch)

            #定义优化操作
            train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                epsilon=1e-08, use_locking=False).minimize(cost)

        #初始化线程和变量
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        tl.layers.initialize_global_variables(sess)
        for _ in range(10):
            data=sess.run(x_train_batch)
            print(data.shape)

            # print(sess.run(cost))
        # step=0
        # for epoch in range(n_epoch):
        #     start_time = time.time()
        #     train_loss,train_acc,n_batch=0,0,0
        #     for s in range(n_step_epoch):
        #         err,ac,_,=sess.run([cost,acc,train_op])
        #         step +=1
        #         train_loss +=err
        #         train_acc +=ac
        #         n_batch +=1
        #
        #     if epoch +1==1 or (epoch+1) % print_freq==0:
        #         print("Epoch %d : Step %d-%d of %d took %fs" % (epoch, step, step + n_step_epoch, n_step, time.time() - start_time))
        #         print("   train loss: %f" % (train_loss/ n_batch))
        #         print("   train acc: %f" % (train_acc/ n_batch))
        #
        #     # save the network to .npz file
        #     # if (epoch + 1) % (print_freq * 50) == 0:
        #     #     print("Save model " + "!" * 10)
        #     #     tl.files.save_npz(network.all_params, name='model.npz')


        coord.request_stop()
        coord.join(threads)
