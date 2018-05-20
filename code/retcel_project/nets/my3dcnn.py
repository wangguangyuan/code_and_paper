
import tensorlayer as tl
from tensorlayer.layers import *
import tensorflow as tf
import numpy as np
import math

class SqueezeLayer(Layer):
    '''
    The :class : SqueezeLayer is squeese layer,Removes dimensions of size 1 from the shape of a tensor.
    Given a tensor `input`, this operation returns a tensor of the same type with
    all dimensions of size 1 removed. If you don't want to remove all size 1
    dimensions, you can remove specific size 1 dimensions by specifying
    `axis`.

    Args:
        input: A `Tensor`. The `input` to squeeze.
        axis: An optional list of `ints`. Defaults to `[]`.
        If specified, only squeezes the dimensions listed. The dimension
        index starts at 0. It is an error to squeeze a dimension that is not 1.
        name: A name for the operation (optional).
        squeeze_dims: Deprecated keyword argument that is now axis.

    Returns:
        A `Tensor`. Has the same type as `input`.
        Contains the same data as `input`, but has one or more dimensions of
        size 1 removed.
    '''

    def __init__( self,
                  layer,
                  axis = None,
                  name = None,
                  squeeze_dims = None
                  ):

        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print(" [TL] SqueezeLayer %s: squeeze axis:%s" % (self.name, str(axis)))

        self.outputs = tf.squeeze(self.inputs, axis=axis, name=name, squeeze_dims=squeeze_dims)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])

def my3dcnn(x,num_classes=2,is_train=True,reuse=False,spatial_squeeze=True,scope='3dcnn'):

    depth,width,height,=x.get_shape()[1:4]
    print(type(depth))
    shape_depth=math.ceil(int(depth)/8)
    shape_width=math.ceil(int(width)/32)
    shape_height=math.ceil(int(height)/32)
    print(shape_depth,shape_width,shape_height)

    with tf.variable_scope(scope,reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x,name='inputs')

        '''conv1'''
        network=Conv3dLayer(inputs,act = tf.nn.relu,shape = [3, 3, 3, 1, 32],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv1_1')

        network=Conv3dLayer(network,act = tf.nn.relu,shape = [3, 3, 3, 32, 32],
                   strides=[1, 1, 1, 1, 1],padding='SAME',name='conv1_2')

        network=PoolLayer(network,ksize=[1, 2, 2, 2, 1],strides=[1, 1, 2, 2, 1],
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

        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 128, 128],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv3_3')
        network=PoolLayer(network,ksize=[1, 2, 2, 2, 1],strides=[1, 2, 2, 2, 1],
                    padding='SAME',pool=tf.nn.max_pool3d,name='pool3')

        '''conv4'''
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 128, 256],
                    strides=[1, 1, 1, 1, 1],padding='SAME', name='conv4_1')
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 256, 256],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv4_2')
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 256, 256],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv4_3')
        network=PoolLayer(network,ksize=[1, 2, 2, 2, 1],strides=[1, 2, 2, 2, 1],
                    padding='SAME',pool=tf.nn.max_pool3d,name='pool4')

        '''conv5'''
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 256, 512],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv5_1')
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 512, 512],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv5_2')
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 512, 512],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv5_3')
        network=PoolLayer(network,ksize=[1, 2, 2, 2, 1],strides=[1, 1, 2, 2, 1],
                    padding='SAME',pool=tf.nn.max_pool3d,name='pool5')

        # Use conv2d instead of fully_connected layers.
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[shape_depth, shape_width, shape_height, 512, 512],
                        strides=[1, 1, 1, 1, 1],padding='VALID',name='fc6')

        network=DropoutLayer(network,keep=0.5,is_fix=True,is_train=is_train,name='drop6')

        network=Conv3dLayer(network,act=tf.nn.relu,shape=[1, 1, 1, 512, 256],
                        strides=[1, 1, 1, 1, 1],padding='SAME',name='fc7')

        network=DropoutLayer(network, keep=0.5, is_fix=True, is_train=is_train, name='drop7')
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[1, 1, 1, 256, num_classes],
                        strides=[1, 1, 1, 1, 1],padding='SAME',name='fc8')

        if spatial_squeeze==True:
            network=SqueezeLayer(network,axis=[1,2,3],name='fc8/squeezed')
        return network

        # y=network.outputs
        # ce=tl.cost.cross_entropy(y,y_,name='cost')
        # L2 = 0
        # for p in tl.layers.get_variables_with_name('relu/W', True, True):
        #     L2 += tf.contrib.layers.l2_regularizer(0.004)(p)
        # cost = ce + L2
        #
        # correct_prediction = tf.equal(tf.cast(tf.argmax(y,1), tf.int32), y_)
        # acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        #
        # return network, cost, acc

def my3dcnn_small(x,num_classes=2,is_train=True,reuse=False,spatial_squeeze=True,scope='3dcnn'):

    depth,width,height,=x.get_shape()[1:4]
    print(type(depth))
    shape_depth=math.ceil(int(depth)/8)
    shape_width=math.ceil(int(width)/32)
    shape_height=math.ceil(int(height)/32)
    print(shape_depth,shape_width,shape_height)

    with tf.variable_scope(scope,reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x,name='inputs')

        '''conv1'''
        network=Conv3dLayer(inputs,act = tf.nn.relu,shape = [3, 3, 3, 1, 8],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv1_1')

        network=Conv3dLayer(network,act = tf.nn.relu,shape = [3, 3, 3, 8, 16],
                   strides=[1, 1, 1, 1, 1],padding='SAME',name='conv1_2')

        network=PoolLayer(network,ksize=[1, 2, 2, 2, 1],strides=[1, 1, 2, 2, 1],
                    padding='SAME',pool = tf.nn.max_pool3d,name ='pool1')

        '''conv2'''
        network=Conv3dLayer(network,act = tf.nn.relu,shape = [3, 3, 3, 16,16],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv2_1')

        network=Conv3dLayer(network,act = tf.nn.relu,shape = [3, 3, 3, 16, 16],
                   strides=[1, 1, 1, 1, 1],padding='SAME',name='conv2_2')

        network=PoolLayer(network,ksize=[1, 2, 2, 2, 1],strides=[1, 2, 2, 2, 1],
                    padding='SAME',pool = tf.nn.max_pool3d,name='pool2')


        '''conv3'''
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 16, 32],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv3_1')

        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 32, 32],
                   strides=[1, 1, 1, 1, 1],padding='SAME',name='conv3_2')

        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 32, 32],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv3_3')
        network=PoolLayer(network,ksize=[1, 2, 2, 2, 1],strides=[1, 2, 2, 2, 1],
                    padding='SAME',pool=tf.nn.max_pool3d,name='pool3')

        '''conv4'''
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 32, 64],
                    strides=[1, 1, 1, 1, 1],padding='SAME', name='conv4_1')
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 64, 64],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv4_2')
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 64, 64],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv4_3')
        network=PoolLayer(network,ksize=[1, 2, 2, 2, 1],strides=[1, 2, 2, 2, 1],
                    padding='SAME',pool=tf.nn.max_pool3d,name='pool4')

        '''conv5'''
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 64, 128],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv5_1')
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 128, 128],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv5_2')
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 128, 128],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv5_3')
        network=PoolLayer(network,ksize=[1, 2, 2, 2, 1],strides=[1, 1, 2, 2, 1],
                    padding='SAME',pool=tf.nn.max_pool3d,name='pool5')

        # Use conv2d instead of fully_connected layers.
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[shape_depth, shape_width, shape_height, 128, 128],
                        strides=[1, 1, 1, 1, 1],padding='VALID',name='fc6')

        network=DropoutLayer(network,keep=0.5,is_fix=True,is_train=is_train,name='drop6')

        network=Conv3dLayer(network,act=tf.nn.relu,shape=[1, 1, 1, 128, 512],
                        strides=[1, 1, 1, 1, 1],padding='SAME',name='fc7')

        network=DropoutLayer(network, keep=0.5, is_fix=True, is_train=is_train, name='drop7')
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[1, 1, 1, 512, num_classes],
                        strides=[1, 1, 1, 1, 1],padding='SAME',name='fc8')

        if spatial_squeeze==True:
            network=SqueezeLayer(network,axis=[1,2,3],name='fc8/squeezed')
        return network

        # y=network.outputs
        # ce=tl.cost.cross_entropy(y,y_,name='cost')
        # L2 = 0
        # for p in tl.layers.get_variables_with_name('relu/W', True, True):
        #     L2 += tf.contrib.layers.l2_regularizer(0.004)(p)
        # cost = ce + L2
        #
        # correct_prediction = tf.equal(tf.cast(tf.argmax(y,1), tf.int32), y_)
        # acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        #
        # return network, cost, acc
def my3dcnn_ss(x,num_classes=2,is_train=True,reuse=False,spatial_squeeze=True,scope='3dcnn'):

    depth,width,height,=x.get_shape()[1:4]
    print(type(depth))
    shape_depth=math.ceil(int(depth)/8)
    shape_width=math.ceil(int(width)/32)
    shape_height=math.ceil(int(height)/32)
    print(shape_depth,shape_width,shape_height)

    with tf.variable_scope(scope,reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x,name='inputs')

        '''conv1'''
        network=Conv3dLayer(inputs,act = tf.nn.relu,shape = [3, 3, 3, 1, 8],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv1_1')

        network=Conv3dLayer(network,act = tf.nn.relu,shape = [3, 3, 3, 8, 16],
                   strides=[1, 1, 1, 1, 1],padding='SAME',name='conv1_2')

        network=PoolLayer(network,ksize=[1, 2, 2, 2, 1],strides=[1, 1, 2, 2, 1],
                    padding='SAME',pool = tf.nn.max_pool3d,name ='pool1')

        '''conv2'''
        network=Conv3dLayer(network,act = tf.nn.relu,shape = [3, 3, 3, 16,16],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv2_1')

        network=Conv3dLayer(network,act = tf.nn.relu,shape = [3, 3, 3, 16, 16],
                   strides=[1, 1, 1, 1, 1],padding='SAME',name='conv2_2')

        network=PoolLayer(network,ksize=[1, 2, 2, 2, 1],strides=[1, 2, 2, 2, 1],
                    padding='SAME',pool = tf.nn.max_pool3d,name='pool2')


        '''conv3'''
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 16, 32],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv3_1')

        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 32, 32],
                   strides=[1, 1, 1, 1, 1],padding='SAME',name='conv3_2')

        network=Conv3dLayer(network,act=tf.nn.relu,shape=[1, 1, 1, 32, 32],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv3_3')
        network=PoolLayer(network,ksize=[1, 2, 2, 2, 1],strides=[1, 2, 2, 2, 1],
                    padding='SAME',pool=tf.nn.max_pool3d,name='pool3')

        '''conv4'''
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 32, 64],
                    strides=[1, 1, 1, 1, 1],padding='SAME', name='conv4_1')
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 64, 64],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv4_2')
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[1, 1, 1, 64, 64],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv4_3')
        network=PoolLayer(network,ksize=[1, 2, 2, 2, 1],strides=[1, 2, 2, 2, 1],
                    padding='SAME',pool=tf.nn.max_pool3d,name='pool4')

        '''conv5'''
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 64, 128],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv5_1')
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[1, 1, 1, 128, 128],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv5_2')
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 128, 128],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv5_3')
        network=PoolLayer(network,ksize=[1, 2, 2, 2, 1],strides=[1, 1, 2, 2, 1],
                    padding='SAME',pool=tf.nn.max_pool3d,name='pool5')

        # Use conv2d instead of fully_connected layers.
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[shape_depth, shape_width, shape_height, 128, 128],
                        strides=[1, 1, 1, 1, 1],padding='VALID',name='fc6')

        network=DropoutLayer(network,keep=0.5,is_fix=True,is_train=is_train,name='drop6')

        network=Conv3dLayer(network,act=tf.nn.relu,shape=[1, 1, 1, 128, 512],
                        strides=[1, 1, 1, 1, 1],padding='SAME',name='fc7')

        network=DropoutLayer(network, keep=0.5, is_fix=True, is_train=is_train, name='drop7')
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[1, 1, 1, 512, num_classes],
                        strides=[1, 1, 1, 1, 1],padding='SAME',name='fc8')

        if spatial_squeeze==True:
            network=SqueezeLayer(network,axis=[1,2,3],name='fc8/squeezed')
        return network

def my3dcnn_sss(x,num_classes=2,is_train=True,reuse=False,spatial_squeeze=True,scope='3dcnn'):
    depth,width,height,=x.get_shape()[1:4]
    print(type(depth))
    shape_depth=math.ceil(int(depth)/8)
    shape_width=math.ceil(int(width)/32)
    shape_height=math.ceil(int(height)/32)
    print(shape_depth,shape_width,shape_height)

    with tf.variable_scope(scope,reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x,name='inputs')

        '''conv1'''
        network=Conv3dLayer(inputs,act = tf.nn.relu,shape = [3, 3, 3, 1, 8],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv1_1')

        network=Conv3dLayer(network,act = tf.nn.relu,shape = [3, 3, 3, 8, 16],
                   strides=[1, 1, 1, 1, 1],padding='SAME',name='conv1_2')

        network=PoolLayer(network,ksize=[1, 2, 2, 2, 1],strides=[1, 1, 2, 2, 1],
                    padding='SAME',pool = tf.nn.max_pool3d,name ='pool1')

        '''conv2'''
        network=Conv3dLayer(network,act = tf.nn.relu,shape = [3, 3, 3, 16,16],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv2_1')

        network=Conv3dLayer(network,act = tf.nn.relu,shape = [3, 3, 3, 16, 16],
                   strides=[1, 1, 1, 1, 1],padding='SAME',name='conv2_2')

        network=PoolLayer(network,ksize=[1, 2, 2, 2, 1],strides=[1, 2, 2, 2, 1],
                    padding='SAME',pool = tf.nn.max_pool3d,name='pool2')


        '''conv3'''
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 16, 32],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv3_1')

        network=Conv3dLayer(network,act=tf.nn.relu,shape=[1, 1, 1, 32, 64],
                   strides=[1, 1, 1, 1, 1],padding='SAME',name='conv3_2')

        network=Conv3dLayer(network,act=tf.nn.relu,shape=[1, 1, 1, 64, 32],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv3_3')
        network=PoolLayer(network,ksize=[1, 2, 2, 2, 1],strides=[1, 2, 2, 2, 1],
                    padding='SAME',pool=tf.nn.max_pool3d,name='pool3')

        '''conv4'''
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 32, 32],
                    strides=[1, 1, 1, 1, 1],padding='SAME', name='conv4_1')
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[1, 1, 1, 32, 128],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv4_2')
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[1, 1, 1, 128, 64],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv4_3')
        network=PoolLayer(network,ksize=[1, 2, 2, 2, 1],strides=[1, 2, 2, 2, 1],
                    padding='SAME',pool=tf.nn.max_pool3d,name='pool4')

        '''conv5'''
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[3, 3, 3, 64, 64],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv5_1')
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[1, 1, 1, 64, 128],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv5_2')
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[1, 1, 1, 128, 32],
                    strides=[1, 1, 1, 1, 1],padding='SAME',name='conv5_3')
        network=PoolLayer(network,ksize=[1, 2, 2, 2, 1],strides=[1, 1, 2, 2, 1],
                    padding='SAME',pool=tf.nn.max_pool3d,name='pool5')

        # Use conv2d instead of fully_connected layers.
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[shape_depth, shape_width, shape_height, 32, 64],
                        strides=[1, 1, 1, 1, 1],padding='VALID',name='fc6')

        network=DropoutLayer(network,keep=0.5,is_fix=True,is_train=is_train,name='drop6')

        network=Conv3dLayer(network,act=tf.nn.relu,shape=[1, 1, 1, 64, 128],
                        strides=[1, 1, 1, 1, 1],padding='SAME',name='fc7')

        network=DropoutLayer(network, keep=0.5, is_fix=True, is_train=is_train, name='drop7')
        network=Conv3dLayer(network,act=tf.nn.relu,shape=[1, 1, 1, 128, num_classes],
                        strides=[1, 1, 1, 1, 1],padding='SAME',name='fc8')

        if spatial_squeeze==True:
            network=SqueezeLayer(network,axis=[1,2,3],name='fc8/squeezed')
        return network

if __name__=='__main__':
    # sess = tf.InteractiveSession()
    data=np.ones([10,20,100,100,1],np.float32)
    data_tensor=tf.convert_to_tensor(data)
    print(data_tensor)
    net=my3dcnn(data_tensor)
    # init_op=tl.layers.initialize_global_variables(sess)
    net.print_layers()
    net.print_params(False)
    train_param = tl.layers.get_variables_with_name('3dcnn')
    all_paras=net.all_params
    print(all_paras)

pass
# batch_szie = 3
# n_epoch = 50033
# learning_rate = 0.001
# n_step_epoch = int(382 / batch_szie)
# n_step = n_epoch * n_step_epoch
# print_freq = 20
#
# max_box=np.array([20,190,140])
#
# with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#
#     # 读取数据
#
#     x_train, y_train = img_pro.read_and_label_decode('/data/b/tfrecord/dce_train_tfrecord.tfrecord',max_box)
#     x_test, y_test = img_pro.read_and_label_decode('/data/b/tfrecord/dce_test_tfrecord.tfrecord',max_box)
#
#     x_tr = tf.expand_dims(x_train, -1)
#     x_te = tf.expand_dims(x_test,-1)
#
#
#     # 数据组合成batch_size
#     x_train_batch, y_train_batch = tf.train.shuffle_batch([x_tr, y_train],
#                                                           batch_size=batch_szie,
#                                                           capacity=2000,
#                                                           min_after_dequeue=1000,
#                                                           num_threads=16)
#
#     x_test_batch, y_test_batch = tf.train.batch([x_te, y_test],
#                                                 batch_size=batch_szie,
#                                                 capacity=2000,
#                                                 num_threads=3)
#
#     # 初始化线程
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#
#     # data=sess.run(x_test_batch)
#     # print(data.shape)
#
#
#     with tf.device('/gpu:0'):
#         # 定义训练模型
#         network, cost, acc = my3dcnn(x_train_batch, y_train_batch,reuse=False)
#         _,cost_test,acc_test = my3dcnn(x_test_batch,y_test_batch,reuse=True)
#
#         # 定义优化操作
#         train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
#                                           epsilon=1e-08, use_locking=False).minimize(cost)
#
#     #初始化变量
#     tl.layers.initialize_global_variables(sess)
#     print(sess.run(cost))
#     print(sess.run(train_op))

    # step = 0
    # for epoch in range(n_epoch):
    #     start_time = time.time()
    #     train_loss, train_acc, n_batch = 0, 0, 0
    #     for s in range(n_step_epoch):
    #         err, ac, _, = sess.run([cost, acc, train_op])
    #         step += 1
    #         train_loss += err
    #         train_acc += ac
    #         n_batch += 1
    #
    #     if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
    #         print("Epoch %d : Step %d-%d of %d took %fs" % (
    #         epoch, step, step + n_step_epoch, n_step, time.time() - start_time))
    #         print("   train loss: %f" % (train_loss / n_batch))
    #         print("   train acc: %f" % (train_acc / n_batch))
    #
    #         test_loss, test_acc,n_batch = 0,0,0
    #         for _ in range(80):
    #             err,ac = sess.run([cost_test,acc_test])
    #             test_loss += err
    #             test_acc += ac
    #             n_batch += 1
    #         print(" test loss: %f" %(test_loss/n_batch))
    #         print(" test acc: %f" %(test_acc/n_batch))
    #
    #     # save the network to .npz file
    #     if (epoch + 1) % (print_freq * 5) == 0:
    #         print("Save model " + "!" * 10)
    #         tl.files.save_npz(network.all_params, name='model.npz')

    # coord.request_stop()
    # coord.join(threads)

