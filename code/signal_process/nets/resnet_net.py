import collections
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.layers.python.layers import initializers

class Block(collections.namedtuple('Block',['scope','unit_fn','args'])):
    'A named tuple describing a Resnet block'
@add_arg_scope
def max_pool1d(inputs,
               kernel_size,
               stride=2,
               padding='same',
               data_format='channels_last',
               outputs_collections=None,
               scope=None):
    with tf.name_scope(scope, 'MaxPool1D', [inputs]) as sc:
        outputs = tf.layers.max_pooling1d(inputs, kernel_size, stride, padding, data_format, scope)
        return utils.collect_named_outputs(outputs_collections, sc, outputs)


def subsample(inputs,factor,scope=None):
    if factor == 1:
        return inputs
    else:
        return max_pool1d(inputs, 1, stride=factor, scope=scope)


def conv2d_same(inputs, num_outs, kernal_size, stride, scope=None):
    if stride == 1:
        return slim.convolution(inputs, num_outs, kernal_size,
                                stride=1,
                                padding='SAME',
                                scope=scope)
    else:
        pad_total = kernal_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [0, 0]])
        return slim.convolution(inputs, num_outs, kernal_size,
                                stride=stride,
                                padding='VALID',
                                scope=scope)

@add_arg_scope
def stack_blocks_dense(net, blocks, outputs_collections=None):
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' %(i+1), values=[net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit

                    net = block.unit_fn(net,
                                        depth=unit_depth,
                                        depth_bottleneck=unit_depth_bottleneck,
                                        stride=unit_stride)
            net = utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net


@add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        depth_in = utils.last_dimension(inputs.get_shape(), min_rank=3)

        if depth_in == depth:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.convolution(inputs, depth, 1,
                                        stride=stride,
                                        padding='SAME',
                                        normalizer_fn=None,
                                        activation_fn=None,
                                        scope='shortcut')

        residual = slim.dropout(preact, keep_prob=0.8, scope='dropout1')

        residual = conv2d_same(residual, depth_bottleneck, 16, stride, scope='conv1')

        residual = slim.dropout(residual, keep_prob=0.8, scope='dropout2')

        residual = slim.convolution(residual, depth, 16,
                                    stride=1,
                                    padding='SAME',
                                    normalizer_fn=None,
                                    activation_fn=None,
                                    scope='conv2'
                                    )

        output = shortcut+residual
        return utils.collect_named_outputs(outputs_collections, sc.name, output)


@add_arg_scope
def bottleneck2(inputs, depth, depth_bottleneck, stride, outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        # preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        preact = inputs  #tf.nn.relu(inputs)
        depth_in = utils.last_dimension(inputs.get_shape(), min_rank=3)

        if depth_in == depth:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.convolution(inputs, depth, 1,
                                        stride=stride,
                                        padding='SAME',
                                        normalizer_fn=None,
                                        activation_fn=None,
                                        scope='shortcut')

        # residual = slim.dropout(preact, keep_prob=0.8, scope='dropout1')

        residual = conv2d_same(preact, depth_bottleneck, 16, stride, scope='conv1')

        # residual = slim.dropout(residual, keep_prob=0.8, scope='dropout2')

        residual = slim.convolution(residual, depth, 16, stride=1, padding='SAME', scope='conv2')
        # residual = slim.convolution(residual, depth, 16,
        #                             stride=1,
        #                             padding='SAME',
        #                             normalizer_fn=None,
        #                             activation_fn=None,
        #                             scope='conv2'
        #                             )

        output = shortcut+residual
        return utils.collect_named_outputs(outputs_collections, sc.name, output)


def resnet_arg_scope(activation_fn=tf.nn.relu,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True,
                     is_training=True):

    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'is_training': is_training
    }

    with slim.arg_scope([slim.convolution],
                        weights_initializer=initializers.xavier_initializer(),
                        activation_fn=activation_fn,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        ):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc

def resnet_arg_scope2(activation_fn=tf.nn.relu,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):

    # 不加BachNormal
    # batch_norm_params = {
    #     'decay': batch_norm_decay,
    #     'epsilon': batch_norm_epsilon,
    #     'scale': batch_norm_scale,
    #     'updates_collections': tf.GraphKeys.UPDATE_OPS,
    # }

    with slim.arg_scope([slim.convolution],
                        weights_initializer=initializers.xavier_initializer(),
                        activation_fn=activation_fn,
                        normalizer_fn=None,
                        ) as arg_sc:
        # with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc

def resnet_v22(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              reuse=None,
              scope=None
              ):
    with tf.variable_scope(scope, 'resnet_v22', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.convolution, bottleneck, stack_blocks_dense],
                            outputs_collections=end_points_collection):
            with slim.arg_scope([slim.dropout], is_training=is_training):
                net = inputs
                net = slim.convolution(net, 64, 16, stride=1, padding='SAME', scope='conv1')

                shortcut = subsample(net, factor=2, scope='shortcut')
                net = conv2d_same(net, 64, 16, stride=2, scope='conv2')
                # net = slim.dropout(net, keep_prob=0.8, scope='droput')
                net = slim.convolution(net, 64, 16, stride=1, padding='SAME', scope='conv3')
                # net = slim.convolution(net, 64, 16,
                #                        stride=1,
                #                        padding='SAME',
                #                        normalizer_fn=None,
                #                        activation_fn=None,
                #                        scope='conv3'
                #                        )

                net = net + shortcut
                net = stack_blocks_dense(net, blocks)
                # net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
                print('last:', net)
                if num_classes is not None:
                    net = slim.flatten(net, scope='flatten')
                    net =slim.fully_connected(net, num_classes,
                                              activation_fn=tf.nn.relu,
                                              normalizer_fn=None,
                                              scope='fc')
                end_points = utils.convert_collection_to_dict(end_points_collection)
                if num_classes is not None:
                    end_points['predictions'] = slim.softmax(net, scope='predictions')
                return net, end_points


def resnet_v2(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              reuse=None,
              scope=None
              ):
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.convolution, bottleneck, stack_blocks_dense],
                            outputs_collections=end_points_collection):
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
                net = inputs
                net = slim.convolution(net, 64, 16, stride=1, padding='SAME', scope='conv1')

                shortcut = subsample(net, factor=2, scope='shortcut')
                net = conv2d_same(net, 64, 16, stride=2, scope='conv2')
                net = slim.dropout(net, keep_prob=0.8, scope='droput')
                net = slim.convolution(net, 64, 16,
                                       stride=1,
                                       padding='SAME',
                                       normalizer_fn=None,
                                       activation_fn=None,
                                       scope='conv3'
                                       )
                net = net + shortcut
                net = stack_blocks_dense(net, blocks)
                net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
                print('last:', net)
                if num_classes is not None:
                    net = slim.flatten(net, scope='flatten')
                    net =slim.fully_connected(net, num_classes,
                                              activation_fn=tf.nn.relu,
                                              normalizer_fn=None,
                                              scope='fc')
                end_points = utils.convert_collection_to_dict(end_points_collection)
                if num_classes is not None:
                    end_points['predictions'] = slim.softmax(net, scope='predictions')
                return net, end_points

def resnet_v2_34(inputs,
                 num_classes=None,
                 is_training=True,
                 reuse=None,
                 scope='resnet_v2_34'):
    blocks = [
        Block('block1', bottleneck, [(64, 64, 1), (64, 64, 2)] * 2),
        Block('block2', bottleneck, [(128, 128, 1), (128, 128, 2)] * 2),
        Block('block3', bottleneck, [(192, 192, 1), (192, 192, 2)] * 2),
        Block('block4', bottleneck, [(256, 256, 1), (256, 256, 2), (256, 256, 1)])]
    return resnet_v2(inputs, blocks, num_classes, is_training, reuse, scope)

def resnet_v2_34_2(inputs,
                 num_classes=None,
                 is_training=True,
                 reuse=None,
                 scope='resnet_v2_34_2'):
    blocks = [
        Block('block1', bottleneck2, [(64, 64, 1), (64, 64, 2)] * 2),
        Block('block2', bottleneck2, [(128, 128, 1), (128, 128, 2)] * 2),
        Block('block3', bottleneck2, [(192, 192, 1), (192, 192, 2)] * 2),
        Block('block4', bottleneck2, [(256, 256, 1), (256, 256, 2), (256, 256, 1)])]
    return resnet_v22(inputs, blocks, num_classes, is_training, reuse, scope)

if __name__ == '__main__':
    '''
    sess = tf.InteractiveSession()
    data = np.ones([10, 128, 3], np.float32)
    data_tensor = tf.convert_to_tensor(data, name='inputs')
    print(data_tensor)

    inputs = conv2d_same(data_tensor, 4, 16, 2)
    print('conv2d_same:', inputs)
    depth_in = utils.last_dimension(inputs.get_shape(), min_rank=3)
    print(depth_in)

    net = slim.convolution(data_tensor, 4, 1, 2, normalizer_fn=slim.batch_norm, padding='SAME',activation_fn=tf.nn.relu)
    print('conv[1,1]:', net)

    net = tf.layers.max_pooling1d(net, 1, 2, padding='valid')
    print('after max pool:', net)

    net = subsample(net,2,'subsample')
    print('subsample:', net)

    image_data = np.ones([1, 128, 128, 3],np.float32)

    image_tensor = slim.max_pool2d(image_data, [2, 2], stride=2)
    image_rank = image_tensor.get_shape().ndims
    print('image_rank:', image_rank)
    print(image_tensor)
    blocks=[Block('block', bottleneck, [(64, 64, 1)]+[(64, 64, 2)])]
    # result=bottleneck(data_tensor, 64, 64, stride=2,scope='test')
    with slim.arg_scope(resnet_arg_scope(is_traning=False)):
        result = stack_blocks_dense(data_tensor, blocks)
    print(result)

    reduc_resu = tf.reduce_mean(image_tensor, [1, 2], name='pool', keep_dims=True)
    print('reduce_mean:', reduc_resu)
    net_flatten = slim.flatten(net)
    print('flatten:', net_flatten)
    '''
    tensorborad_path = r'D:\logdir'
    sess = tf.InteractiveSession()
    data = np.ones([10, 4096, 1], np.float32)
    data_tensor = tf.convert_to_tensor(data, name='inputs')
    print('data:', data_tensor)
    with slim.arg_scope(resnet_arg_scope()) as sc:
        net, end_points = resnet_v2_34(data_tensor, 2)
    trainable = tf.trainable_variables()
    print('trainable:', trainable)
    print('end_points:', end_points)
    print('end:', net)
    # train_writer = tf.summary.FileWriter(tensorborad_path,sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)
    # train_writer.close()


