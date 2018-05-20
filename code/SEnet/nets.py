from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib import slim
from tensorflow.contrib.framework.python.ops import add_arg_scope
import tensorflow as tf
import numpy as np

def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, np.ndarray, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")


#装饰器使该函数可以应用slim.arg_scope来管理传入参数
@add_arg_scope
def Global_Average_Pooling(inputs,
                           axis=[1, 2],
                           outputs_collections=None,
                           scope=None):
    """ Global Average Pooling.
        Input:
            4-D Tensor [batch, height, width, in_channels].
        Output:
            2-D Tensor [batch, pooled dim]
        Arguments:
            inputs: `Tensor`. Incoming 4-D Tensor.
            outputs_collections: The collections to which the outputs are added.
            scope: Optional scope for name_scope.
        """

    input_shape = get_incoming_shape(inputs)
    assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"

    with tf.name_scope(scope, 'GlobalAvgPool', [inputs]) as sc:
        outputs = tf.reduce_mean(inputs, axis=axis)

    return utils.collect_named_outputs(outputs_collections, sc, outputs)

#SEnet 原始squeeze_excitation_layerc层实现
@add_arg_scope
def Squeeze_excitation_layer(inputs, ratio,
                             outputs_collections=None,
                             scope=None):

    channal = utils.last_dimension(inputs.get_shape(), min_rank=4)
    num_outputs = channal // ratio
    with tf.name_scope(scope, 'SE_layer', [inputs]) as sc:
        squeeze = Global_Average_Pooling(inputs)

        flatten = slim.flatten(squeeze)

        excitation = slim.fully_connected(flatten,
                                          num_outputs,
                                          activation_fn=tf.nn.relu,
                                          normalizer_fn=None,
                                          scope='se_fc1')

        excitation = slim.fully_connected(excitation,
                                          channal,
                                          activation_fn=tf.nn.sigmoid,
                                          normalizer_fn=None,
                                          scope='se_fc2')

        excitation = tf.reshape(excitation, [-1, 1, 1, channal])


        #广播机制
        scale = inputs * excitation

    return utils.collect_named_outputs(outputs_collections, sc, scale)

	
#尝试对SEnet改进后的squeeze_excitation_layerc层实现	
@add_arg_scope
def Squeeze_excitation_layer2(inputs, ratio,
                             outputs_collections=None,
                             scope=None):
    channal = utils.last_dimension(inputs.get_shape(), min_rank=4)
    num_outputs = channal // ratio
    with tf.name_scope(scope, 'SE_layer2', [inputs]) as sc:
        squeeze = slim.conv2d(inputs, num_outputs, [1, 1],
                              stride=1,
                              activation_fn=tf.nn.relu,
                              normalizer_fn=None,
                              scope='squeeze')

        excitation = slim.conv2d(squeeze, channal, [1, 1],
                                 stride=1,
                                 activation_fn=tf.nn.sigmoid,
                                 normalizer_fn=None,
                                 scope='excitation')

        scale = inputs * excitation

    return utils.collect_named_outputs(outputs_collections, sc, scale)

if __name__=='__main__':
    data_tensor = tf.placeholder(tf.float32, [16, 32, 32, 3])
    with tf.name_scope('TF-Slim') as vs:
        end_points_collection = vs + '_end_points'
        net = slim.conv2d(data_tensor, 64, 3, 1,
                          padding='SAME',
                          activation_fn=tf.nn.relu,
                          scope='conv2d2')

        with slim.arg_scope([Global_Average_Pooling, slim.fully_connected, slim.flatten, Squeeze_excitation_layer],
                            outputs_collections=end_points_collection):
            outputs = Squeeze_excitation_layer(net, 16)
            print(outputs)

        end_points = utils.convert_collection_to_dict(end_points_collection)
        print(end_points)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
	#用tensorboard可视化模型结构，用于判断结构是否正确
    train_writer = tf.summary.FileWriter('D:/logdir/', tf.get_default_graph())

    init = tf.global_variables_initializer()
    sess.run(init)


