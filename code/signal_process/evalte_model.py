import tensorflow as tf
from nets import resnet_v2_34, resnet_arg_scope
from tensorflow.contrib import slim
from data_prosses import minibatches, get_one_minibatch, data_augument
import numpy as np
import os

test_data_path = r'F:\data2\test_data.txt'
restore_from = r'D:/signal_recongition/myowncheckpoints/model.ckpt-19999'
batch_size = 128

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')

def train():
    with tf.device('/cpu:0'):
        inputs = tf.placeholder(dtype=tf.float32, shape=[batch_size, 4096, 1])
        labels = tf.placeholder(dtype=tf.int32, shape=[batch_size])

    with tf.device('/gpu:0'):
        with slim.arg_scope(resnet_arg_scope()):
            net, end_points = resnet_v2_34(inputs, num_classes=10, is_training=False)

        # restore_var = tf.global_variables()
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net,
                                                              labels=labels,
                                                              name='cross_entropy_loss')
        reduce_loss = tf.reduce_mean(loss, name='mean_loss')

        # train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(reduce_loss, global_step=global_step)
        pre = tf.cast(tf.argmax(net, 1), tf.int32)
        correct_prediction = tf.equal(pre, labels)
        accury = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()

        sess.run(init)

        with tf.device('cpu:0'):
            # Load variables if the checkpoint is provided.
            if restore_from is not None:
                loader = tf.train.Saver(var_list=var_list)
                load(loader, sess, restore_from)


        data = np.loadtxt(test_data_path, str)
        # data = data[:10000]
        batch_data, batch_label = get_one_minibatch(data, batch_size, f=data_augument, shuffle=True)
        dd = batch_data[:, :, 0:1]
        print(dd.shape)
        feed_dict = {inputs: dd, labels: batch_label}
        loss_value, acc, pred = sess.run([reduce_loss, accury, pre], feed_dict=feed_dict)
        print(batch_label)
        print(pred)
        print('loss: %f, accuray: %f' %(loss_value, acc))


if __name__ == '__main__':
    train()




