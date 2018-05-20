from data_prosses import read_and_decode, get_batch
#from .data_prosses import read_and_decode
from nets import resnet_v2_34, resnet_arg_scope
import numpy as np
from tensorflow.python.ops import control_flow_ops
import argparse
import tensorflow as tf
from tensorflow.contrib import slim
import time
import os

data_directory = r'F:\data2\data_train_normal_16.tfrecords'   #tfrecord data path
checkpoints_path = r'D:/signal_recongition/myowncheckpoints_16/'   #保存model
tensorboard_path = r'D:\signal_recongition\mytensorboard_16'   #保存tensorboard
restore_from = None #r'D:/signal_recongition/myowncheckpoints/model.ckpt-2000'


num_classes = 16            #classes
max_steps = 20000           #训练次数
batch_size = 128
learning_rate = 1e-2
init_learning_rate = 0.1
learning_rate_decay_factor = 0.1
num_examples_per_epoch_for_train = 10000       #总的数据样本
num_epochs_per_decay = 5                       #放大多少倍

MOVING_AVERAGE_DECAY = 0.9999       # The decay to use for the moving average.

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

def get_arguments():
    parser = argparse.ArgumentParser()
    #Basic model parameters.
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='Number of images to process in a batch')

    parser.add_argument('--data_dir', type=str,
                        default=data_directory,
                        help='Path to the data')

    parser.add_argument('--log_device_placement', type=bool, default=False,
                        help='Whether to log device placement.')

    parser.add_argument('--max_steps', type=int, default=max_steps,
                        help='Number of batches to run.')

    parser.add_argument('--checkpoints_path', type=str, default=checkpoints_path,
                        help='Directory where to write checkpoint.')

    parser.add_argument("--num_classes", type=int, default=num_classes,
                        help="Number of classes to predict (including background).")

    parser.add_argument("--learning_rate", type=float, default=learning_rate,
                        help="Learning rate for training")

    return parser.parse_args()

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

args = get_arguments()


def train():
    with tf.device('/cpu:0'):
        coord = tf.train.Coordinator()
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        # num_batches_per_epoch = (num_examples_per_epoch_for_train / args.batch_size)
        # decay_steps = int(num_batches_per_epoch * num_batches_per_epoch)
        #
        # lr = tf.train.exponential_decay(init_learning_rate,
        #                                 global_step,
        #                                 decay_steps,
        #                                 staircase=True)

        signals, labels = read_and_decode(args.data_dir)
        signals = signals[:, 0:1]
        labels = tf.cast(labels, tf.int32)
        signals_batch, labels_batch = get_batch(signals, labels, args.batch_size)

    with tf.device('/gpu:0'):
        with slim.arg_scope(resnet_arg_scope()):
            net, end_points = resnet_v2_34(signals_batch, num_classes=args.num_classes)

        # restore_var = tf.global_variables()
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        # saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net,
                                                              labels=labels_batch,
                                                              name='cross_entropy_loss')
        reduce_loss = tf.reduce_mean(loss, name='mean_loss')

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(reduce_loss, global_step=global_step)

        optimizer = tf.train.AdamOptimizer(args.learning_rate)
        train_step = slim.learning.create_train_op(reduce_loss, optimizer, global_step=global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            reduce_loss = control_flow_ops.with_dependencies([updates], reduce_loss)

        correct_prediction = tf.equal(tf.cast(tf.argmax(net, 1), tf.int32), labels_batch)
        accury = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.device('/cpu:0'):
            # for var in tf.trainable_variables():
            #     tf.summary.histogram(var.op.name, var)
            tf.summary.scalar('loss', reduce_loss)
            tf.summary.scalar('acuury', accury)

            merged = tf.summary.merge_all()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        train_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
        sess.run(init)

        with tf.device('cpu:0'):
            # saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=20)
            # Saver for storing checkpoints of the model.
            saver = tf.train.Saver(var_list=var_list, max_to_keep=20)

            # Load variables if the checkpoint is provided.
            if restore_from is not None:
                loader = tf.train.Saver(var_list=var_list)
                load(loader, sess, restore_from)

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        signals_bt = sess.run(signals_batch)
        print(signals_bt.shape)

        start_epoch_time = 0
        total_loss, accuy, n_batch = 0, 0, 0
        for step in range(args.max_steps):
            start_time = time.time()
            if (step+1) % 1000 == 0:
                loss_value,  acc, _ = sess.run([reduce_loss, accury, train_step])
                total_loss += loss_value
                accuy += acc
                n_batch += 1
                save(saver, sess, args.checkpoints_path, step)
            else:
                loss_value, acc,  _ = sess.run([reduce_loss, accury, train_step])
                total_loss += loss_value
                accuy += acc
                n_batch += 1
                # print(step, ll)
            duration = time.time() - start_time
            print('step {:d}\t loss = {:.3f}, accury = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, acc, duration))

            if step % 100 == 0:
                summary_all = sess.run(merged)
                train_writer.add_summary(summary_all, step)
                print(' ** train_loss: %f acc: %f took %fs (2d with distortion)' %
                      (total_loss / n_batch, accuy / n_batch, time.time() - start_epoch_time))

                start_epoch_time = time.time()
                total_loss, accuy, n_batch = 0, 0, 0

        train_writer.close()
        coord.request_stop()
        coord.join(threads=threads)

if __name__ == '__main__':
    train()