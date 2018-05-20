import tensorflow as tf
try:
    from retcel_project.deeplab_resnet import UnetArtrous,prepare_label
except:
    from deeplab_resnet import UnetArtrous, prepare_label
try:
    from retcel_project.image_process import segment_image_process as img_seg
except:
    from image_process import segment_image_process as img_seg

import numpy as np
import time
import os
import argparse

d2_max_box = np.array([512, 512])
data_directory = '/data/b/wangguangyuan/rectal_segment/DCE/tfrecord/dce_train.tfrecord'   #tfrecord data path
checkpoints_path = '/data/b/wangguangyuan/rectal_segment/DCE/tfrecord/myown_checkpoint/unet_atrous_muitlGPU/'   #保存model
tensorboard_path = '/data/b/wangguangyuan/rectal_segment/DCE/tfrecord/myown_tensorboard/unet_atrous_mutilGPU'   #保存tensorboard

num_gpus = 4               #numbers of gpu
num_classes = 2            #classes
max_steps = 10000                #训练次数
batch_size = 4
init_learning_rate = 0.1
learning_rate_decay_factor = 0.1
num_examples_per_epoch_for_train = 10000       #总的数据样本
num_epochs_per_decay = 5                       #放大多少倍

TOWER_NAME = 'tower'                           #gpu标志
INPUT_SIZE = '512,512'   #input size
IGNORE_LABEL = 0          # ignore label
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

    parser.add_argument('--num_gpus', type=int, default=num_gpus,
                        help='How many GPUs to use.')

    parser.add_argument('--log_device_placement', type=bool, default=False,
                        help='Whether to log device placement.')

    parser.add_argument('--max_steps', type=int, default=max_steps,
                        help='Number of batches to run.')

    parser.add_argument('--checkpoints_path', type=str, default=checkpoints_path,
                        help='Directory where to write checkpoint.')

    parser.add_argument("--ignore_label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")

    parser.add_argument("--input_size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--num_classes", type=int, default=num_classes,
                        help="Number of classes to predict (including background).")

    # parser.add_argument('--log_device_placement', type=bool, default=False,
    #                     help='Whether to log device placement.')

    return parser.parse_args()

args = get_arguments()


def _loss(logits, labels):
    # Pixel-wise softmax loss.
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels,name='cross_entropy_per_example')
    reduced_loss = tf.reduce_mean(loss, name='cross_entropy')
    return reduced_loss


def tower_loss(image_batch, label_batch, is_train=True, num_classes=2, reuse_variables=None):

    # Create network.
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        net = UnetArtrous({'data': image_batch}, is_training=is_train, num_classes=num_classes)

    # Predictions.
    raw_output = net.layers['uconv1']
    prediction = tf.reshape(raw_output, [-1, args.num_classes])

    label_proc = prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]), num_classes=num_classes)
    gt = tf.reshape(label_proc, [-1, args.num_classes])

    losses = _loss(prediction, gt)

    return losses


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)
          # Append on a 'tower' dimension which we will average over below.
          grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        height, width = map(int, args.input_size.split(','))
        # Create queue coordinator.
        coord = tf.train.Coordinator()
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)

        # Calculate the learning rate schedule.
        num_batches_per_epoch = (num_examples_per_epoch_for_train / args.batch_size)

        decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(init_learning_rate,
                                        global_step,
                                        decay_steps,
                                        learning_rate_decay_factor,
                                        staircase=True)

        # Create an optimizer that performs gradient descent.
        opt = tf.train.GradientDescentOptimizer(lr)

        # Create an optimizer that performs gradient descent.
        # opt = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY,
        #                                 momentum=RMSPROP_MOMENTUM,
        #                                 epsilon=RMSPROP_EPSILON)

        # Get images and labels for CIFAR-10.
        images, labels = img_seg.distored_inputs(args.data_dir, d2_max_box,
                                                               args.batch_size,
                                                               input_size=[height,width],
                                                               random_scale=True,
                                                               random_mirror=True,
                                                               ignore_label=0)

        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([images, labels],
                                                                    capacity=2 * args.num_gpus)

        # Calculate the gradients for each model tower.
        tower_grads = []
        reuse = False
        for i in [2, 3, 4, 5]:
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (TOWER_NAME, i)):
                    # Dequeues one batch for the GPU
                    image_batch, label_batch = batch_queue.dequeue()
                    # Calculate the loss for one tower of the CIFAR model. This function
                    # constructs the entire CIFAR model but shares the variables across
                    # all towers.
                    loss = tower_loss(image_batch, label_batch,
                                      is_train=True,
                                      num_classes=args.num_classes,
                                      reuse_variables=reuse)

                    print(loss)
                    # Reuse variables for the next tower.
                    reuse = True

                    # Calculate the gradients for the batch of data on this CIFAR tower.
                    grads = opt.compute_gradients(loss)

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        sess.run(init)
        
        train_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
        threads = tf.train.start_queue_runners(sess=sess)

        for step in range(args.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = args.batch_size * args.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / args.num_gpus

                format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')

                print(format_str % (step, loss_value,
                                    examples_per_sec, sec_per_batch))

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == args.max_steps:
                checkpoint_path = os.path.join(args.checkpoints_path, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        train_writer.close()
        coord.request_stop()
        coord.join(threads)


def main():
    train()


if __name__ == '__main__':
    main()
