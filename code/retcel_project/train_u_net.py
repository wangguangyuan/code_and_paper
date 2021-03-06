import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os
import argparse

try:
    from retcel_project.image_process import segment_image_process as img_seg
except:
    from image_process import segment_image_process as img_seg
try:
    from retcel_project.nets import u_net_segment as mymodel
except:
    import nets.u_net_segment as mymodel

import time


d2_max_box = np.array([512, 512])
BATCH_SIZE = 8
DATA_DIRECTORY = '/data/b/wangguangyuan/rectal_segment/DCE/tfrecord/ts_fs_train.tfrecord'   #tfrecord data path
IGNORE_LABEL = 0          # ignore label
INPUT_SIZE = '512,512'   #input size
LEARNING_RATE = 1e-4      #learning rate
NUM_CLASSES = 2           #class of segmnt
NUM_STEPS = 10000         #all iterate steps

RESTORE_FROM = None
SAVE_NUM_IMAGES = 1
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = '/data/b/wangguangyuan/rectal_segment/DCE/tfrecord/myown_checkpoint/unet/'
DEVICE = '/gpu:3'

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

tensorboard_path = '/data/b/wangguangyuan/rectal_segment/logdir_dce_0.01'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")

    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing tfrecord data.")

    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")

    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")

    parser.add_argument("--is-training", action="store_true", default=True,
                        help="Whether to updates the running means and variances during the training.")

    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for training.")

    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")

    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")

    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")

    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")

    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")

    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")

    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")

    return parser.parse_args()


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')


def load(saver, sess, ckpt_path):
    '''Load trained weights.
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''

    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def main():
    with tf.device('/cpu:0'):
        """Create the model and start the training."""
        args = get_arguments()

        height, width = map(int, args.input_size.split(','))

        # Create queue coordinator.
        coord = tf.train.Coordinator()

        # Load reader.
        with tf.name_scope("create_inputs"):
            print('read data from:', args.data_dir)
            start_time = time.time()
            # image_batch, label_batch = img_seg.distored_inputs(args.data_dir, d2_max_box,
            #                                                    args.batch_size,
            #                                                    input_size=[height, width],
            #                                                    random_scale=True,
            #                                                    random_mirror=True,
            #                                                    ignore_label=0)

            image_batch, label_batch = img_seg.distored_inputs2(args.data_dir, d2_max_box,
                                                                args.batch_size,
                                                                maxmin_nor=True,
                                                                rotation=True)

            duration = time.time() - start_time
            print('read data took %f sec' % duration)

    with tf.device(DEVICE):
        # 定义训练模型
        net = mymodel.u_net(image_batch, is_train=True, reuse=False, n_out=args.num_classes)
        # net_test = mymodel.u_net(x_test_batch, is_train=False, reuse=True)

        # train loss
        raw_output = net.outputs

        restore_var = tf.global_variables()
        trainable = tf.trainable_variables()

        with tf.device('/cpu:0'):
            for var in trainable:
                tf.summary.histogram(var.op.name, var)

        print(raw_output)
        prediction = tf.reshape(raw_output, [-1, args.num_classes])

        print(prediction)
        with tf.device('/cpu:0'):
            input_batch = tf.squeeze(label_batch, squeeze_dims=[3])
            label_proc = tf.one_hot(input_batch, depth=args.num_classes)

        gt = tf.reshape(label_proc, [-1, args.num_classes])

        print('gt')
        print(gt)

        # Pixel-wise softmax loss.
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        reduced_loss = tf.reduce_mean(loss)

        with tf.device('/cpu:0'):
            tf.summary.scalar('loss', reduced_loss)

        # Define loss and optimisation parameters.
        optimiser = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        optim = optimiser.minimize(reduced_loss, var_list=trainable)

        with tf.device('/cpu:0'):
            merged = tf.summary.merge_all()

        with tf.device('/cpu:0'):
            iou_loss = tl.cost.iou_coe(raw_output, label_proc, axis=[1, 2, 3])
            tf.summary.scalar('iou_loss', iou_loss)

        # Set up tf session and initialize variables.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        train_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)

        with tf.device('/cpu:0'):
            # Saver for storing checkpoints of the model.
            saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=20)

            # Load variables if the checkpoint is provided.
            if args.restore_from is not None:
                loader = tf.train.Saver(var_list=restore_var)
                load(loader, sess, args.restore_from)

        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        start_epoch_time = 0
        total_loss, total_iou, n_batch = 0, 0, 0
        # Iterate over training steps.
        for step in range(args.num_steps):
            start_time = time.time()

            if step % args.save_pred_every == 0:
                loss_value, images, labels, IoU, _ = sess.run(
                    [reduced_loss, image_batch, label_batch, iou_loss, optim])

                total_loss += loss_value
                total_iou += IoU
                n_batch += 1

                # save mode to args.snapshot_dir
                save(saver, sess, args.snapshot_dir, step)
            else:
                loss_value, IoU, _ = sess.run([reduced_loss, iou_loss, optim])

                total_loss += loss_value
                total_iou += IoU
                n_batch += 1

            duration = time.time() - start_time
            print('step {:d} \t loss = {:.3f}, iou = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, iou_loss,
                                                                                       duration))

            if step % 100 == 0:
                summary_all = sess.run(merged)
                train_writer.add_summary(summary_all, step)

                print(" ** train_loss: %f iou: %f took %fs (2d with distortion)" %
                      (total_loss / n_batch, total_iou / n_batch, time.time() - start_epoch_time))

                total_loss, total_iou, n_batch = 0, 0, 0
                start_epoch_time = time.time()

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()

