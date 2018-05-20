import argparse
import os
import time
import tensorflow as tf
from deeplab_resnet import DeepLabResNetModel, prepare_label
import tensorlayer as tl
from multiprocess_data import Multiprocess_Data


BATCH_SIZE = 8
DATA_LIST_PATH = r'D:\AIchanger\data\train\train.txt'
LEARNING_RATE = 1e-5
MOMENTUM = 0.9
NUM_CLASSES = 15
NUM_STEPS = 20001
POWER = 0.9
RESTORE_FROM = r'D:\AIchanger\keypoints_prediction\restore_checkpoints\model.ckpt-19999'
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = './myowncheckpoint/'
WEIGHT_DECAY = 0.0005
tensorboard_path = './mymodel_tensorbaord/'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")

    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")

    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")

    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")

    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")

    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")

    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")

    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")

    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")

    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")

    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")

    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    return parser.parse_args()


def save(saver, sess, logdir, step):
    '''Save weights.
    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      logdir: path to the snapshots directory.
      step: current training step.
    '''

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
        args = get_arguments()

        with tf.name_scope('create_inputs'):
            image_batch = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, 256, 256, 3])
            label_batch = tf.placeholder(dtype=tf.uint8, shape=[args.batch_size, 256, 256, 1])


    # 定义多进程读取数据
    multi_process = Multiprocess_Data(data_path=args.data_list,
                                      batch_size=args.batch_size,
                                      capacity=100,
                                      num_threads=3)

    with tf.device('/gpu:0'):
        #create network
        net = DeepLabResNetModel({'data': image_batch},
                                 is_training=True,
                                 num_classes=args.num_classes)

        raw_output = net.layers['fc1_voc12']
        restore_var = tf.global_variables()
        trainable = tf.trainable_variables()

        with tf.device('/cpu:0'):
            for var in trainable:
                tf.summary.histogram(var.op.name, var)

        print('raw_outout:', raw_output)

        prediction = tf.reshape(raw_output, [-1, args.num_classes])
        print('prediction:', prediction)

        with tf.device('/cpu:0'):
            label_proc = prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]), num_classes=args.num_classes)

        print('label_proc:', label_batch)

        gt = tf.reshape(label_proc, [-1, args.num_classes])

        print('gt:', gt)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        reduced_loss = tf.reduce_mean(loss)
        with tf.device('/cpu:0'):
            tf.summary.scalar('loss', reduced_loss)

        #Processed predictions
        raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3, ])
        with tf.device('/cpu:0'):
            input_batch = tf.squeeze(label_batch, squeeze_dims=[3])
            label_one_hot = tf.one_hot(input_batch, depth=args.num_classes)
        with tf.device('/cpu:0'):
            iou_loss = tl.cost.iou_coe(raw_output_up, label_one_hot, axis=[1, 2, 3])
            tf.summary.scalar('iou_loss', iou_loss)

        optimiser = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        optim = optimiser.minimize(reduced_loss, var_list=trainable)

        with tf.device('/cpu:0'):
            merged = tf.summary.merge_all()

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

        start_epoch_time = 0
        total_loss, total_iou, n_batch = 0, 0, 0

        # Iterate over training steps.
        for step in range(args.num_steps):
            start_time = time.time()

            x_train, y_train = multi_process.shuffle_batch()
            feed_dict = {image_batch: x_train, label_batch: y_train}

            if (step + 1) % args.save_pred_every == 0:
                loss_value, IoU, _ = sess.run([reduced_loss, iou_loss, optim], feed_dict=feed_dict)
                total_loss += loss_value
                total_iou += IoU
                n_batch += 1

                # save mode to args.snapshot_dir
                save(saver, sess, args.snapshot_dir, step)
            else:
                loss_value, IoU, _ = sess.run([reduced_loss, iou_loss, optim], feed_dict=feed_dict)
                total_loss += loss_value
                total_iou += IoU
                n_batch += 1

            duration = time.time() - start_time
            print('step {:d} \t loss = {:.3f}, iou = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, IoU,
                                                                                       duration))

            if step % 100 == 0:
                summary_all = sess.run(merged,feed_dict=feed_dict)
                train_writer.add_summary(summary_all, step)
                print(" ** train_loss: %f iou: %f took %fs (2d with distortion)" %
                      (total_loss / n_batch, total_iou / n_batch, time.time() - start_epoch_time))

                total_loss, total_iou, n_batch = 0, 0, 0
                start_epoch_time = time.time()


if __name__ == '__main__':
    main()

