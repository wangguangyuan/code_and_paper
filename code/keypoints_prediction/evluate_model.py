import argparse
import os
import tensorflow as tf
from deeplab_resnet import DeepLabResNetModel, prepare_label
import tensorlayer as tl
from multiprocess_data import get_one_minibatch, data_augument
import numpy as np
from scipy import misc

BATCH_SIZE = 4
DATA_LIST_PATH = r'D:\AIchanger\data\train\train.txt'
NUM_CLASSES = 15
NUM_STEPS = 20001
RESTORE_FROM = r'D:\AIchanger\keypoints_prediction\restore_checkpoints\model.ckpt-19999'


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

    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")

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



    with tf.device('/gpu:0'):
        #create network
        net = DeepLabResNetModel({'data': image_batch},
                                 is_training=False,
                                 num_classes=args.num_classes)

        raw_output = net.layers['fc1_voc12']
        restore_var = tf.global_variables()

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


        #Processed predictions
        raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3, ])
        raw_output_upp = tf.argmax(raw_output_up, dimension=3)
        pred = tf.expand_dims(raw_output_upp, dim=3)

        with tf.device('/cpu:0'):
            input_batch = tf.squeeze(label_batch, squeeze_dims=[3])
            label_one_hot = tf.one_hot(input_batch, depth=args.num_classes)
        with tf.device('/cpu:0'):
            iou_loss = tl.cost.iou_coe(raw_output_up, label_one_hot, axis=[1, 2, 3])

        # Set up tf session and initialize variables.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()

        sess.run(init)

        with tf.device('/cpu:0'):
            # Load variables if the checkpoint is provided.
            if args.restore_from is not None:
                loader = tf.train.Saver(var_list=restore_var)
                load(loader, sess, args.restore_from)

        x_train, y_train, shapes = get_one_minibatch(args.data_list, args.batch_size, data_augument, True)
        feed_dict = {image_batch: x_train, label_batch: y_train}


        loss_value, IoU, pred_maskers = sess.run([reduced_loss, iou_loss, pred], feed_dict=feed_dict)
        print('loss:{}, IoU: {}\n'.format(loss_value, IoU))
        print('pred_masker shape: ', pred_maskers.shape)
        for num, masker in enumerate(pred_maskers):
            pred_masker = np.squeeze(masker)
            orignal_masker = np.squeeze(y_train[num])

            misc.imsave(r'D:\tmp\{}_orignal_masker.jpg'.format(num), orignal_masker*255)
            misc.imsave(r'D:\tmp\{}_pred_masker.jpg'.format(num), pred_masker * 500)


if __name__ == '__main__':
    main()
