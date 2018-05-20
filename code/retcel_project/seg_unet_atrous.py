from __future__ import print_function
import argparse
import os
import time
import tensorflow as tf
import numpy as np

try:
    from retcel_project.deeplab_resnet import UnetArtrous,prepare_label
except:
    from deeplab_resnet import UnetArtrous, prepare_label

try:
    from retcel_project.image_process import segment_image_process as img_seg
except:
    from image_process import segment_image_process as img_seg


d2_max_box = np.array([512, 512])

BATCH_SIZE = 6
DATA_DIRECTORY = '/data/b/wangguangyuan/rectal_segment/DCE/tfrecord/dce_train.tfrecord'   #tfrecord data path
IGNORE_LABEL = 0          # ignore label
INPUT_SIZE = '512,512'   #input size
LEARNING_RATE = 1e-4      #learning rate
NUM_CLASSES = 2           #class of segmnt
NUM_STEPS = 40000         #all iterate steps

RESTORE_FROM = None
SAVE_NUM_IMAGES = 1
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = '/data/b/wangguangyuan/rectal_segment/DCE/tfrecord/myown_checkpoint/unet_atrous/'
DEVICE='/gpu:8'


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
            image_batch, label_batch = img_seg.distored_inputs(args.data_dir, d2_max_box,
                                                               args.batch_size,
                                                               input_size=[height,width],
                                                               random_scale=True,
                                                               random_mirror=True,
                                                               ignore_label=0)

            duration = time.time() - start_time
            print('read data took %f sec' % duration)

    with tf.device(DEVICE):
        # Create network.
        net = UnetArtrous({'data': image_batch}, is_training=args.is_training, num_classes=args.num_classes)
        # For a small batch size, it is better to keep
        # the statistics of the BN layers (running means and variances)
        # frozen, and to not update the values provided by the pre-trained model.
        # If is_training=True, the statistics will be updated during the training.
        # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
        # if they are presented in var_list of the optimiser definition.

        # Predictions.
        raw_output = net.layers['uconv1']
        # Which variables to load. Running means and variances are not trainable,
        # thus all_variables() should be restored.
        # Restore all variables, or all except the last ones.

        restore_var = tf.global_variables()
        trainable = tf.trainable_variables()                   #train model
        # trainable = [v for v in tf.trainable_variables() if 'fc1_voc12' in v.name]  # Fine-tune only the last layers.

        print(raw_output)
        prediction = tf.reshape(raw_output, [-1, args.num_classes])

        print(prediction)

        with tf.device('/cpu:0'):
            label_proc = prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]), num_classes=args.num_classes)

        print(label_proc)

        gt = tf.reshape(label_proc, [-1, args.num_classes])

        print('gt')
        print(gt)

        # Pixel-wise softmax loss.
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        reduced_loss = tf.reduce_mean(loss)

        # Processed predictions.
        raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3, ])
        raw_output_up = tf.argmax(raw_output_up, dimension=3)
        pred = tf.expand_dims(raw_output_up, dim=3)

        # Define loss and optimisation parameters.
        optimiser = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        optim = optimiser.minimize(reduced_loss, var_list=trainable)

        # Set up tf session and initialize variables.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
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

        # Iterate over training steps.
        for step in range(args.num_steps):
            start_time = time.time()

            if step % args.save_pred_every == 0:
                loss_value, images, labels, preds, _ = sess.run(
                    [reduced_loss, image_batch, label_batch, pred, optim])

                #save mode to args.snapshot_dir
                save(saver, sess, args.snapshot_dir, step)
            else:
                loss_value, _ = sess.run([reduced_loss, optim])
            duration = time.time() - start_time
            print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()
