# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

def image_scaling(img, label):
    """
    Randomly scales the images between 0.5 to 1.5 times the original size.
    Args:
      img: Training image to scale.
      label: Segmentation mask to scale.
    """
    scale = tf.random_uniform([1], minval=0.9, maxval=1.1, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(img, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])

    return img, label


def image_mirroring(img, label):
    """
    Randomly mirrors the images.
    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
    """
    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    label = tf.reverse(label, mirror)
    return img, label

def random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w, ignore_label=0):
    """
    Randomly crop and pads the input images.

    Args:
      image: Training image to crop/ pad.
      label: Segmentation mask to crop/ pad.
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
      ignore_label: Label to ignore during the training.
    """

    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label          # Needs to be subtracted and later added due to 0 padding.
    combined = tf.concat(axis=2, values=[image, label])
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]), tf.maximum(crop_w, image_shape[1]))

    # last_image_dim = tf.shape(image)[-1]
    # last_label_dim = tf.shape(label)[-1]
    last_image_dim = image.get_shape().as_list()[-1]
    last_label_dim = label.get_shape().as_list()[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h,crop_w,last_image_dim+last_label_dim])
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)

    # Set static shape so that tensorflow knows shape at compile time.
    img_crop.set_shape((crop_h, crop_w, last_image_dim))
    label_crop.set_shape((crop_h,crop_w, last_label_dim))

    return img_crop, label_crop

def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.

    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    for line in f:
        try:
            image, mask = line.strip("\n").split(' ')
        except ValueError: # Adhoc for test.
            image = mask = line.strip("\n")
        images.append(data_dir + image)
        masks.append(data_dir + mask)
    return images, masks

def read_images_from_disk(input_queue, input_size, random_scale, random_mirror, ignore_label, img_mean): # optional pre-processing arguments
    """Read one image and its corresponding mask with optional pre-processing.
    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      random_mirror: whether to randomly mirror the images prior
                    to random crop.
      ignore_label: index of label to ignore during the training.
      img_mean: vector of mean colour values.

    Returns:
      Two tensors: the decoded image and its mask.
    """

    img_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])

    img = tf.image.decode_jpeg(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= img_mean

    label = tf.image.decode_png(label_contents, channels=1)

    if input_size is not None:
        h, w = input_size

        # Randomly scale the images and labels.
        if random_scale:
            img, label = image_scaling(img, label)

        # Randomly mirror the images and labels.
        if random_mirror:
            img, label = image_mirroring(img, label)

        # Randomly crops the images and labels.
        img, label = random_crop_and_pad_image_and_labels(img, label, h, w, ignore_label)

    return img, label

class ImageReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, data_list, input_size,
                 random_scale, random_mirror, ignore_label, img_mean, coord):
        '''Initialise an ImageReader.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          random_mirror: whether to randomly mirror the images prior to random crop.
          ignore_label: index of label to ignore during the training.
          img_mean: vector of mean colour values.
          coord: TensorFlow queue coordinator.
        '''
        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size
        self.coord = coord

        self.image_list, self.label_list = read_labeled_image_list(self.data_dir, self.data_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.labels],
                                                   shuffle=input_size is not None) # not shuffling if it is val
        self.image, self.label = read_images_from_disk(self.queue, self.input_size, random_scale, random_mirror, ignore_label, img_mean)

    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.

        Args:
          num_elements: the batch size.

        Returns:
          Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.'''
        image_batch, label_batch = tf.train.batch([self.image, self.label],
                                                  num_elements)
        return image_batch, label_batch


if __name__ == '__main__':
    # sess = tf.InteractiveSession()
    # img = np.ones([512,512])
    # img = tf.convert_to_tensor(img)
    # scale = tf.random_uniform([1], minval=0.3, maxval=1.2, dtype=tf.float32, seed=None)
    # h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    # w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    # dd = tf.stack([h_new, w_new])
    # new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    #
    # dd,scale, h_new, w_new, new_shape = sess.run([dd,scale, h_new, w_new, new_shape])
    # print(dd, scale, h_new, w_new, new_shape)
    # pass

    import matplotlib.pyplot as plt
    import dicom
    dicm_path = r'D:\PA14_H_6\DCE\DCM\IMG-0213-00015.dcm'
    dicm_array = dicom.read_file(dicm_path).pixel_array
    print('dicm_array shape :', dicm_array.shape)

    dicm_tensor = tf.convert_to_tensor(dicm_array)

    dicm_tensor_dims = tf.expand_dims(dicm_tensor, 2)
    dicm_tensor_dims = tf.image.convert_image_dtype(dicm_tensor_dims,tf.float32)
    # dicm_tensor_flip = tf.image.random_flip_left_right(dicm_tensor_dims)
    #随机缩放图像
    # image ,label = image_scaling(dicm_tensor_dims,dicm_tensor_dims)

    #随机镜像图像
    # image, label = image_mirroring(dicm_tensor_dims, dicm_tensor_dims)

    image, label = random_crop_and_pad_image_and_labels(dicm_tensor_dims, dicm_tensor_dims, 420, 420, 1)
    shape = image.get_shape().as_list()
    img = image[:,:,0]
    img_shape = img.get_shape().as_list()
    print('img_shape:',img_shape)
    print('shape:',shape)
    sess = tf.InteractiveSession()
    # path = r'D:\PA14_H_6\DCE\JPG_Marker\IMG-0214-00015.jpg'
    # image_con = tf.read_file(path)
    # image = tf.image.decode_jpeg(image_con)

    # image_ff = tf.expand_dims(image[:, :, 0], 2)
    # print(image_ff.eval().shape)
    #
    # image_flip = tf.image.random_flip_left_right(image_ff)

    # image_array = image.eval()
    # print(image_array.shape)

    image, label = sess.run([image,label])
    print('image shape :', image.shape)
    print('label shape :', label.shape)
    plt.figure(0)
    plt.imshow(dicm_array,cmap='gray')
    plt.figure(1)
    plt.imshow(np.squeeze(image),cmap='gray')
    plt.figure(2)
    plt.imshow(np.squeeze(label),cmap='gray')
    plt.show()
