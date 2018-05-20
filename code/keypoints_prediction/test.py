"""Training script for the DeepLab-ResNet network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC,
which contains approximately 10000 images for training and 1500 images for validation.
"""


# import tensorflow as tf
# from deeplab_resnet import DeepLabResNetModel
# import numpy as np

# data=np.ones([10,321,321,3])
# image_batch=tf.convert_to_tensor(data)
#
# net = DeepLabResNetModel({'data': image_batch}, is_training=True, num_classes=15)

from skimage import io


path = r'D:\AIchanger\data\train\human_images\0000252aea98840a550dac9a78c476ecb9f47ffa_human2.jpg'


image_data = io.imread(path)

pass