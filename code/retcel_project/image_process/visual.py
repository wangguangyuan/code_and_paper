# -*- coding: utf-8 -*-

import scipy.misc
import numpy as np
try:
    from retcel_project.image_process import image_process as img_pro
except:
    import image_process as img_pro

def save_image(image, image_path=''):
    """Save one image.

    Parameters
    -----------
    images : numpy array [w, h, c]
    image_path : string.
    """
    try: # RGB
        scipy.misc.imsave(image_path, image)
    except: # Greyscale
        scipy.misc.imsave(image_path, image[:,:,0])
        
def save_images(images, size, image_path=''):
    """Save mutiple images into one single image.

    Parameters
    -----------
    images : numpy array [batch, w, h, c]
    size : list of two int, row and column number.
        number of images should be equal or less than size[0] * size[1]
    image_path : string.

    Examples
    ---------
    >>> images = np.random.rand(64, 100, 100, 3)
    >>> tl.visualize.save_images(images, [8, 8], 'temp.png')
    """
    def merge(images, size):
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j*h:j*h+h, i*w:i*w+w, :] = image
        return img
    def imsave(images, size, path):
        return scipy.misc.imsave(path, merge(images, size))

    assert len(images) <= size[0] * size[1], "number of images should be equal or less than size[0] * size[1] {}".format(len(images))
    return imsave(images, size, image_path)

def vis_imgs(X, y, path):
    """ show one slice """
    if y.ndim == 2:
        y = y[:,:,np.newaxis]
    assert X.ndim == 3
    save_images(np.asarray([X[:,:,0,np.newaxis],y]), size=(1, 2),
        image_path=path)

def vis_imgs2(X, y_, y, path):
    """ show one slice with target """
    if y.ndim == 2:
        y = y[:,:,np.newaxis]
    if y_.ndim == 2:
        y_ = y_[:,:,np.newaxis]
    assert X.ndim == 3
    save_images(np.asarray([X[:,:,0,np.newaxis], y_, y]), size=(1, 3),
        image_path=path)
    
if __name__=='__main__':
    data, label, bound_box = img_pro.read_pkl(r'D:\orignal datt\liver orignal\orignal\MRI_H_1\PA0_H_1\DWI')
    maskers=img_pro.read_masker_data(r'D:\orignal datt\liver orignal\orignal\MRI_H_1\PA0_H_1\DWI')
    img_pro.save_maskers(r'D:\after_process_data\liver\maskers',maskers,'maskers','maskers')
    maskers=maskers[:,:,:,np.newaxis]
    dat=maskers[9:13,:,:,:]
    save_images(dat,(1,6),r'D:\after_process_data\liver\test.png')
    vis_imgs(dicmm,da,r'D:\after_process_data\liver\te.png')