# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 18:16:29 2017

@author: wgy
"""

image_bmp_path=r'C:\User\wangguangyuan\python\retcel_3d\data\DWI\BMP\IMG-0012-00009.bmp'
image_bmp_masker_path=r'C:\User\wangguangyuan\python\retcel_3d\data\DWI\BMP_Marker\IMG-0014-00009.bmp'
image_dicm_path=r'C:\User\wangguangyuan\python\retcel_3d\data\DWI\DCM\IMG-0013-00009.dcm'

image_data=cv2.imread(image_bmp_path)
image_data_masker=cv2.imread(image_bmp_masker_path)
image_dicm=dicom.read_file(image_dicm_path)
image_mm=image_data_masker-image_data

#image_data=imread(image_bmp_path)
#image_data_masker=imread(image_bmp_masker_path)

image_gray=color.rgb2gray(image_data)
image_gray_masker=color.rgb2gray(image_data_masker)

image_masker=image_gray_masker-image_gray

#image_mm=image_data_masker-image_data
#image_bi=morphology.convex_hull_image(image_mm[:,:,2])
#contours, hierarchy = cv2.findContours(image_bi.astype(np.int),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)

im_th=image_mm[:,:,2]
im_th=im_th.astype(np.uint8)
im_out=fillholse(im_th)
plt.imshow(im_out,cmap=plt.cm.gray)

plt.show()

path=r'./data'


_,second_path,_=read_files.get_files_path(path)
dicm_array,maskers_out=img_pro.extract_one_patient_roi(second_path[0],
                                                       ["BMP","BMP_Marker","DCM"])

img_pro.save_maskers(second_path[0],maskers_out,'data','dat')
bounding_box=img_pro.find_border(maskers_out)
img_pro.plot_3d(maskers_out,0)