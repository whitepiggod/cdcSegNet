# import SimpleITK
# import numpy
# import cv2
#
# '''
# SimpleITK读取.dcm图像的基本操作
# 并将dicom格式图像转换为jpg格式的图片
# '''
#
# path = './xxxxx/1-1.dcm'
#
# itk_img = SimpleITK.ReadImage(path)
# img_array = SimpleITK.GetArrayFromImage(itk_img)
#
# print("-" * 60)
# print("Img array: ", img_array.shape)  # 读取图像大小
#
# print("-" * 60)
# origin = numpy.array(itk_img.GetOrigin())  # 读取图像原点坐标
# print("Origin (x,y,z): ", origin)
#
# print("-" * 60)
# direction = numpy.array(itk_img.GetDirection())  # 读取图像方向
# print("Direction: ", direction)
#
# print("-" * 60)
# spacing = numpy.array(itk_img.GetSpacing())  # 读取图像尺度信息
# print("Spacing (x,y,z): ", spacing)
#
# print("-" * 60)
# cv2.imwrite("dicom.jpg", img_array[0, :, :])  # dicom格式的图像转化为jpg格式的
#
# # showimg = img_array[0, :, :]
# # cv2.imshow("showimag", showimg)
# # cv2.waitKey()

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
image = sitk.ReadImage("./xxxxx/1-040.dcm")
image_array = np.squeeze(sitk.GetArrayFromImage(image))
plt.imshow(image_array)
plt.show()
