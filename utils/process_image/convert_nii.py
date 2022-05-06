import numpy as np
import os  # 遍历文件夹
import nibabel as nib  # nii格式一般都会用到这个包
import imageio  # 转换成图像
import SimpleITK as sitk
import cv2
import pydicom


def nii_to_image(filepath):
    filenames = os.listdir(filepath)  # 读取nii文件夹
    slice_trans = []

    for f in filenames:
        # 开始读取nii文件
        img_path = os.path.join(filepath, f)
        img = nib.load(img_path)  # 读取nii
        img_fdata = img.get_fdata()
        fname = f.replace('.nii', '')  # 去掉nii的后缀名
        img_f_path = os.path.join(imgfile, fname)
        # 创建nii对应的图像的文件夹
        # if not os.path.exists(img_f_path):
        #     os.mkdir(img_f_path)  # 新建文件夹


        # 开始转换为图像
        (x, y, z) = img.shape
        for i in range(z):  # z是图像的序列
            silce = img_fdata[:, :, i]  # 选择哪个方向的切片都可以
            # imageio.imwrite(os.path.join(img_f_path, '{}.png'.format(i)), silce)
            imageio.imwrite(img_f_path+'-{}.png'.format(i), silce)
            # 保存图像


def dcm2png_single(dcm_path):
    save_pa = r'./png/single'
    if not os.path.exists(save_pa):
        os.makedirs(save_pa)
    img_name = os.path.split(dcm_path.replace('.dcm', '.png'))  # 替换后缀名并分离路径
    img_name = img_name[-1]
    ds = sitk.ReadImage(dcm_path, force=True)       # 注意这里，强制读取
    img = ds.pixel_array  # 提取图像信息
    cv2.imwrite(os.path.join(save_pa, img_name), img)

if __name__ == '__main__':
    # filepath = os.listdir('originImage')
    filepath = './datasets/50case-COVID-19/全部/train/image'
    imgfile = './datasets/50case-COVID-19/全部/train/image01'
    nii_to_image(filepath)
