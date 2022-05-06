import random
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as tf


class Augmentation:
    def __init__(self):
        pass

    def rotate(self, image, mask, angle=None):
        if angle == None:
            angle = transforms.RandomRotation.get_params([-10, 10])  # -180~180随机选一个角度旋转
        if isinstance(angle, list):
            angle = random.choice(angle)
        image = image.rotate(angle)
        mask = mask.rotate(angle)
        #image = tf.to_tensor(image)
        #mask = tf.to_tensor(mask)
        return image, mask

    def flip(self, image, mask):  # 水平翻转和垂直翻转
        if random.random() >= 0.5:
            image = tf.hflip(image)
            mask = tf.hflip(mask)
        # if random.random() <= 0.5:
        #     image = tf.vflip(image)
        #     mask = tf.vflip(mask)
        #image = tf.to_tensor(image)
        #mask = tf.to_tensor(mask)
        return image, mask

    def randomResizeCrop(self, image, mask, scale=(0.8, 1.2),
                         ratio=(1, 1)):  # scale表示随机crop出来的图片会在的0.8倍至1倍之间，ratio表示长宽比
        img = np.array(image)
        h_image, w_image = img.shape[1:3]
        resize_size = h_image
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
        image = tf.resized_crop(image, i, j, h, w, resize_size)
        mask = tf.resized_crop(mask, i, j, h, w, resize_size)
        #image = tf.to_tensor(image)
        #mask = tf.to_tensor(mask)
        return image, mask

    def adjustContrast(self, image, mask):
        factor = transforms.RandomRotation.get_params([0, 10])  # 这里调增广后的数据的对比度
        image = tf.adjust_contrast(image, factor)
        # mask = tf.adjust_contrast(mask,factor)
        #image = tf.to_tensor(image)
        #mask = tf.to_tensor(mask)
        return image, mask

    def adjustBrightness(self, image, mask):
        factor = transforms.RandomRotation.get_params([1, 2])  # 这里调增广后的数据亮度
        image = tf.adjust_brightness(image, factor)
        # mask = tf.adjust_contrast(mask, factor)
        #image = tf.to_tensor(image)
        #mask = tf.to_tensor(mask)
        return image, mask

    def centerCrop(self, image, mask, size=None):  # 中心裁剪
        if size == None: size = image.size  # 若不设定size，则是原图。
        image = tf.center_crop(image, size)
        mask = tf.center_crop(mask, size)
        #image = tf.to_tensor(image)
        #mask = tf.to_tensor(mask)
        return image, mask

    def adjustSaturation(self, image, mask):  # 调整饱和度
        factor = transforms.RandomRotation.get_params([1, 2])  # 这里调增广后的数据对比度
        image = tf.adjust_saturation(image, factor)
        # mask = tf.adjust_saturation(mask, factor)
        #image = tf.to_tensor(image)
        #mask = tf.to_tensor(mask)
        return image, mask


def augmentationData(img_path, mask_path, option=[0], save_dir=None):
    '''
    :param image_path: 图片的路径
    :param mask_path: mask的路径
    :param option: 需要哪种增广方式：1为旋转，2为翻转，3为随机裁剪并恢复原本大小，4为调整对比度，5为中心裁剪(不恢复原本大小)，6为调整亮度,7为饱和度
    :param save_dir: 增广后的数据存放的路径
    '''
    aug = Augmentation()
    image = Image.open(img_path)
    mask = Image.open(mask_path)
    if 1 in option:

        image, mask = aug.rotate(image, mask)

    if 2 in option:

        image, mask = aug.flip(image, mask)

    if 3 in option:

        image, mask = aug.randomResizeCrop(image, mask)

    if 4 in option:

        image, mask = aug.adjustContrast(image, mask)

    if 5 in option:

        image, mask = aug.centerCrop(image, mask)

    if 6 in option:

        image, mask = aug.adjustBrightness(image, mask)

    if 7 in option:

        image, mask = aug.adjustSaturation(image, mask)
    tran = transforms.ToTensor()
    image = tran(image)
    mask = tran(mask)
    #normMean = [0.706591, 0.580208, 0.534363]  # [180.18121, 147.95314, 136.26263]
    #normStd = [0.157334, 0.163555, 0.178653]  # [40.12028, 41.706684, 45.556572]
    normalize = transforms.Normalize(mean=[0.70704955, 0.58076555, 0.5346492], std=[0.15704633, 0.1643903, 0.17946811])
    image = normalize(image)
    return image, mask

