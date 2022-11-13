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
            angle = transforms.RandomRotation.get_params([-10, 10])  # -180~180 randomly select an angle for rotation
        if isinstance(angle, list):
            angle = random.choice(angle)
        image = image.rotate(angle)
        mask = mask.rotate(angle)
        #image = tf.to_tensor(image)
        #mask = tf.to_tensor(mask)
        return image, mask

    def flip(self, image, mask):  # Flip Horizontal and Flip Vertical
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
                         ratio=(1, 1)):  # Scale indicates that the image randomly cropped will be between 0.8 and 1 times of, and ratio indicates the aspect ratio
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
        factor = transforms.RandomRotation.get_params([0, 10])  # Here is the contrast of the augmented data
        image = tf.adjust_contrast(image, factor)
        # mask = tf.adjust_contrast(mask,factor)
        #image = tf.to_tensor(image)
        #mask = tf.to_tensor(mask)
        return image, mask

    def adjustBrightness(self, image, mask):
        factor = transforms.RandomRotation.get_params([1, 2])  # Here, the brightness of the adjusted data
        image = tf.adjust_brightness(image, factor)
        # mask = tf.adjust_contrast(mask, factor)
        #image = tf.to_tensor(image)
        #mask = tf.to_tensor(mask)
        return image, mask

    def centerCrop(self, image, mask, size=None):  # Center clipping
        if size == None: size = image.size  # If size is not set, it is the original image.
        image = tf.center_crop(image, size)
        mask = tf.center_crop(mask, size)
        #image = tf.to_tensor(image)
        #mask = tf.to_tensor(mask)
        return image, mask

    def adjustSaturation(self, image, mask):  # Adjust Saturation
        factor = transforms.RandomRotation.get_params([1, 2])  # Here is the adjusted data contrast
        image = tf.adjust_saturation(image, factor)
        # mask = tf.adjust_saturation(mask, factor)
        #image = tf.to_tensor(image)
        #mask = tf.to_tensor(mask)
        return image, mask


def augmentationData(img_path, mask_path, option=[0], save_dir=None):
    '''
    :param image_path:
    :param mask_path:
    :param option: Which augmentation method is required: 1 is rotation, 2 is flipping, 3 is random clipping and restoring the original size, 4 is adjusting contrast, 5 is central clipping (not restoring the original size), 6 is adjusting brightness, 7 is saturation
    :param save_dir:
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

