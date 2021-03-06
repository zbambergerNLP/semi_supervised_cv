"""
Implement image augmentation functions used as part of MoCo.

A 224×224-pixel crop is taken from a randomly resized image, and then undergoes random color jittering,
random horizontal flip, and random grayscale conversion, all available in PyTorch’s torchvision package.
"""

import consts
import torch
import torchvision.transforms as transforms
from fastai.vision.all import *
from fastai.data.external import untar_data, URLs
from fastai.data.transforms import get_image_files


def augment_images(augmentation_coin_tosses, images, color_jitter_transform, gaussian_blur_transform):
    """
    Apply augmentations y to image x based on the coin toss value in augmentation_coin_tosses[x][y].

    :param augmentation_coin_tosses: A tensor with shape [batch_size, 4] who's boolean values correspond to whether
        each augmentation should be applied onto the given image entry in the batch.
    :param images: A tensor with shape [batch_size, num_channels, height, width] corresponding to a batch of images.
    :param color_jitter_transform: A torchvision.transforms.ColorJitter object which can be used to apply the jitter
        transform onto an image tensor.
    :param gaussian_blur_transform: A torchvision.transforms.GaussianBlur object which can be used to apply the
        Gaussian Blur transform onto an image tensor.
    :return: A tensor of images with the same shape as the input images, but with augmentations applied according to
        Bernoulli coin tosses.
    """
    augmented_images = torch.clone(images)
    for index, image in enumerate(images):
        if augmentation_coin_tosses[index][0]:
            augmented_images[index] = color_jitter_transform(images[index])
        if augmentation_coin_tosses[index][1]:
            augmented_images[index] = transforms.functional.hflip(images[index])
        if augmentation_coin_tosses[index][2]:
            augmented_images[index] = transforms.functional.invert(images[index])
        if augmentation_coin_tosses[index][3]:
            augmented_images[index] = gaussian_blur_transform(images[index])
    return augmented_images


def augment(images,
            jitter_prob,
            horizontal_flip_prob,
            grayscale_conversion_prob,
            gaussian_blur_prob,
            jitter_brightness=1,
            jitter_contrast=1,
            jitter_saturation=1,
            jitter_hue=0,
            kernel_size=5,
            sigma=(0.1, 2.0)):
    """
    A cropped image undergoes augmentation through color jittering, random horizontal flip, and
    random grayscale conversion.

    :param images: A tensor with dimensions  [batch_size, channels, height, width] representing a batch of images.
    :param jitter_prob: A float in [0, 1] signifying the probability of applying a jittering augmentation to the image.
    :param horizontal_flip_prob: A float in [0, 1] signifying the probability of applying a horizontal flip to the
        inputted image.
    :param grayscale_conversion_prob: A float in [0, 1] signifying the probability of applying a grayscale conversion
        onto the inputted image.
    :param gaussian_blur_prob: A float in [0, 1] signifying the probability of applying Gaussian blur onto the
        inputted image.
    :param jitter_brightness: How much to jitter brightness. brightness_factor is chosen uniformly from
        [max(0, 1 - brightness), 1 + brightness] or the given [min, max]. Should be non negative numbers.
    :param contrast: How much to jitter contrast. contrast_factor is chosen uniformly from
        [max(0, 1 - contrast), 1 + contrast] or the given [min, max]. Should be non negative numbers.
    :param saturation: How much to jitter saturation. saturation_factor is chosen uniformly from
        [max(0, 1 - saturation), 1 + saturation] or the given [min, max]. Should be non negative numbers.
    :param hue: How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
        Should have 0= hue = 0.5 or -0.5 = min = max = 0.5.
    :param kernel_size: Size of the Gaussian kernel used for Gaussian Blur augmentation.
    :param sigma: Standard deviation to be used for creating kernel to perform blurring. If float, sigma is fixed. If
        it is tuple of float (min, max), sigma is chosen uniformly at random to lie in the given range.

    :return: A tensor of images with the same shape as the input tensor, but with some images augmented.
    """
    batch_size = images.shape[0]
    augmentation_coin_tosses = torch.bernoulli(
        torch.tensor([
            [jitter_prob, horizontal_flip_prob, grayscale_conversion_prob, gaussian_blur_prob] for _ in range(batch_size)],
            dtype=torch.float))
    color_jitter_transform = transforms.ColorJitter(brightness=jitter_brightness,
                                                    contrast=jitter_contrast,
                                                    saturation=jitter_saturation,
                                                    hue=jitter_hue)
    gaussian_blur_transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    return augment_images(augmentation_coin_tosses, images, color_jitter_transform, gaussian_blur_transform)


def label_func(fname):
    """
    Map a coded label name to its English description equivalent.

    :param fname: A string representing a file name.
    """
    return consts.IMAGENETTE_LABEL_DICT[parent_label(fname)]
