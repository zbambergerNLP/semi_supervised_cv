"""
Implement image augmentation functions used as part of MoCo.

A 224×224-pixel crop is taken from a randomly resized image, and then undergoes random color jittering,
random horizontal flip, and random grayscale conversion, all available in PyTorch’s torchvision package.
"""

import torch
import torchvision.transforms as transforms
from fastai.vision.all import *
from fastai.data.external import untar_data, URLs
from fastai.data.transforms import get_image_files

label_dict = dict(
    n01440764='tench',
    n02102040='English springer',
    n02979186='cassette player',
    n03000684='chain saw',
    n03028079='church',
    n03394916='French horn',
    n03417042='garbage truck',
    n03425413='gas pump',
    n03445777='golf ball',
    n03888257='parachute'
)




def augment_images(augmentation_coin_tosses, images, color_jitter_transform):
    """
    Apply augmentations y to image x based on the coin toss value in augmentation_coin_tosses[x][y].

    :param augmentation_coin_tosses: A tensor with shape [batch_size, 3] who's boolean values correspond to whether
        each augmentation should be applied onto the given image entry in the batch.
    :param images: A tensor with shape [batch_size, num_channels, height, width] corresponding to a batch of images.
    :param color_jitter_transform: A torchvision.transforms.ColorJitter object which can be used to apply the jitter
        transform onto an image tensor.
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
    return augmented_images


def augment(images,
            jitter_prob,
            horizontal_flip_prob,
            grayscale_conversion_prob,
            jitter_brightness=1,
            jitter_contrast=1,
            jitter_saturation=1,
            jitter_hue=0):
    """
    A cropped image undergoes augmentation through color jittering, random horizontal flip, and
    random grayscale conversion.

    :param images: A tensor with dimensions  [batch_size, channels, height, width] representing a batch of images.
    :param jitter_prob: A float in [0, 1] signifying the probability of applying a jittering augmentation to the image.
    :param horizontal_flip_prob: A float in [0, 1] signifying the probability of applying a horizontal flip to the
        inputted image.
    :param grayscale_conversion_prob: A float in [0, 1] signifying the probability of applying a grayscale conversion
        onto the inputted image.
    :param jitter_brightness: How much to jitter brightness. brightness_factor is chosen uniformly from
        [max(0, 1 - brightness), 1 + brightness] or the given [min, max]. Should be non negative numbers.
    :param contrast: How much to jitter contrast. contrast_factor is chosen uniformly from
        [max(0, 1 - contrast), 1 + contrast] or the given [min, max]. Should be non negative numbers.
    :param saturation: How much to jitter saturation. saturation_factor is chosen uniformly from
        [max(0, 1 - saturation), 1 + saturation] or the given [min, max]. Should be non negative numbers.
    :param hue: How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
        Should have 0= hue = 0.5 or -0.5 = min = max = 0.5.

    :return: A tensor of images with the same shape as the input tensor, but with some images augmented.
    """
    batch_size = images.shape[0]
    augmentation_coin_tosses = torch.bernoulli(
        torch.tensor([
            [jitter_prob, horizontal_flip_prob, grayscale_conversion_prob] for _ in range(batch_size)],
            dtype=torch.float))
    color_jitter_transform = transforms.ColorJitter(brightness=jitter_brightness,
                                                    contrast=jitter_contrast,
                                                    saturation=jitter_saturation,
                                                    hue=jitter_hue)

    return augment_images(augmentation_coin_tosses, images, color_jitter_transform)


def load_imagenette():
    """
    Return dataloaders for the imagenette dataset.
    """
    path = untar_data(URLs.IMAGENETTE_160)
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=label_func,
        splitter=GrandparentSplitter(),
        item_tfms=RandomResizedCrop(224, min_scale=0.35),
        batch_tfms=Normalize.from_stats(*imagenet_stats),
    )
    return dblock.dataloaders(path)


def label_func(fname):
    """
    Map a coded label name to its English description equivalent.

    :param fname: A string representing a file name.
    """
    return label_dict[parent_label(fname)]


if __name__ == "__main__":
    dls = load_imagenette()
    a = next(iter(dls))
    b = a.one_batch()[0]
    c = augment(b, 0.5, 0.5, 0.5)


