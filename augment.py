"""
Implement image augmentation functions used as part of MoCo.

A 224×224-pixel crop is taken from a randomly resized image, and then undergoes random color jittering,
random horizontal flip, and random grayscale conversion, all available in PyTorch’s torchvision package.
"""

import torch
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


def augment(image):
    # Questions:
    # 1. With what probability to we resize an image before taking a 224x224 pixel crop from it?
    # 2. If we choose to resize an image before taking a random crop from it, how do we parameterize the resize
    # operation?
    # 3.
    pass


def load_imagenette():
    path = untar_data(URLs.IMAGENETTE_160)
    files = get_image_files(path)
    fnames = get_image_files(path)
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=label_func,
        splitter=GrandparentSplitter(),
        item_tfms=RandomResizedCrop(224, min_scale=0.35),
        batch_tfms=Normalize.from_stats(*imagenet_stats),
    )
    dls = dblock.dataloaders(path)


def label_func(fname):
    return label_dict[parent_label(fname)]


if __name__ == "__main__":
    load_imagenette()

