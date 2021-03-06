from torch.utils.data import Dataset
import os
import torch
import numpy as np
from skimage import io, transform

import consts
from augment import label_func


def create_csv_file(dir, filename):
    """
    :param dir: The directory in which the csv file corresponding to the dataset. Each entry in the CSV corresponds to
        the path of a file from the dataset.
    :param filename: The name of the CSV file we are creating.
    """
    # Open the file in the write mode
    with open(filename, 'w') as fileHandle:
        for d in os.listdir(dir):
            d_full_path = os.path.join(dir, d)
            if os.path.isdir(d_full_path) and d in consts.IMAGENETTE_LABEL_DICT:
                for path in os.listdir(d_full_path):
                    full_path = os.path.join(d_full_path, path)
                    if os.path.isfile(full_path):

                        fileHandle.write(f'{full_path}\n')
                        print(f'{full_path}\n')


class Rescale(object):
    """Rescale the image in a sample to a given size"""

    def __init__(self, output_size):
        """
        :param output_size: (tuple or int) Desired output size. If tuple, output is matched to output_size. If int,
            smaller of image edges is matched to output_size keeping aspect ratio the same.
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        """Perform a Rescale operation on an image, converting it to the desired height and width.

        :param sample: A sample image tensor of shape [height, width, ...] on which to perform Rescaling.
        :return: A rescaled image tensor with the desired height and width (specified by `self.output_size`).
        """
        image = sample

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return img


class RandomCrop(object):
    """Crop randomly the image in a sample."""

    def __init__(self, output_size):
        """
        :param output_size: (tuple or int) Desired output size. If int, square crop is made.
        """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        """
        :param sample: A sample image tensor of shape [height, width, ...] on which to perform Rescaling.
        :return: A crop image tensor that is taken from the original image tensor. The height and width of the produced
            crop are specified by `self.output_size`.
        """
        image = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        """
        :param sample: An image that is either grayscale or RGB.
        :return: A torch tensor version of the original image such that its dimensions are [C, H, W]. C contains the RGB
            channel values, while H and W represent height and width respectively.
        """
        image = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
            image = np.repeat(image, 3, axis=2)

        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)


class ImagenetteDataset(Dataset):
    """Imagenette dataset."""

    def __init__(self,  root_dir, csv_file, transform=None, labels=False, debug=False):
        """
        :param root_dir: The string directory with all the images.
        :param csv_file: String path to the csv file with paths to all the images.
        :param transform: (callable, optional) transform to be applied on a sample.
        :param labels: True if the dataset should contain integer labels to represent the type of content of the image.
            False if the dataset should consist strictly of the images, without their labels.
        :param debug: True if we are training in an easy-debug mode where training and evaluation must run quickly.
            False otherwise.
        """
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.transform = transform
        self.labels = labels
        self.debug = debug
        self.translate_labels = {'tench': 0,
                                 'English springer': 1,
                                 'cassette player': 2,
                                 'chain saw': 3,
                                 'church': 4,
                                 'French horn': 5,
                                 'garbage truck': 6,
                                 'gas pump': 7,
                                 'golf ball': 8,
                                 'parachute': 9}

        with open(csv_file, newline='') as f:
            self.paths_to_images = f.read().splitlines()

    def __len__(self):
        if self.debug:
            return 16
        else:
            return len(self.paths_to_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = io.imread(self.paths_to_images[idx])

        if self.transform:
            image = self.transform(image)

        if self.labels:
            tag = self.translate_labels[label_func(self.paths_to_images[idx])]
            return image, tag
        else:
            return image


if __name__ == '__main__':
    print(consts.image_dir)
    print(consts.csv_filename)
    create_csv_file(dir=consts.image_dir, filename=consts.csv_filename)
