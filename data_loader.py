from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np
from skimage import io, transform
from torchvision import transforms, utils

# import consts_noam as consts
import consts_noam as consts


def create_csv_file(dir, filename):
    # Open the file in the write mode
    with open(filename, 'w') as fileHandle:
        for d in os.listdir(dir):
            d_full_path = os.path.join(dir, d)
            if os.path.isdir(d_full_path):
                for path in os.listdir(d_full_path):
                    full_path = os.path.join(d_full_path, path)
                    if os.path.isfile(full_path):

                        fileHandle.write(f'{full_path}\n')
                        print(f'{full_path}\n')


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
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
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
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
        image = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
            image = np.repeat(image,3,axis=2)

        image = image.transpose((2, 0, 1))
        return  torch.from_numpy(image)


class ImagenetteDataset(Dataset):
    """Imagenette dataset."""

    def __init__(self,  root_dir, csv_file, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with paths to all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.

        """
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.transform = transform

        with open(csv_file, newline='') as f:
            self.paths_to_images = f.read().splitlines()

    def __len__(self):
        return len(self.paths_to_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = io.imread(self.paths_to_images[idx])

        if self.transform:
            image = self.transform(image)

        return image


if __name__ == '__main__':
    print(consts.image_dir)
    print(consts.csv_filename)
    create_csv_file(dir=consts.image_dir, filename=consts.csv_filename)
