import os
import random

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


__all__ = ['UnalignedDataset']


class UnalignedDataset(Dataset):
    def __init__(self, is_train, load_size=286, fine_size=256):
        root_dir = 'datasets/monet2photo'

        self.load_size = load_size
        self.fine_size = fine_size

        if is_train:
            dir_A = os.path.join(root_dir, 'trainA')
            dir_B = os.path.join(root_dir, 'trainB')
        else:
            dir_A = os.path.join(root_dir, 'testA')
            dir_B = os.path.join(root_dir, 'testB')

        self.image_paths_A = self._make_dataset(dir_A)
        self.image_paths_B = self._make_dataset(dir_B)

        self.size_A = len(self.image_paths_A)
        self.size_B = len(self.image_paths_B)

        self.transform = self._make_transform(is_train)

    def __getitem__(self, index):
        index_A = index % self.size_A
        path_A = self.image_paths_A[index_A]

        index_B = random.randint(0, self.size_B - 1)
        path_B = self.image_paths_B[index_B]

        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('RGB')

        A = self.transform(img_A)
        B = self.transform(img_B)

        return {'A': A, 'B': B, 'path_A': path_A, 'path_B': path_B}

    def __len__(self):
        return max(self.size_A, self.size_B)

    def _make_dataset(self, dir):
        images = []
        for fname in os.listdir(dir):
            if fname.endswith('.jpg'):
                path = os.path.join(dir, fname)
                images.append(path)
        sorted(images)

        return images

    def _make_transform(self, is_train):
        transforms_list = []
        transforms_list.append(transforms.Resize((self.load_size, self.load_size), Image.BICUBIC)),
        transforms_list.append(transforms.RandomCrop(self.fine_size))
        if is_train:
            transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        return transforms.Compose(transforms_list)