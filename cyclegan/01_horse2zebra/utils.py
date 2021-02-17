import copy
import os
import random

import numpy as np
import torch
from PIL import Image

import torchvision.transforms as transforms

__all__ = ['create_data', 'tensor2im', 'save_image', 'save_network', 'ImagePool']

from torch.autograd import Variable


def create_data(infile):
    """ Load image file & Convert to tensor

    :param infile: Target file path
    :return: Single image data as Tensor format
    """
    raw_img = Image.open(infile)
    tensor_image = _transform_image(raw_img)
    return tensor_image


def _transform_image(img, img_size=286):
    """Transform image

    :param img: numpy or pillow format image data
    :param img_size: size of input
    :return: tensor image data which is able to be predicted
    """
    input_transform = transforms.Compose([
        transforms.Resize(img_size, Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    timg = input_transform(img)
    timg.unsqueeze_(0)
    return timg


def tensor2im(input_image, imtype=np.uint8):
    """Convert tensor to numpy

    :param input_image: tensor image
    :param imtype: output format
    :return: numpy image
    """
    image_tensor = input_image.data
    image_numpy = image_tensor.cpu().float().numpy()

    if image_numpy.shape == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0

    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    """Convert pillow format & save as file

    :param image_numpy: image as numpy
    :param image_path: output file path
    :return: None
    """
    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    image_pil.save(image_path)


def save_network(network, network_label, epoch_label, log_dir='./checkpoints'):
    save_filename = "%s_net_%s.pth" % (epoch_label, network_label)
    save_path = os.path.join(log_dir, save_filename)
    torch.save(copy.deepcopy(network).cpu().state_dict(), save_path)


class ImagePool(object):
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return Variable(images)

        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs += 1
                self.images.append(image)
                return_images.append(image)

            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)

                else:
                    return_images.append(image)

        return_images = Variable(torch.cat(return_images, 0))
        return return_images


