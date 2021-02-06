import numpy as np
from PIL import Image

import torchvision.transforms as transforms


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
    image_numpy = image_tensor[0].cpu().float().numpy()

    if image_numpy.shape[0] == 1:
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