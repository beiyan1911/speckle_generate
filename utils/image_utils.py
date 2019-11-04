"""This module contains simple helper functions """
from __future__ import print_function

import os

import numpy as np
import torch
from PIL import Image
from imageio import imsave

# todo: 解决mac 保存报错，详细原因？
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def tensor2im(input_image):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
    """
    if not isinstance(input_image, np.ndarray):
        image_numpy = input_image.cpu().detach().numpy()  # convert it into a numpy array
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def make_grid(array, nrow=8, padding=2, pad_value=0):
    assert len(np.shape(array)) == 4
    num, dim, h, w = np.shape(array)
    ncol = int(num // nrow)
    real_num = ncol * nrow
    if dim == 1:  # 灰度图
        result = np.ones((nrow * h + (nrow + 1) * padding, ncol * h + (ncol + 1) * padding), dtype=np.float32) * float(
            pad_value)

        for i in range(real_num):
            trow = i // ncol
            tcol = i % ncol

            img = array[i, 0]
            _min = np.min(img)
            _max = np.max(img)
            img = (img - _min) / (_max - _min)

            result[(trow + 1) * padding + trow * h:(trow + 1) * padding + (trow + 1) * h,
            (tcol + 1) * padding + tcol * w:(tcol + 1) * padding + (tcol + 1) * w] = img

    else:
        pass

    result = np.array(result * 255.0, dtype=np.uint8)
    return result


def __write_images(image_outputs, display_image_num, file_name):
    # image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs]  # expand gray-scale images to 3 channels
    image_tensor = np.concatenate([images[:display_image_num] for images in image_outputs], 0)

    result = make_grid(image_tensor, nrow=display_image_num, padding=10, pad_value=1)

    imsave(file_name, result)


def write_2images(image_outputs, display_image_num, image_directory, postfix):
    n = len(image_outputs)
    __write_images(image_outputs[0:n // 2], display_image_num, '%s/gen_a2b_%s.jpg' % (image_directory, postfix))
    __write_images(image_outputs[n // 2:n], display_image_num, '%s/gen_b2a_%s.jpg' % (image_directory, postfix))
