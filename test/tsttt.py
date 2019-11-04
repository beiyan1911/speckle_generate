import numpy as np
from utils.image_utils import write_2images
import os
from utils.summary_utils import SummaryHelper
import time


def get_data():
    x_a = np.random.randint(0, 255, (2, 1, 256, 256))
    x_a2b = np.random.randint(0, 255, (2, 1, 256, 256))
    x_b = np.random.randint(0, 255, (2, 1, 256, 256))
    x_b2a = np.random.randint(0, 255, (2, 1, 256, 256))
    return x_a, x_a2b, x_b, x_b2a


def summary_demo():
    save_path = '.'

    writer = SummaryHelper(save_path, 'train', 5)

    summary = {'loss': 1.0, 'psnr': 3.1}
    writer.add_summary(summary, 1)
    time.sleep(10)

    summary = {'loss': 1.3, 'psnr': 3.5}
    writer.add_summary(summary, 2)
    time.sleep(10)

    summary = {'loss': 1.8, 'psnr': 3.9}
    writer.add_summary(summary, 3)
    time.sleep(10)

    summary = {'loss': 2.3, 'psnr': 4.7}
    writer.add_summary(summary, 4)
    time.sleep(10)

    summary = {'loss': 2.8, 'psnr': 4.9}
    writer.add_summary(summary, 5)
    time.sleep(10)

    summary = {'loss': 3.2, 'psnr': 5.6}
    writer.add_summary(summary, 6)
    time.sleep(10)


if __name__ == '__main__':
    # data = get_data()
    # write_2images(data, 2, './test', 'test_%08d_%04d' % (1, 1))
    summary_demo()
