import numpy as np
from utils.util import write_2images


def get_data():
    x_a = np.random.randint(0, 255, (2, 1, 256, 256))
    x_a2b = np.random.randint(0, 255, (2, 1, 256, 256))
    x_b = np.random.randint(0, 255, (2, 1, 256, 256))
    x_b2a = np.random.randint(0, 255, (2, 1, 256, 256))
    return x_a, x_a2b, x_b, x_b2a


data = get_data()
write_2images(data, 2, './test', 'test_%08d_%04d' % (1, 1))
