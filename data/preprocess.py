from astropy.io import fits
import os
import matplotlib.pyplot as plt
from astropy.utils.data import get_pkg_data_filename
from astropy.visualization import astropy_mpl_style
import numpy as np
import glob2
import random
from imageio import imsave

# set special stype
# plt.style.use(astropy_mpl_style)

# path_prefix = '/Users/beiyan/Documents/Projects/data/fits/'
# HR_image_path = os.path.join(path_prefix, 'Tio_12708_2018-05-08T06_46_12_level1+.fits')
# LR_image_path = os.path.join(path_prefix, '064612/T_000000.fits')

# *****************************  show example —******************
# path_prefix = '/Users/beiyan/Documents/Projects/data/065204_redata/'
# HR_image_path = os.path.join(path_prefix, 'Tio_12671_2017-08-19T06_52_04_level1+.fits')
# LR_image_path = os.path.join(path_prefix, 'Tio_12671_2017-08-19T06_52_34_level1+.fits')
# HR_image_data = fits.getdata(HR_image_path, ext=0)
# LR_image_data = fits.getdata(LR_image_path, ext=0)
#
# plt.figure()
# plt.title('HR img')
# plt.imshow(HR_image_data[100:356,100:356], cmap='gray')

# with fits.open(HR_image_path) as hdul:
#     print("*********** info *****************")
#     print(hdul.info())
#     print("*********** other info *****************")
#     print(hdul[0].header)
#     print(hdul[0].header[7])

# plt.figure()
# plt.title('LR img')
# plt.imshow(LR_image_data, cmap='gray')
#
# plt.show()
# *****************************  end show  —******************

crop_size = 256

# 从200张 斑点图中 截取 20000 张图片
lr_dir = '/Users/beiyan/Documents/Projects/data/fits/064612'
save_path = '../../speckle'
if not os.path.exists(save_path):
    os.mkdir(save_path)

paths = sorted(glob2.glob(os.path.join(lr_dir, '*.fits')))
assert len(paths) == 200

total_index = 0
for f in paths[:100]:
    data = fits.getdata(f, ext=0)[100:-100, 100: -100]
    h, w = np.shape(data)
    for i in range(200):
        total_index += 1
        xt = random.randint(1, w - crop_size)
        yt = random.randint(1, h - crop_size)
        crop_data = data[yt:yt + crop_size, xt:xt + crop_size]
        _min = np.min(crop_data)
        _max = np.max(crop_data)
        nor_crop_data = np.array((crop_data - _min) / (_max - _min) * 255.0, dtype=np.uint8)

        save_filename = os.path.join(save_path, '%05d.png' % total_index)

        imsave(save_filename, nor_crop_data)

        if total_index % 200 == 0:
            print(total_index)


# 从截取 米粒图
hr_dir = '/Users/beiyan/Documents/Projects/data/065204_redata/'
save_path = '../../mili'
if not os.path.exists(save_path):
    os.mkdir(save_path)

paths = sorted(glob2.glob(os.path.join(hr_dir,'*.fits')))


total_index = 0
for f in paths[:50]:
    data = fits.getdata(f, ext=0)[100:-100, 100: -100]
    h, w = np.shape(data)
    for i in range(400):
        total_index += 1
        xt = random.randint(1, w - crop_size)
        yt = random.randint(1, h - crop_size)
        crop_data = data[yt:yt + crop_size, xt:xt + crop_size]
        _min = np.min(crop_data)
        _max = np.max(crop_data)
        nor_crop_data = np.array((crop_data - _min) / (_max - _min) * 255.0, dtype=np.uint8)

        save_filename = os.path.join(save_path, '%05d.png' % total_index)

        imsave(save_filename, nor_crop_data)

        if total_index % 200 == 0:
            print(total_index)

print(len(paths))