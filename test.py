import os
import time

# import torchvision.utils as vutils
import numpy as np
import torch
from imageio import imsave

from data import create_data_loader
from models import create_model
from options.options import BaseOptions
from utils.util import make_grid

if __name__ == '__main__':
    opt = BaseOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.phase = 'test'
    opt.num_test = 2
    opt.num_style = 3
    model = create_model(opt)

    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

    test_loader = create_data_loader(opt, 'test', opt.batch_size, opt.serial_batches, num_workers=opt.num_workers)

    model.setup()  # regular setup: load and print networks; create schedulers

    A_encode = model.netG_A.encode
    B_encode = model.netG_B.encode
    B_decode = model.netG_B.decode

    # if opt.eval:
    #     model.eval()
    for i, data in enumerate(test_loader):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        image_a = data['A']
        image_a_path = data['A_paths'][0]

        tag = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        ori_name = os.path.basename(image_a_path).split('.')[0]
        save_path = os.path.join(opt.test_dir, ori_name + "_" + tag + ".png")

        content, _ = A_encode(image_a)

        style = torch.randn(opt.num_style, opt.gen['style_dim'], 1, 1).to(device)

        outputs = []
        outputs.append(image_a.clone().cpu().detach().numpy())
        for j in range(opt.num_style):
            s = style[j].unsqueeze(0)
            output = B_decode(content, s)
            output = (output + 1) / 2.
            outputs.append(output.cpu().detach().numpy())

        result = np.concatenate(outputs, 0)

        grid_img = make_grid(result, 1, padding=10, pad_value=1)

        imsave(save_path, grid_img)
