import argparse
import os
import shutil

import yaml

from utils.image_utils import mkdir


class BaseOptions():

    def __init__(self):
        self.initialized = False
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def initialize(self):
        if not self.initialized:
            #  demo_edges2handbags  demo_speckle
            self.parser.add_argument('--data_root', default='../datasets/demo_speckle/', type=str, help='# of iter to linearly decay learning rate to zero')
            self.parser.add_argument('--config', default='configs/demo_speckle.yaml', type=str, help='Path to the config file.')
            self.parser.add_argument('--output_path', default='../outputs', type=str, help="outputs path")
            self.parser.add_argument('--model', default='munit', type=str, help="MUNIT|UNIT")
            self.parser.add_argument('--phase', default='train', type=str, help='train, val, test, etc')
            self.parser.add_argument('--epoch_count', default=1, type=int, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
            self.parser.add_argument('--direction', default='AtoB', type=str, help='AtoB or BtoA')
            self.parser.add_argument('--preprocess', default='', type=str, help='scaling and cropping of images at load time [resize_and_crop | crop | none]')
            self.parser.add_argument('--print_network', action='store_true', help='if specified, print more debugging information')
            self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
            self.parser.add_argument('--gpu_ids', default='-1', type=str, help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
            self.parser.add_argument('--max_dataset_size', default=float("inf"), type=int, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
            self.parser.add_argument('--niter', default=100, type=int, help='# of iter at starting learning rate')
            self.parser.add_argument('--lr_decay_iters', default=50, type=int, help='multiply by a gamma every lr_decay_iters iterations')
            self.parser.add_argument('--niter_decay', default=100, type=int, help='# of iter to linearly decay learning rate to zero')
            self.parser.add_argument('--batch_size', default=8, type=int, help='batch size')

    def gather_options(self):

        self.initialize()
        opt = self.parser.parse_args()

        # combine yaml experiment setting
        with open(opt.config, 'r') as stream:
            config = yaml.load(stream, yaml.FullLoader)

        # add yaml dict to parse
        for key, value in config.items():
            if isinstance(value, list):
                for v in value:
                    getattr(opt, key, []).append(v)
            else:
                setattr(opt, key, value)

        # setting output paths
        data_name = os.path.splitext(os.path.basename(opt.config))[0]
        model_name = opt.model + '___' + data_name

        opt.checkpoints_dir = os.path.join(opt.output_path, model_name, 'checkpoints')
        opt.sample_dir = os.path.join(opt.output_path, model_name, 'samples')
        opt.log_dir = os.path.join(opt.output_path, model_name, 'logs')
        opt.test_dir = os.path.join(opt.output_path, model_name, 'test')

        mkdir(opt.output_path)
        mkdir(os.path.join(opt.output_path, model_name))
        mkdir(opt.checkpoints_dir)
        mkdir(opt.sample_dir)
        mkdir(opt.log_dir)
        mkdir(opt.test_dir)
        shutil.copy(opt.config, os.path.join(opt.checkpoints_dir,'config.yaml'))

        return opt

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            message += '{:>20}:   {:<30}\n'.format(str(k), str(v))
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        file_name = os.path.join(opt.checkpoints_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)

        return opt
