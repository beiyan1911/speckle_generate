import os
from abc import ABC
from models.networks import get_scheduler
import torch
from collections import OrderedDict


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.phase == 'train'
        self.device = torch.device('cuda:{}'.format(self.opt.gpu_ids[0])) if self.opt.gpu_ids else torch.device(
            'cpu')  # get device name: CPU or GPU

        self.loss_names = []  # loss 名字，用于可视化loss变化过程
        self.model_names = []  # model 名字，用户模型save和load
        self.optimizer_names = []  # use for optimizer load and save
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0

    def setup(self):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        load_prefix = self.opt.load_flag if self.opt.load_flag else 'latest'
        epoch = self.load_networks(load_prefix)

        if self.isTrain:
            self.schedulers = [get_scheduler(optimizer, self.opt) for optimizer in self.optimizers]
        self.print_networks(self.opt.print_network)
        return epoch

    def load_networks(self, prefix):
        """Load all the networks from the disk.

        Parameters:
            prefix (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        epoch = 0
        # load optimizer and other setting
        load_filename = '%s_optimizer.pth' % prefix
        load_path = os.path.join(self.opt.checkpoints_dir, load_filename)
        if os.path.exists(load_path):
            print('loading the optimizer from %s' % load_path)
            opt_dict = torch.load(load_path)
            epoch = opt_dict['epoch']
            for name in self.optimizer_names:
                if isinstance(name, str):
                    optimizer = getattr(self, name)
                    optimizer.load_state_dict(opt_dict[name])

        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (prefix, name)
                load_path = os.path.join(self.opt.checkpoints_dir, load_filename)
                if not os.path.exists(load_path):
                    continue
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                net.load_state_dict(state_dict)
        return epoch

    def save_networks(self, prefix, epoch):
        """
        :param prefix:
        :param epoch:
        :return:
        """

        # save model
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (prefix, name)
                save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 1 and torch.cuda.is_available():
                    torch.save(net.module.state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.state_dict(), save_path)

        # save optimizer and other setting
        opt_dict = {}
        for name in self.optimizer_names:
            if isinstance(name, str):
                optimizer = getattr(self, name)
                opt_dict[name] = optimizer.state_dict()
        opt_dict['epoch'] = epoch
        save_filename = '%s_optimizer.pth' % (prefix)
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        torch.save(opt_dict, save_path)

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('------------- Networks initialized ----------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------------')
