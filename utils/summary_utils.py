import numpy as np
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter


class SummaryHelper(object):
    def __init__(self, save_path, comment, flush_secs):
        super(SummaryHelper, self).__init__()
        self.writer = SummaryWriter(logdir=save_path, comment=comment, flush_secs=flush_secs)

    def add_summary(self, current_summary, global_step):
        for key, value in current_summary.items():
            if isinstance(value, np.ndarray):
                self.writer.add_image(key, value, global_step)
            elif isinstance(value, float):
                self.writer.add_scalar(key, value, global_step)

    @staticmethod
    def print_current_losses(epoch, iters, losses):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        """
        message = '(epoch: %d, iters: %d) ' % (epoch, iters)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        print(message)  # print the message
