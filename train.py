import os
import time

import torch.backends.cudnn as cudnn

from data import create_data_loader
from models import create_model
from options import options
from utils.summary_utils import SummaryHelper
from utils.util import write_2images

if __name__ == '__main__':
    opt = options.BaseOptions().parse()
    cudnn.benchmark = True

    model = create_model(opt)

    train_loader = create_data_loader(opt, 'train', opt.batch_size, opt.serial_batches, num_workers=opt.num_workers)
    valid_loader = create_data_loader(opt, 'valid', 1, False, 1)

    # valid 数据集
    valid_data = model.get_valid_dataset(valid_loader)
    del valid_loader

    train_Summary = SummaryHelper(save_path=os.path.join(opt.log_dir, 'train'), comment=opt.model, flush_secs=20)
    total_iters = 0
    epoch_start, summary_step = model.setup()

    for epoch in range(epoch_start, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0  # epoch内迭代的数据量

        for i, data in enumerate(train_loader):
            iter_start_time = time.time()

            total_iters += 1
            epoch_iter += 1
            model.set_input(data)
            model.optimize_parameters()

            # *************** show visuals, logs, loss *************************#
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                SummaryHelper.print_current_losses(epoch, epoch_iter, losses)

            # ***************   save latest ckp*************************#
            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                model.save_networks('latest', epoch)
                losses = model.get_current_losses()
                train_Summary.add_summary(losses)
                test_image_outputs = model.sample(valid_data)
                write_2images(test_image_outputs, opt.display_size, opt.sample_dir,
                              'test_%08d_%04d' % (epoch + 1, epoch_iter))

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))

            model.save_networks('epoch-%04d' % epoch, epoch, summary_step=summary_step)
        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
