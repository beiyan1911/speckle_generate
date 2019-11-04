import torch
import torch.nn as nn
import numpy as np
from models.base_model import BaseModel
from models.networks import AdaINGen, MsImageDis, get_scheduler
from models.networks import init_net
from utils.net_util import vgg_preprocess, load_vgg16, get_scheduler


class MunitModel(BaseModel):
    def __init__(self, opts):

        BaseModel.__init__(self, opts)

        lr = self.opt.lr

        self.model_names = ['G_A', 'G_B']
        self.loss_names = ['d_total', 'g_total', 'g_rec_x_a', 'g_rec_x_b', 'g_rec_s_a', 'g_rec_s_b', 'g_rec_c_a',
                           'g_rec_c_b', 'g_adv_a', 'g_adv_b']
        self.visual_names = []
        # Initiate the networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
            self.netD_A = init_net(MsImageDis(self.opt.input_dim_a, self.opt.dis), init_type=self.opt.init,
                                   gpu_ids=self.gpu_ids)
            self.netD_B = init_net(MsImageDis(self.opt.input_dim_b, self.opt.dis), init_type=self.opt.init,
                                   gpu_ids=self.gpu_ids)
        self.netG_A = init_net(AdaINGen(self.opt.input_dim_a, self.opt.gen), init_type=self.opt.init,
                               gpu_ids=self.gpu_ids)
        self.netG_B = init_net(AdaINGen(self.opt.input_dim_b, self.opt.gen), init_type=self.opt.init,
                               gpu_ids=self.gpu_ids)

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = self.opt.gen['style_dim']

        # fix the noise used in sampling
        display_size = self.opt.display_size
        self.s_a_fixed = torch.randn(display_size, self.style_dim, 1, 1).to(self.device)
        self.s_b_fixed = torch.randn(display_size, self.style_dim, 1, 1).to(self.device)

        if self.isTrain:
            # Setup the optimizers

            d_params = list(self.netD_A.parameters()) + list(self.netD_B.parameters())
            g_params = list(self.netG_A.parameters()) + list(self.netG_B.parameters())

            self.optimizer_D = torch.optim.Adam([p for p in d_params if p.requires_grad],
                                                lr=lr, betas=(self.opt.beta1, self.opt.beta2),
                                                weight_decay=self.opt.weight_decay)
            self.optimizer_G = torch.optim.Adam([p for p in g_params if p.requires_grad],
                                                lr=lr, betas=(self.opt.beta1, self.opt.beta2),
                                                weight_decay=self.opt.weight_decay)
            self.optimizer_names = ['optimizer_D', 'optimizer_G']  # 为空则不保存 optimizer
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_G)

            self.scheduler_D = get_scheduler(self.optimizer_D, self.opt)
            self.scheduler_G = get_scheduler(self.optimizer_G, self.opt)
            self.schedulers = [self.scheduler_D, self.scheduler_G]

        # Load VGG model if needed
        if (self.opt.vgg_w is not None) and self.opt.vgg_w > 0:
            self.vgg = load_vgg16(self.opt.vgg_model_path + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.images_A = input['A' if AtoB else 'B'].to(self.device)
        self.images_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.netG_A.eval()
        self.netG_B.eval()
        self.netD_A.eval()
        self.netD_B.eval()
        c_a, s_a_fake = self.netG_A.encode(x_a)
        c_b, s_b_fake = self.netG_B.encode(x_b)
        x_ba = self.netG_A.decode(c_b, self.s_a_fixed)
        x_ab = self.netG_B.decode(c_a, self.s_b_fixed)
        self.netG_A.train()
        self.netG_B.train()
        self.netD_A.train()
        self.netD_B.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b):
        self.optimizer_G.zero_grad()

        s_a_rand = torch.randn(x_a.size(0), self.style_dim, 1, 1).to(self.device)
        s_b_rand = torch.randn(x_b.size(0), self.style_dim, 1, 1).to(self.device)
        # encode
        c_a, s_a_encoded = self.netG_A.encode(x_a)
        c_b, s_b_encoded = self.netG_B.encode(x_b)
        # decode (within domain)
        x_a_rec = self.netG_A.decode(c_a, s_a_encoded)
        x_b_rec = self.netG_B.decode(c_b, s_b_encoded)
        # decode (cross domain)
        x_ba = self.netG_A.decode(c_b, s_a_rand)
        x_ab = self.netG_B.decode(c_a, s_b_rand)
        # encode again
        c_b_rec, s_a_rand_rec = self.netG_A.encode(x_ba)
        c_a_rec, s_b_rand_rec = self.netG_B.encode(x_ab)
        # decode again (if needed)
        x_aba = self.netG_A.decode(c_a_rec, s_a_encoded) if self.opt.recon_x_cyc_w > 0 else None
        x_bab = self.netG_B.decode(c_b_rec, s_b_encoded) if self.opt.recon_x_cyc_w > 0 else None
        # reconstruction loss
        self.loss_g_rec_x_a = self.recon_criterion(x_a_rec, x_a)
        self.loss_g_rec_x_b = self.recon_criterion(x_b_rec, x_b)
        self.loss_g_rec_s_a = self.recon_criterion(s_a_rand_rec, s_a_rand)
        self.loss_g_rec_s_b = self.recon_criterion(s_b_rand_rec, s_b_rand)
        self.loss_g_rec_c_a = self.recon_criterion(c_a_rec, c_a)
        self.loss_g_rec_c_b = self.recon_criterion(c_b_rec, c_b)
        self.loss_g_cycrec_x_a = self.recon_criterion(x_aba, x_a) if self.opt.recon_x_cyc_w > 0 else 0
        self.loss_g_cycrec_x_b = self.recon_criterion(x_bab, x_b) if self.opt.recon_x_cyc_w > 0 else 0
        # GAN loss
        self.loss_g_adv_a = self.netD_A.calc_gen_loss(x_ba)
        self.loss_g_adv_b = self.netD_B.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_g_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if self.opt.vgg_w > 0 else 0
        self.loss_g_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if self.opt.vgg_w > 0 else 0
        # total loss
        self.loss_g_total = self.opt.gan_w * self.loss_g_adv_a + \
                            self.opt.gan_w * self.loss_g_adv_b + \
                            self.opt.recon_x_w * self.loss_g_rec_x_a + \
                            self.opt.recon_s_w * self.loss_g_rec_s_a + \
                            self.opt.recon_c_w * self.loss_g_rec_c_a + \
                            self.opt.recon_x_w * self.loss_g_rec_x_b + \
                            self.opt.recon_s_w * self.loss_g_rec_s_b + \
                            self.opt.recon_c_w * self.loss_g_rec_c_b + \
                            self.opt.recon_x_cyc_w * self.loss_g_cycrec_x_a + \
                            self.opt.recon_x_cyc_w * self.loss_g_cycrec_x_b + \
                            self.opt.vgg_w * self.loss_g_vgg_a + \
                            self.opt.vgg_w * self.loss_g_vgg_b
        self.loss_g_total.backward()
        self.optimizer_G.step()

    def dis_update(self, x_a, x_b):
        self.optimizer_D.zero_grad()
        s_a = torch.randn(x_a.size(0), self.style_dim, 1, 1).to(self.device)
        s_b = torch.randn(x_b.size(0), self.style_dim, 1, 1).to(self.device)
        # encode
        c_a, _ = self.netG_A.encode(x_a)
        c_b, _ = self.netG_B.encode(x_b)
        # decode (cross domain)
        x_ba = self.netG_A.decode(c_b, s_a)
        x_ab = self.netG_B.decode(c_a, s_b)
        # D loss
        self.loss_d_a = self.netD_A.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_d_b = self.netD_B.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_d_total = self.opt.gan_w * self.loss_d_a + self.opt.gan_w * self.loss_d_b
        self.loss_d_total.backward()
        self.optimizer_D.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def get_valid_dataset(self, data_loader):

        AtoB = self.opt.direction == 'AtoB'

        x_a = []
        x_b = []
        # train_display_images_a = torch.stack([ for i in range(display_size)]).cuda()
        for i in range(self.opt.display_size):
            data = data_loader.dataset[i]
            x_a.append(data['A' if AtoB else 'B'])
            x_b.append(data['B' if AtoB else 'A'])

        valid_image_a = torch.stack(x_a).to(self.device)
        valid_images_b = torch.stack(x_b).to(self.device)
        return {'images_a': valid_image_a, 'images_b': valid_images_b}

    def sample(self, sample_data):
        x_a, x_b = sample_data['images_a'], sample_data['images_b']
        self.netG_A.eval()
        self.netG_B.eval()

        s_a2 = torch.randn(x_a.size(0), self.style_dim, 1, 1).to(self.device)
        s_b2 = torch.randn(x_b.size(0), self.style_dim, 1, 1).to(self.device)
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.netG_A.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.netG_B.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.netG_A.decode(c_a, s_a_fake).cpu().detach().numpy())
            x_b_recon.append(self.netG_B.decode(c_b, s_b_fake).cpu().detach().numpy())
            x_ba1.append(self.netG_A.decode(c_b, self.s_a_fixed[i].unsqueeze(0)).cpu().detach().numpy())
            x_ba2.append(self.netG_A.decode(c_b, s_a2[i].unsqueeze(0)).cpu().detach().numpy())
            x_ab1.append(self.netG_B.decode(c_a, self.s_b_fixed[i].unsqueeze(0)).cpu().detach().numpy())
            x_ab2.append(self.netG_B.decode(c_a, s_b2[i].unsqueeze(0)).cpu().detach().numpy())
        x_a_recon, x_b_recon = np.concatenate(x_a_recon), np.concatenate(x_b_recon)
        x_ba1, x_ba2 = np.concatenate(x_ba1), np.concatenate(x_ba2)
        x_ab1, x_ab2 = np.concatenate(x_ab1), np.concatenate(x_ab2)
        self.netG_A.train()
        self.netG_B.train()
        return x_a.cpu().detach().numpy(), x_a_recon, x_ab1, x_ab2, x_b.cpu().detach().numpy(), x_b_recon, x_ba1, x_ba2

    def optimize_parameters(self):
        self.dis_update(self.images_A, self.images_B)
        self.gen_update(self.images_A, self.images_B)
