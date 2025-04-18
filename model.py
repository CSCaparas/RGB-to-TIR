import torch
import torch.nn as nn
import networks
from torch.optim import lr_scheduler
from collections import OrderedDict

class Pix2Pix():
    def __init__(self, in_channels, out_channels, isTrain=True, gan_mode='lsgan', device='cuda', lr=0.0002):
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.isTrain = isTrain
        self.device = device

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        self.netG = networks.Unet(in_channels, out_channels).to(self.device)

        if self.isTrain:
            self.netD = networks.PatchGAN(in_channels + out_channels, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d).to(self.device)

        if self.isTrain:
            self.criterionGAN = networks.GANLoss(gan_mode).to(device)
            self.criterionL1 = nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr, betas=(0.5, 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D]

    def setup(self): # from basemodel
        if self.isTrain:
            self.schedulers = [lr_scheduler.StepLR(self.optimizer_G, step_size=50, gamma=0.1), lr_scheduler.StepLR(self.optimizer_D, step_size=100, gamma=0.1)]
        if not self.isTrain:
            for name in self.model_names:
                load_filename = 'model_current_' + name + '.pth'
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_filename)
                state_dict = torch.load(load_filename, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                net.load_state_dict(state_dict)

    def set_input(self, input, target):
        self.input = input.to(self.device)
        self.target = target.to(self.device)

    def forward(self):
        self.fake = self.netG(self.input)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.input, self.fake), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.input, self.target), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.input, self.fake), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake, self.target) * 100.0 # weight for l1 loss--tunable
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            #self.compute_visuals()

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret
    
    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_path = 'model_current_' + name + '.pth'
                net = getattr(self, 'net' + name)

                torch.save(net.cpu().state_dict(), save_path)


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad