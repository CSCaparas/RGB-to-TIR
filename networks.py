import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

"""
UNet model for image segmentation (for use as pix2pix generator)

The model consists of a downward path and an upward path. 
The downward path consists of a series of convolutions and pooling layers that reduce the spatial dimensions of the input image. 
The upward path consists of a series of transposed convolutions and concatenations with the skip connections from the downward path.

For more information and references to use cases, refer to:
https://paperswithcode.com/method/u-net
"""
class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(Unet, self).__init__()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downward path
        for feature in features:
            self.down.append(nn.Sequential(
                nn.Conv2d(in_channels, feature, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(0.5)
            ))
            in_channels = feature

        # Upward path
        for feature in reversed(features):
            self.up.append(nn.Sequential(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2),
                nn.Conv2d(feature * 2, feature, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(0.5)
            ))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1] * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []

        for down in self.down:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx, up in enumerate(self.up):
            x = up[0](x)  # Transposed convolution
            x = F.interpolate(x, size=skips[idx].shape[2:], mode='bilinear', align_corners=True) # Upsample
            x = torch.cat((x, skips[idx]), dim=1) # Concatenate skip connection
            x = up[1:](x)  # Convolutions after concatenation

        return self.final_conv(x)

"""
70x70 PatchGAN discriminator model for image segmentation (for use as pix2pix discriminator)
"""
class PatchGAN(nn.Module):
    """
    Parameters:
        in_channels (int): number of input channels
        ndf (int): number of filters in the first conv layer
        n_layers (int): number of conv layers in the discriminator
        norm_layer (torch.nn.Module): normalization layer
    """
    def __init__(self, in_channels=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(PatchGAN, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
            
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError('GAN mode %s not implemented' % gan_mode)
            
    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
        
    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss