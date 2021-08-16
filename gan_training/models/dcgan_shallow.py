import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from gan_training.models import blocks


class Generator(nn.Module):
    def __init__(self,
                 num_k,
                 conditioning,
                 z_dim=256,
                 nc=3,
                 ngf=64,
                 embed_dim=256,
                 embed_norm=True,
                 **kwargs):
        super(Generator, self).__init__()


        if conditioning == 'embedding':
            self.get_latent = blocks.LatentEmbeddingConcat(num_k, embed_dim)
            self.fc = nn.Linear(z_dim + embed_dim, 4 * 4 * ngf * 8)
        elif conditioning == 'embedding_add':
            self.get_latent = blocks.LatentEmbeddingAdd(num_k, embed_dim, norm=embed_norm)
            self.fc = nn.Linear(embed_dim, 4 * 4 * ngf * 8)
        elif conditioning == 'unconditional':
            self.get_latent = blocks.Identity()
            self.fc = nn.Linear(z_dim, 4 * 4 * ngf * 8)
        else:
            raise NotImplementedError(
                f"{conditioning} not implemented for generator")

        bn = blocks.BatchNorm2d

        self.num_k = num_k

        self.conv1 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1)
        self.bn1 = bn(ngf * 4, num_k)

        self.conv2 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1)
        self.bn2 = bn(ngf * 2, num_k)

        self.conv3 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1)
        self.bn3 = bn(ngf, num_k)

        self.conv_out = nn.Sequential(nn.Conv2d(ngf, nc, 3, 1, 1), nn.Tanh())

    def forward(self, input, y):
        y = y.clamp(None, self.num_k - 1)
        out = self.get_latent(input, y)
        out = self.fc(out)

        out = out.view(out.size(0), -1, 4, 4)
        out = F.relu(self.bn1(self.conv1(out), y))
        out = F.relu(self.bn2(self.conv2(out), y))
        out = F.relu(self.bn3(self.conv3(out), y))
        return self.conv_out(out)


class Discriminator(nn.Module):
    def __init__(self,
                 num_k,
                 conditioning,
                 features='penultimate',
                 pack_size=1,
                 nc=3,
                 ndf=64,
                 **kwargs):
        super(Discriminator, self).__init__()

        self.num_k = num_k

        self.conv1 = nn.Sequential(nn.Conv2d(nc * pack_size, ndf, 4, 2, 1),
                                   nn.BatchNorm2d(ndf),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
                                   nn.BatchNorm2d(ndf * 2),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
                                   nn.BatchNorm2d(ndf * 4),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
                                   nn.BatchNorm2d(ndf * 8),
                                   nn.LeakyReLU(0.2, inplace=True))

        if conditioning == 'mask':
            self.fc_out_cond = blocks.LinearConditionalMaskLogits(ndf * 8 * 4 , num_k)
        elif conditioning == 'unconditional':
            self.fc_out_uncond = blocks.LinearUnconditionalLogits(ndf * 8 * 4)
        else:
            raise NotImplementedError(
                f"{conditioning} not implemented for discriminator")

        self.pack_size = pack_size
        self.features = features
        self.conditioning = conditioning
        # print(f'Getting features from {self.features}')

    def stack(self, x):
        #pacgan
        nc = self.pack_size
        if nc == 1:
            return x
        x_new = []
        for i in range(x.size(0) // nc):
            imgs_to_stack = x[i * nc:(i + 1) * nc]
            x_new.append(torch.cat([t for t in imgs_to_stack], dim=0))
        return torch.stack(x_new)

    def forward(self, input, y=None, get_features=False):
        input = self.stack(input)
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(out.size(0), -1)
        if get_features: 
            return out.view(out.size(0), -1)
        y = y.clamp(None, self.num_k - 1)
        if self.conditioning == 'unconditional':
            result = self.fc_out_uncond(out, y)
        else:
            result = self.fc_out_cond(out, y)
        assert (len(result.shape) == 1)
        return result

class Encoder(nn.Module):
    def __init__(self,
                 nc=3,
                 ndf=64,
                 embed_dim=-1,
                 **kwargs):
        super(Encoder, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(nc, ndf, 4, 2, 1),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
                                   nn.LeakyReLU(0.2, inplace=True))

        self.fc_out = nn.Linear(ndf * 8 * 4, embed_dim)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(out.size(0), -1)
        result = self.fc_out(out)
        return result
