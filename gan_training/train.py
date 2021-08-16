# coding: utf-8
import torch
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd
import numpy as np


class Trainer(object):
    def __init__(self,
                 generator,
                 discriminator,
                 encoder,
                 g_optimizer,
                 d_optimizer,
                 q_optimizer,
                 gan_type,
                 reg_type,
                 reg_param,
                 encoder_type,
                 encoder_param):

        self.generator = generator
        self.discriminator = discriminator
        self.encoder = encoder
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.q_optimizer = q_optimizer
        self.gan_type = gan_type
        self.reg_type = reg_type
        self.reg_param = reg_param
        self.encoder_type = encoder_type
        self.encoder_param = encoder_param

        self.margin = 0.25

        print('gan type: ', gan_type)
        print('D reg type:', reg_type)
        print('D reg gamma: ', reg_param)
        print('Q loss type: ', encoder_type)
        print('Q loss param: ', encoder_param)

    def generator_trainstep(self, y, z):
        assert (y.size(0) == z.size(0))
        toggle_grad(self.generator, True)
        toggle_grad(self.discriminator, False)

        self.generator.train()
        self.discriminator.train()
        self.g_optimizer.zero_grad()

        x_fake = self.generator(z, y)
        d_fake = self.discriminator(x_fake, y)
        gloss = self.compute_loss(d_fake, 1)
        gloss.backward()

        self.g_optimizer.step()

        return gloss.item()

    def discriminator_trainstep(self, x_real, y, z):
        toggle_grad(self.generator, False)
        toggle_grad(self.discriminator, True)
        self.generator.train()
        self.discriminator.train()
        self.d_optimizer.zero_grad()

        # On real data
        x_real.requires_grad_()

        d_real = self.discriminator(x_real, y)
        dloss_real = self.compute_loss(d_real, 1)

        if self.reg_type == 'real' or self.reg_type == 'real_fake':
            dloss_real.backward(retain_graph=True)
            reg = self.reg_param * compute_grad2(d_real, x_real).mean()
            reg.backward()
        else:
            dloss_real.backward()

        # On fake data
        with torch.no_grad():
            x_fake = self.generator(z, y)

        x_fake.requires_grad_()
        d_fake = self.discriminator(x_fake, y)
        dloss_fake = self.compute_loss(d_fake, 0)

        if self.reg_type == 'fake' or self.reg_type == 'real_fake':
            dloss_fake.backward(retain_graph=True)
            reg = self.reg_param * compute_grad2(d_fake, x_fake).mean()
            reg.backward()
        else:
            dloss_fake.backward()

        if self.reg_type == 'wgangp':
            reg = self.reg_param * self.wgan_gp_reg(x_real, x_fake, y)
            reg.backward()
        elif self.reg_type == 'wgangp0':
            reg = self.reg_param * self.wgan_gp_reg(
                x_real, x_fake, y, center=0.)
            reg.backward()

        self.d_optimizer.step()

        dloss = (dloss_real + dloss_fake)
        if self.reg_type == 'none':
            reg = torch.tensor(0.)

        return dloss.item(), reg.item()

    def encoder_trainstep(self, y, z, target_embeds):
        assert (y.size(0) == z.size(0))
        # assert (y.size(0) == target_embeds.size(0))
        toggle_grad(self.generator, False)
        toggle_grad(self.discriminator, False)
        toggle_grad(self.encoder, True)

        self.generator.eval()
        self.discriminator.eval()
        self.encoder.train()
        self.q_optimizer.zero_grad()

        x_fake = self.generator(z, y)
        x_fake.requires_grad_()
        embeds = self.encoder(x_fake)
        qloss = self.compute_embed_loss(embeds, target_embeds)
        
        qloss.backward()

        self.q_optimizer.step()

        return qloss.item()

    def compute_embed_loss(self, embeds, target_embeds):
        if self.encoder_type == "l2":
            loss = F.mse_loss(embeds, target_embeds) * self.encoder_param[0]
        elif self.encoder_type == "l1":
            loss = F.l1_loss(embeds, target_embeds) * self.encoder_param[0]
        elif self.encoder_type == "mean_l2":
            embeds_mean = embeds.mean(dim=0)
            mean_loss = F.mse_loss(embeds_mean, target_embeds[0]) * self.encoder_param[0]
            reg_loss = F.mse_loss(embeds, target_embeds.repeat(embeds.shape[0], 1)) * self.encoder_param[1]
            loss = mean_loss + reg_loss
        elif self.encoder_type == "mean_l1":
            embeds_mean = embeds.mean(dim=0)
            mean_loss = F.mse_loss(embeds_mean, target_embeds[0]) * self.encoder_param[0]
            reg_loss = F.l1_loss(embeds, target_embeds.repeat(embeds.shape[0], 1)) * self.encoder_param[1]
            loss = mean_loss + reg_loss
        elif self.encoder_type == "l2_margin":
            diff = embeds - target_embeds
            norm = torch.norm(diff, p=2, dim=1)
            dist = torch.clamp(norm - self.margin, min=0.0)
            loss = torch.mean(dist ** 2)
        elif self.encoder_type == "l1_margin":
            diff = embeds - target_embeds
            norm = torch.norm(diff, p=2, dim=1)
            dist = torch.clamp(norm - self.margin, min=0.0)
            loss = torch.mean(dist)
        elif self.encoder_type == "mean_l2_margin":
            embeds_mean = embeds.mean(dim=0)
            mean_loss = F.mse_loss(embeds_mean, target_embeds[0]) * self.encoder_param[0]
            
            diff = embeds - target_embeds
            norm = torch.norm(diff, p=2, dim=1)
            dist = torch.clamp(norm - self.margin, min=0.0)
            reg_loss = torch.mean(dist ** 2) * self.encoder_param[1]

            loss = mean_loss + reg_loss
        elif self.encoder_type == "l1_margin":
            embeds_mean = embeds.mean(dim=0)
            mean_loss = F.mse_loss(embeds_mean, target_embeds[0]) * self.encoder_param[0]
            
            diff = embeds - target_embeds
            norm = torch.norm(diff, p=2, dim=1)
            dist = torch.clamp(norm - self.margin, min=0.0)
            reg_loss = torch.mean(dist) * self.encoder_param[1]

            loss = mean_loss + reg_loss
        else:
            raise NotImplementedError

        return loss

    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)

        if self.gan_type == 'standard':
            loss = F.binary_cross_entropy_with_logits(d_out, targets)
        elif self.gan_type == 'wgan':
            loss = (2 * target - 1) * d_out.mean()
        else:
            raise NotImplementedError

        return loss

    def wgan_gp_reg(self, x_real, x_fake, y, center=1.):
        batch_size = y.size(0)
        eps = torch.rand(batch_size, device=y.device).view(batch_size, 1, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.discriminator(x_interp, y)

        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()

        return reg


# Utility functions
def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(outputs=d_out.sum(),
                              inputs=x_in,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def update_average(model_tgt, model_src, beta):
    toggle_grad(model_src, False)
    toggle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)
