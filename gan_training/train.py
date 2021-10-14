# coding: utf-8
import torch
from torch.nn import functional as F
from torch import autograd

class Trainer(object):
    def __init__(self,
                 generator,
                 discriminator,
                 encoder,
                 g_optimizer,
                 d_optimizer,
                 q_optimizer):

        self.generator = generator
        self.discriminator = discriminator
        self.encoder = encoder
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.q_optimizer = q_optimizer

    def generator_trainstep(self, y, z, condition):
        assert (y.size(0) == z.size(0))
        self.generator.train()

        x_fake = self.generator(z, y)
        d_fake = self.discriminator(x_fake, y, condition)
        g_loss = -d_fake.mean()
        # g_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return g_loss.item()

    def discriminator_trainstep(self, x_real, y, z, condition):
        self.discriminator.train()

        # Sampling
        x_fake = self.generator(z, y).detach()
        self.d_optimizer.zero_grad()
        
        x_real.requires_grad_()
        d_real = self.discriminator(x_real, y, condition)
        d_fake = self.discriminator(x_fake, y, condition)
        # d_loss_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
        # d_loss_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
        d_loss_real = -d_real.mean()
        d_loss_fake = d_fake.mean()

        reg_loss = 10.0 * compute_grad2(d_real, x_real).mean()
        
        d_loss = d_loss_real + d_loss_fake + reg_loss
        d_loss.backward()
        self.d_optimizer.step()

        return d_loss_real.item(), d_loss_fake.item(), reg_loss.item()

    def encoder_trainstep(self, y, z, target_embeds):
        assert (y.size(0) == z.size(0))
        # assert (y.size(0) == target_embeds.size(0))

        self.generator.eval()
        self.encoder.train()

        x_fake = self.generator(z, y)
        x_fake.requires_grad_()
        embeds = self.encoder(x_fake)
        qloss = F.mse_loss(embeds, target_embeds) * 0.1
        
        self.q_optimizer.zero_grad()
        qloss.backward()

        self.q_optimizer.step()

        self.generator.train()

        return qloss.item()


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
