from gan_training.models import (dcgan_deep, dcgan_shallow)

generator_dict = {
    'dcgan_deep': dcgan_deep.Generator,
    'dcgan_shallow': dcgan_shallow.Generator
}

discriminator_dict = {
    'dcgan_deep': dcgan_deep.Discriminator,
    'dcgan_shallow': dcgan_shallow.Discriminator
}

encoder_dict = {
    'dcgan_shallow': dcgan_shallow.Encoder,
    'dcgan_deep': dcgan_deep.Encoder,
}
