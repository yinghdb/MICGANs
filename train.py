import argparse
import os
import copy
import pprint
from os import path

import torch
import numpy as np
from torch import nn

from gan_training import utils
from gan_training.train import Trainer, update_average
from gan_training.logger import Logger
from gan_training.checkpoints import CheckpointIO
from gan_training.inputs import get_dataset
from gan_training.distributions import get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (load_config, get_clusterer, build_models, build_optimizers)
from gan_training.utils import weights_init
import time

torch.backends.cudnn.benchmark = True

# Arguments
parser = argparse.ArgumentParser(
    description='Train a GAN with different regularization strategies.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--outdir', type=str, help='used to override outdir (useful for multiple runs)')
parser.add_argument('--model_epoch', type=int, default=-1, help='which model epoch to load from, -1 loads the most recent model')
parser.add_argument('--devices', nargs='+', type=str, default=['0'], help='devices to use')

args = parser.parse_args()
config = load_config(args.config, 'configs/default.yaml')
out_dir = config['training']['out_dir'] if args.outdir is None else args.outdir

def main():
    pp = pprint.PrettyPrinter(indent=1)
    pp.pprint({
        'data': config['data'],
        'generator': config['generator'],
        'discriminator': config['discriminator'],
        'encoder': config['encoder'],
        'multi_gauss': config['multi_gauss'],
        'clusterer': config['clusterer'],
        'training': config['training'], 
        'pretrained': config['pretrained']
    })
    is_cuda = torch.cuda.is_available()

    # Short hands
    batch_size = config['training']['batch_size']
    num_k = config['condition']['num_k']

    checkpoint_dir = path.join(out_dir, 'chkpts')

    # Create missing directories
    if not path.exists(out_dir):
        os.makedirs(out_dir)
    if not path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Logger
    checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)

    device = torch.device("cuda:0" if is_cuda else "cpu")

    train_dataset, _ = get_dataset(
        name=config['data']['type'],
        data_dir=config['data']['train_dir'],
        size=config['data']['img_size'],
        deterministic=config['data']['deterministic'])
    image_num = len(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0, #config['training']['nworkers'],
        shuffle=True,
        pin_memory=True,
        sampler=None,
        drop_last=True)

    eval_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0, #config['training']['nworkers'],
        shuffle=False,
        pin_memory=True,
        sampler=None,
        drop_last=False)

    # collect ground-truth labels for evaluation
    data_label_ims = np.zeros(image_num, dtype=int)
    for _, real_label, real_index in eval_loader:
        data_label_ims[real_index] = real_label
    label_num = len(np.unique(data_label_ims))

    # Create models
    generator, discriminator, encoder, multi_gauss = build_models(config)

    # Put models on gpu if needed
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    encoder = encoder.to(device)
    multi_gauss = multi_gauss.to(device)

    g_optimizer, d_optimizer, q_optimizer = build_optimizers(generator, discriminator, encoder, config)

    devices = [int(x) for x in args.devices]
    generator = nn.DataParallel(generator, device_ids=devices)
    discriminator = nn.DataParallel(discriminator, device_ids=devices)
    encoder = nn.DataParallel(encoder, device_ids=devices)
    multi_gauss = nn.DataParallel(multi_gauss, device_ids=devices)

    g_module = generator.module
    d_module = discriminator.module
    q_module = encoder.module
    mg_module = multi_gauss.module

    # Register modules to checkpoint
    checkpoint_io.register_modules(generator=g_module,
                                   discriminator=d_module,
                                   encoder=q_module,
                                   multi_gauss=mg_module,
                                   g_optimizer=g_optimizer,
                                   d_optimizer=d_optimizer,
                                   q_optimizer=q_optimizer)
    
    # Logger
    logger = Logger(log_dir=path.join(out_dir, 'logs'),
                    img_dir=path.join(out_dir, 'imgs'),
                    label_mode_dir=path.join(out_dir, 'label_mode'),
                    mode_label_dir=path.join(out_dir, 'mode_label'),
                    sorted_mode_label_dir=path.join(out_dir, 'sorted_mode_label'),
                    monitoring=config['training']['monitoring'],
                    monitoring_dir=path.join(out_dir, 'monitoring'))

    # Noise Distribution
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)

    # Test samples
    ntest = config['training']['ntest']
    x_test, y_test = utils.get_nsamples(train_loader, ntest)
    z_test = zdist.sample((ntest, ))
    # x_test, y_test = x_test.to(device), y_test.to(device)
    # utils.save_images(x_test, path.join(out_dir, 'real.png'))
    # logger.add_imgs(x_test, 'gt', 0)

    # Test generator
    if config['training']['take_model_average']:
        print('Taking model average')
        bad_modules = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
        for model in [generator, discriminator]:
            for name, module in model.named_modules():
                for bad_module in bad_modules:
                    if isinstance(module, bad_module):
                        print('Batch norm in discriminator not compatible with exponential moving average')
                        exit()
        generator_test = copy.deepcopy(generator)
        checkpoint_io.register_modules(generator_test=generator_test)
    else:
        generator_test = generator

    clusterer = get_clusterer(config)(num_k=num_k,
                                      gt_nlabels=config['data']['nlabels'],
                                      multi_gauss=mg_module,
                                      **config['clusterer']['kwargs'])
                                      
    # Load checkpoint
    if args.model_epoch == -1:
        epoch_idx = utils.get_most_recent(checkpoint_dir, 'model')
        epoch_idx = checkpoint_io.load_models(epoch_idx=epoch_idx, pretrained=config['pretrained'])
    else:
        epoch_idx = args.model_epoch
        checkpoint_io.load_models(epoch_idx=epoch_idx, pretrained=config['pretrained'])

    # Evaluator
    evaluator = Evaluator(
        generator_test,
        discriminator,
        encoder, 
        mg_module, 
        train_loader=train_loader,
        batch_size=batch_size,
        device=device)

    # Trainer
    trainer = Trainer(generator,
                      discriminator,
                      encoder,
                      g_optimizer,
                      d_optimizer,
                      q_optimizer,
                      gan_type=config['training']['gan_type'],
                      reg_type=config['training']['reg_type'],
                      reg_param=config['training']['reg_param'],
                      encoder_type=config['training']['encoder_type'],
                      encoder_param=config['training']['encoder_param'])

    # initialization training loop
    if config['training']['stage'] == "initialization":
        print('Start initialization training ...')
        count = 0
        it = 0
        dataiter = iter(train_loader)
        while count < config['training']['n_init']:
            try:
                x_real, y, _ = next(dataiter)
            except StopIteration:
                dataiter = iter(train_loader)
                x_real, y, _ = next(dataiter)

            count += batch_size
            it += 1

            x_real, y = x_real.to(device), y.to(device)
            z = zdist.sample((batch_size, ))
            y = clusterer.get_labels(x_real, y).to(device)
            
            # Discriminator updates
            dloss, reg = trainer.discriminator_trainstep(x_real, y, z)
            logger.add('losses', 'discriminator', dloss, it=it)
            logger.add('losses', 'regularizer', reg, it=it)

            # Generators updates
            gloss = trainer.generator_trainstep(y, z)
            logger.add('losses', 'generator', gloss, it=it)

            if config['training']['take_model_average']:
                update_average(generator_test, generator, beta=config['training']['model_average_beta'])

            # Print stats
            if it % 200 == 0:
                g_loss_last = logger.get_last('losses', 'generator')
                d_loss_last = logger.get_last('losses', 'discriminator')
                d_reg_last = logger.get_last('losses', 'regularizer')
                print('init: [it %4d, n %4dk] g_loss = %.4f, d_loss = %.4f, reg=%.4f'
                    % (it, count // 1000, g_loss_last, d_loss_last, d_reg_last))

            # Sample
            if it % 10000 == 0:
                print('Creating samples...')
                x = evaluator.create_samples(z_test, clusterer.get_labels(x_test, y_test).to(device))
                logger.add_imgs(x, 'all', count // 1000)

                cat_imgs = []
                for y_inst in range(num_k):
                    x = evaluator.create_samples(z_test, y_inst)
                    logger.add_imgs(x, '%04d' % y_inst, count // 1000)
                    cat_imgs.append(x[:8])

                cat_imgs = torch.cat(cat_imgs, dim=0)
                logger.add_imgs(cat_imgs, 'cat', count // 1000, nrow=8)

        print('Saving backup...')
        logger.save_stats('stats.p')
        checkpoint_io.save('model.pt', epoch_idx=epoch_idx)
    # acrp training loop
    elif config['training']['stage'] == "crp":
        print('Start crp training from epoch %d ...' % (epoch_idx))
        while epoch_idx < config['training']['crp_epoch']:
            epoch_idx += 1

            # training encoder
            print('Start encoder training...')
            
            encoder_type=config['training']['encoder_type']
            count_q = 0
            it = 0
            while count_q < config['training']['n_q']:
                count_q += batch_size
                it += 1
                if encoder_type in {"l2", "l1", "l1_margin", "l2_margin"}:
                    y = clusterer.sample_y(batch_size)
                    target_embeds = mg_module.get_means(y)
                else:
                    raise NotImplementedError
                
                z = zdist.sample((batch_size, ))
                q_loss = trainer.encoder_trainstep(y, z, target_embeds)
                if it % 100 == 0:
                    print('encoder: [epoch %0d n_q %4dk] q_loss = %.4f' % (epoch_idx, count_q//1000, q_loss))
            
            purity_score = evaluator.compute_purity_score()
            print('[epoch %0d] purity: %.4f' % (epoch_idx, purity_score))

            if config['training']['crp_epoch'] > 0 and epoch_idx <= config['training']['crp_epoch']:
                # collect embeddings
                print('Start embedding collection...')
                embedding_ims = torch.zeros(image_num, config['multi_gauss']['embed_dim']).to(device)
                with torch.no_grad():
                    for x_real, _, index in eval_loader:
                        x_real = x_real.to(device)
                        im_embeddings = encoder(x_real)
                        embedding_ims[index, :] = im_embeddings
                
                # crp process
                print('Start CRP...')
                mid_results, record_multi_gauss = clusterer.crp(embedding_ims, record=True, dim_reduce=config['multi_gauss']['dim_reduce'])

                # vis count distribution
                y_range, x_ticks = None, None
                for e1 in range(len(mid_results)):
                    for e2 in range(len(mid_results[e1])):
                        mid_picked_class = mid_results[e1][e2]
                        filename=f'{epoch_idx}_{e1}_{e2}'
                        y_range, x_ticks = logger.vis_real_data_training_procedure(mid_picked_class, data_label_ims, num_k, label_num, \
                            filename, y_range, x_ticks)
                
                # add prior distribution
                logger.add('distribution', 'distribution', clusterer.distribution, epoch_idx)

            # traininig G & D
            count_gd = 0
            it = 0
            dataiter = iter(train_loader)
            while count_gd < config['training']['n_gd']:
                # for x_real, y, index in train_loader:
                try:
                    x_real, y, index = next(dataiter)
                except StopIteration:
                    dataiter = iter(train_loader)
                    x_real, y, index = next(dataiter)

                count_gd += batch_size
                it += 1

                x_real, y = x_real.to(device), y.to(device)
                z = zdist.sample((batch_size, ))
                y = clusterer.get_labels(index).to(device)
                
                # Discriminator updates
                dloss, reg = trainer.discriminator_trainstep(x_real, y, z)

                logger.add('losses', 'discriminator', dloss, it=it)
                logger.add('losses', 'regularizer', reg, it=it)

                # Generators updates
                gloss = trainer.generator_trainstep(y, z)
                logger.add('losses', 'generator', gloss, it=it)

                if config['training']['take_model_average']:
                    update_average(generator_test, generator, beta=config['training']['model_average_beta'])

                # Print stats
                if it % 200 == 0:
                    g_loss_last = logger.get_last('losses', 'generator')
                    d_loss_last = logger.get_last('losses', 'discriminator')
                    d_reg_last = logger.get_last('losses', 'regularizer')
                    print('GANs: [epoch %0d, n %4dk] g_loss = %.4f, d_loss = %.4f, reg=%.4f'
                        % (epoch_idx, count_gd // 1000, g_loss_last, d_loss_last, d_reg_last))

            # Sample
            print('Creating samples...')
            x = evaluator.create_samples(z_test, clusterer.sample_y(z_test.shape[0]).to(device))
            logger.add_imgs(x, 'all', epoch_idx)

            cat_imgs = []
            for y_inst in range(num_k):
                x = evaluator.create_samples(z_test, y_inst)
                logger.add_imgs(x, '%04d' % y_inst, epoch_idx)
                cat_imgs.append(x[:8])

            cat_imgs = torch.cat(cat_imgs, dim=0)
            logger.add_imgs(cat_imgs, 'cat', epoch_idx, nrow=8)

            # Backup
            print('Saving backup...')
            checkpoint_io.save('model_%08d.pt' % epoch_idx, epoch_idx=epoch_idx)
            logger.save_stats('stats_%08d.p' % epoch_idx)
            checkpoint_io.save('model.pt', epoch_idx=epoch_idx)

if __name__ == '__main__':
    main()
