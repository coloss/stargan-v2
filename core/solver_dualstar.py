"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from os.path import join as ospj
import time
import datetime
from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import build_model_dualstargan
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils
from metrics.eval import calculate_metrics
from core.solver_base import SolverBase, adv_loss, r1_reg, moving_average
from metrics.vgg import VGG19Loss

import numpy as np
import matplotlib.pyplot as plt

class SolverDualStar(SolverBase):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        # if len(self.args.vgg_loss_layers ) > 0:
        #     assert len(self.args.vgg_loss_layers) == len(self.args.lambda_vgg)
        #     self.vgg_loss = VGG19Loss(dict(zip(self.args.vgg_loss_layers, self.args.lambda_vgg)))
        return build_model_dualstargan(self.args)

    def _create_experiment_name(self):
        name = "DualStarGAN"
        name += "_" + "-".join(self.args.domain_names)
        return name

    def _configure_optimizers(self):
        optimizers = Munch()
        for net in self.nets.keys():
            if net == 'fan':
                continue
            optimizers[net] = torch.optim.Adam(
                params=self.nets[net].parameters(),
                lr=self.args.f_lr if net == 'mapping_network' else self.args.lr,
                betas=(self.args.beta1, self.args.beta2),
                weight_decay=self.args.weight_decay)
        return optimizers

    def _evaluate(self, step):
        # metrics_latent = calculate_metrics(self.nets_ema, self.args, step + 1, mode='latent')
        metrics_ref = calculate_metrics(self.nets_ema, self.args, step + 1, mode='reference')
        # return {**metrics_latent, **metrics_ref}
        return {**metrics_ref}

    def _generate_images(self, inputs, step):
        return utils.debug_image(self.nets_ema, self.args, inputs=inputs, step=step)

    def _training_step(self, inputs, step):
        # x_real, y_org = inputs.x_src, inputs.y_src
        # x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref
        # z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2

        x_real, y_real = inputs.x_src
        x_labels, y_labels = inputs.y_src

        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        if args.w_hpf > 0:
            if args.direction == 'bi':
                masks = nets.fan.get_heatmap(torch.cat([x_real, y_real], dim=0))
            elif args.direction == 'x2y':
                masks = nets.fan.get_heatmap(torch.cat(x_real, dim=0))
                masks = torch.cat([masks, masks], dim=0)
            elif args.direction == 'y2x':
                masks = nets.fan.get_heatmap(torch.cat(y_real, dim=0))
                masks = torch.cat([masks, masks], dim=0)
            else:
                raise ValueError(f"Invalid translation direction: {self.args.direction}")
        else:
            masks = None

        # idx = 0
        # for idx in range(1, x_real.shape[0]):
        #     x_real_np = np.transpose(x_real.detach().cpu().numpy()[idx], [1,2,0])
        #     x_ref_np = np.transpose(x_ref.detach().cpu().numpy()[idx], [1,2,0])
        #     x_ref2_np = np.transpose(x_ref2.detach().cpu().numpy()[idx], [1,2,0])
        #     masks0_np = np.transpose(masks[0].detach().cpu().numpy()[idx], [1,2,0])
        #     masks1_np = np.transpose(masks[1].detach().cpu().numpy()[idx], [1,2,0])

        #     plt.figure()
        #     plt.imshow(x_real_np)
        #     plt.figure()
        #     plt.imshow(x_ref_np)
        #     plt.figure()
        #     plt.imshow(x_ref2_np)
        #     plt.figure()
        #     plt.imshow(masks0_np)
        #     plt.figure()
        #     plt.imshow(masks1_np)
        # plt.show()

        # train the discriminator
        d_loss, d_losses_ref = compute_d_loss(
            nets, args, x_real, y_real, x_labels, y_labels, masks=masks)
        self._reset_grad()
        d_loss.backward()
        optims.discriminator.step()

        # train the generator
        g_loss, g_losses_ref = compute_g_loss(
            nets, args, x_real, y_real, x_labels, y_labels, masks=masks)
        self._reset_grad()
        g_loss.backward()
        optims.style_encoder.step()
        optims.generator.step()

        # compute moving average of network parameters
        moving_average(nets.generator, nets_ema.generator, beta=0.999)
        # if nets.mapping_network is not None:
        #     moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
        moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

        # decay weight for diversity sensitive loss
        if args.lambda_ds > 0:
            args.lambda_ds -= (self.initial_lambda_ds / args.ds_iter)

        losses = {}
        dicts = [d_losses_ref, g_losses_ref]
        prefixes = ['D/ref_', 'G/ref_']

        for li in range(len(prefixes)):
            d = dicts[li]
            prefix = prefixes[li]
            for key, value in d.items():
                losses[prefix + key] = value

        return losses


    @torch.no_grad()
    def sample(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))
        ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test'))

        fname = ospj(args.result_dir, 'reference.jpg')
        print('Working on {}...'.format(fname))
        utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname)

        fname = ospj(args.result_dir, 'video_ref.mp4')
        print('Working on {}...'.format(fname))
        utils.video_ref(nets_ema, args, src.x, ref.x, ref.y, fname)

    @torch.no_grad()
    def evaluate(self):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        # calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')
        calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')


def compute_d_loss(nets, args, x_real, y_real, x_labels, y_labels, masks=None):

    full_image_batch_real = torch.cat([x_real, y_real], dim=0)
    full_label_batch_real = torch.cat([x_labels, y_labels], dim=0)
    # with real images
    full_image_batch_real.requires_grad_()

    out = nets.discriminator(full_image_batch_real, full_label_batch_real)

    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, full_image_batch_real)

    # with fake images
    with torch.no_grad():
        s_real = nets.style_encoder(full_image_batch_real, full_label_batch_real)

    s_x_real = s_real[:x_real.shape[0]]
    s_y_real = s_real[x_real.shape[0]:]

    # flip x and y for target styles
    s_target = torch.cat([s_y_real, s_x_real], dim=0)

    # generate x with style of y and vice versa
    fake_image_batch = nets.generator(full_image_batch_real, s_target, masks=masks)

    full_label_batch_fake = torch.cat([y_labels, x_labels], dim=0)
    out = nets.discriminator(fake_image_batch, full_label_batch_fake)
    loss_fake = adv_loss(out, 0)

    loss = args.lambda_d_real* loss_real + args.lambda_d_fake * loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())


def compute_g_loss(nets, args, x_real, y_real, x_labels, y_labels, masks=None):
    # assert (z_trgs is None) != (x_refs is None)
    # if z_trgs is not None:
    #     z_trg, z_trg2 = z_trgs
    # if x_refs is not None:
    #     x_ref, x_ref2 = x_refs

    full_image_batch_real = torch.cat([x_real, y_real], dim=0)
    full_label_batch_real = torch.cat([x_labels, y_labels], dim=0)
    full_label_batch_fake = torch.cat([y_labels, x_labels], dim=0)

    # # adversarial loss
    # if z_trgs is not None:
    #     s_trg = nets.mapping_network(z_trg, y_trg)
    # else:
    s_real = nets.style_encoder(full_image_batch_real, full_label_batch_real)

    style_x_real = s_real[:x_real.shape[0]]
    style_y_real = s_real[x_real.shape[0]:]

    # flip x and y for target styles
    s_target = torch.cat([style_y_real, style_x_real], dim=0)

    full_image_batch_fake = nets.generator(full_image_batch_real, s_target, masks=masks)
    out = nets.discriminator(full_image_batch_fake, full_label_batch_fake)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss

    # style(G(x)) == style(y)
    # style(x) == style(G(y))
    s_fake = nets.style_encoder(full_image_batch_fake, full_label_batch_fake) #style of translated fake images

    # style_x_real = s_real[:x_real.shape[0]]
    # style_y_real = s_real[x_real.shape[0]:]

    style_x_fake = s_fake[x_real.shape[0]:]
    style_y_fake = s_fake[:x_real.shape[0]]

    style_real = torch.cat([style_x_real, style_y_real], dim=0)
    style_fake = torch.cat([style_x_fake, style_y_fake], dim=0)

    loss_sty = torch.mean(torch.abs(style_real - style_fake))

    # diversity sensitive loss # does not apply here, no random style
    # if z_trgs is not None:
    #     s_trg2 = nets.mapping_network(z_trg2, y_trg)
    # else:
    #     s_trg2 = nets.style_encoder(x_ref2, y_trg)
    # x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
    # x_fake2 = x_fake2.detach()
    # loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # cycle-consistency loss
    # x_fake = full_image_batch_fake[x_real.shape[0]:]
    # y_fake = full_image_batch_fake[:x_real.shape[0]]

    masks = nets.fan.get_heatmap(full_image_batch_fake) if args.w_hpf > 0 else None

    full_image_batch_rec_real_style = nets.generator(full_image_batch_fake, s_real, masks=masks)
    full_image_batch_rec_fake_style = nets.generator(full_image_batch_fake, style_fake, masks=masks)

    loss_cyc_real_style = torch.mean(torch.abs(full_image_batch_rec_real_style - full_image_batch_real))
    loss_cyc_fake_style = torch.mean(torch.abs(full_image_batch_rec_fake_style - full_image_batch_real))

    loss = loss_adv + args.lambda_sty * loss_sty\
           + args.lambda_cyc_real_style * loss_cyc_real_style \
           + args.lambda_cyc_fake_style * loss_cyc_fake_style
           #- args.lambda_ds * loss_ds
    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       cyc_real_style=loss_cyc_real_style.item(),
                       cyc_fake_style=loss_cyc_fake_style.item(),
                       # cyc=loss_cyc.item(),
                       # ds=loss_ds.item(),
                       )
