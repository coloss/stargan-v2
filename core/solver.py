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

from core.model import build_model_stargan
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils
from metrics.eval import calculate_metrics
from core.solver_base import SolverBase, adv_loss, r1_reg, moving_average

import numpy as np
import matplotlib.pyplot as plt

class Solver(SolverBase):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        return build_model_stargan(self.args)

    def _create_experiment_name(self):
        name = "StarGAN"
        if self.args.arch_type != "star":
            if self.args.arch_type[:len('star')] == 'star':
                name += "_" + self.args.arch_type[len('star'):]
            else:
                name += "_" + self.args.arch_type
        if self.args.latent_dim == 0:
            name += "_noZ"
        name += "_" + "-".join(self.args.domain_names)
        if self.args.style_dim != 64:
            name += f"_st{self.args.style_dim}d"
        return name

    def _configure_optimizers(self):
        optimizers = Munch()
        for net in self.nets.keys():
            if net == 'fan':
                continue
            optimizers[net] = torch.optim.Adam(
                params=self.nets[net].parameters(),
                lr=self.args.f_lr if net == 'mapping_network' else self.args.lr,
                betas=[self.args.beta1, self.args.beta2],
                weight_decay=self.args.weight_decay)
        return optimizers

    def _evaluate(self, step):
        metrics_ref = calculate_metrics(self.nets_ema, self.args, step + 1, mode='reference')
        if self.args.latent_dim == 0:
            return metrics_ref
        metrics_latent = calculate_metrics(self.nets_ema, self.args, step + 1, mode='latent')
        return {**metrics_latent, **metrics_ref}
        # metrics = {}
        # for key in metrics_latent:
        #     metrics[key + "_latent"] = metrics_latent[key]
        # for key in metrics_ref:
        #     metrics[key + "_ref"] = metrics_ref[key]

    def _generate_images(self, inputs, step):
        return utils.debug_image(self.nets_ema, self.args, inputs=inputs, step=step)

    def _training_step(self, inputs, step):
        x_real, y_org = inputs.x_src, inputs.y_src
        x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref

        if self.args.latent_dim > 0:
            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2

        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        masks = nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None

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

        if self.args.latent_dim > 0:
            d_loss, d_losses_latent = compute_d_loss(
                nets, args, x_real, y_org, y_trg, z_trg=z_trg, masks=masks)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

        d_loss, d_losses_ref = compute_d_loss(
            nets, args, x_real, y_org, y_trg, x_ref=x_ref, masks=masks)
        self._reset_grad()
        d_loss.backward()
        optims.discriminator.step()

        # train the generator
        if self.args.latent_dim > 0:
            g_loss, g_losses_latent = compute_g_loss(
                nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], masks=masks)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.mapping_network.step()
            optims.style_encoder.step()

        g_loss, g_losses_ref = compute_g_loss(
            nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], masks=masks)
        self._reset_grad()
        g_loss.backward()
        optims.generator.step()
        if self.args.latent_dim == 0:
            optims.style_encoder.step() # when no mapping network, update style encoder here (otherwise it would be nowhere)

        # compute moving average of network parameters
        moving_average(nets.generator, nets_ema.generator, beta=0.999)
        if self.args.latent_dim > 0:
            moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
        moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

        # decay weight for diversity sensitive loss
        if args.lambda_ds > 0:
            args.lambda_ds -= (self.initial_lambda_ds / args.ds_iter)

        losses = {}
        if self.args.latent_dim > 0:
            dicts = [d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref]
            prefixes = ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']
        else:
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
        self._load_checkpoint()
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
        self._load_checkpoint()
        if self.args.latent_dim > 0:
            calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')
        calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')


def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    # with fake images
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)

        x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = args.lambda_d_real * loss_real + args.lambda_d_fake * loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())


def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)

    x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
    x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # cycle-consistency loss
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org, masks=masks)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    loss = loss_adv + args.lambda_sty * loss_sty \
        - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc
    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       ds=loss_ds.item(),
                       cyc=loss_cyc.item())
