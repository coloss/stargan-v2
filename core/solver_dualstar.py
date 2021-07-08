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
from metrics.FRNet import VGGFace2Loss
from metrics.EmoNetLoss import EmoNetLoss

import numpy as np
import matplotlib.pyplot as plt

class SolverDualStar(SolverBase):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        if self.args.direction not in ['bi', 'x2y', 'y2x']:
            raise ValueError(f"Invalid direction specification: {self.args.direction}")

        self.loss_nets = Munch()

        if self.args.lambda_vgg != 0:
            assert len(self.args.vgg_loss_layers) == len(self.args.lambda_vgg_layers)
            self.loss_nets['vgg_loss'] = VGG19Loss(dict(zip(self.args.vgg_loss_layers, self.args.lambda_vgg_layers)))
        else:
            self.loss_nets['vgg_loss'] = None

        if self.args.lambda_face_rec != 0:

            self.loss_nets['facerec_loss'] = VGGFace2Loss(metric='cos', unnormalize=True)
        else:
            self.loss_nets['facerec_loss'] = None

        if self.args.lambda_emo_rec != 0:
            self.loss_nets['emorec_loss'] = EmoNetLoss(unnormalize=True, feat_metric=self.args.metric_emo_rec)
        else:
            self.loss_nets['emorec_loss'] = None
        return build_model_dualstargan(self.args)

    def to(self, *args, **kwargs):
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)

        if dtype is not None:
            if not dtype.is_floating_point:
                raise TypeError('nn.Module.to only accepts floating point '
                                'dtypes, but got desired dtype={}'.format(dtype))

        def convert(t):
            return t.to(device, dtype if t.is_floating_point() else None, non_blocking)

        for key, value in self.loss_nets.items():
            if value is not None:
                self.loss_nets[key] = value._apply(convert)

        return super().to(*args, **kwargs)

    def _create_experiment_name(self):
        name = "DualStarGAN"
        if self.args.arch_type != "star":
            if self.args.arch_type[:len('star')] == 'star':
                name += "_" + self.args.arch_type[len('star'):]
            else:
                name += "_" + self.args.arch_type
        name += "_" + "-".join(self.args.domain_names)
        if self.args.direction != 'bi':
            name += "_" + self.args.direction
        if self.args.lambda_sup_photo is not None and self.args.lambda_sup_photo != 0:
            name += "_SupRec"
            if self.args.lambda_sup_photo != 1.:
                name += f"-{self.args.lambda_sup_photo:0.2f}"
        if self.args.lambda_vgg != 0:
            name += "_VGG"
            if self.args.lambda_vgg != 1.:
                name += f"-{self.args.lambda_vgg:0.2f}"
        if self.args.lambda_face_rec != 0:
            name += "_FR"
            if self.args.lambda_face_rec != 1.:
                name += f"-{self.args.lambda_face_rec:0.2f}"

        if self.args.lambda_emo_rec != 0:
            name += "_ER"
            if self.args.metric_emo_rec is not None:
                name += self.args.metric_emo_rec
            if self.args.lambda_emo_rec != 1.:
                name += f"-{self.args.lambda_emo_rec:0.2f}"
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
                betas=(self.args.beta1, self.args.beta2),
                weight_decay=self.args.weight_decay)
        return optimizers

    def _evaluate(self, step):
        # metrics_latent = calculate_metrics(self.nets_ema, self.args, step + 1, mode='latent')
        metrics_ref = calculate_metrics(self.nets_ema, self.args, step + 1, mode='reference')
        # return {**metrics_latent, **metrics_ref}
        return {**metrics_ref}

    def _generate_images(self, inputs, step):
        return utils.debug_image_paired(self.nets_ema, self.args, inputs=inputs, step=step)
        # return utils.debug_image(self.nets_ema, self.args, inputs=inputs, step=step)

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
                masks = nets.fan.get_heatmap(x_real)
                # masks = torch.cat([masks], dim=0)
            elif args.direction == 'y2x':
                masks = nets.fan.get_heatmap(y_real)
                # masks = torch.cat([masks, masks], dim=0)
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
        d_loss, d_losses_ref = self.compute_d_loss(
            nets, args, x_real, y_real, x_labels, y_labels, masks=masks)
        self._reset_grad()
        d_loss.backward()
        optims.discriminator.step()

        # train the generator
        g_loss, g_losses_ref = self.compute_g_loss(
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
        # calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')
        calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')


    def compute_d_loss(self, nets, args, x_real, y_real, x_labels, y_labels, masks=None):
        if args.direction == 'bi':
            full_image_batch_real = torch.cat([x_real, y_real], dim=0)
            full_label_batch_real = torch.cat([x_labels, y_labels], dim=0)
            full_label_batch_fake = torch.cat([y_labels, x_labels], dim=0)
        elif args.direction == 'x2y':
            full_image_batch_real = x_real
            full_label_batch_real = x_labels
            full_label_batch_fake = y_labels
        elif args.direction == 'y2x':
            full_image_batch_real = y_real
            full_label_batch_real = y_labels
            full_label_batch_fake = x_labels
        else:
            raise ValueError(f"")

        # with real images
        full_image_batch_real.requires_grad_()

        out = nets.discriminator(full_image_batch_real, full_label_batch_real)

        loss_real = adv_loss(out, 1)
        loss_reg = r1_reg(out, full_image_batch_real)

        # with fake images
        with torch.no_grad():
            s_real = nets.style_encoder(full_image_batch_real, full_label_batch_real)

        if args.direction == 'bi':
            s_x_real = s_real[:x_real.shape[0]]
            s_y_real = s_real[x_real.shape[0]:]

            # flip x and y for target styles
            s_target = torch.cat([s_y_real, s_x_real], dim=0)
        elif args.direction == 'x2y':
            s_y_real = s_real
            # flip x and y for target styles
            s_target = s_y_real
        elif args.direction == 'y2x':
            s_x_real = s_real
            # flip x and y for target styles
            s_target = s_x_real
        else:
            raise ValueError(f"")

        # generate x with style of y and vice versa
        fake_image_batch = nets.generator(full_image_batch_real, s_target, masks=masks)

        out = nets.discriminator(fake_image_batch, full_label_batch_fake)
        loss_fake = adv_loss(out, 0)

        loss = args.lambda_d_real* loss_real + args.lambda_d_fake * loss_fake + args.lambda_reg * loss_reg
        return loss, Munch(real=loss_real.item(),
                           fake=loss_fake.item(),
                           reg=loss_reg.item())


    def compute_g_loss(self, nets, args, x_real, y_real, x_labels, y_labels, masks=None):
        # assert (z_trgs is None) != (x_refs is None)
        # if z_trgs is not None:
        #     z_trg, z_trg2 = z_trgs
        # if x_refs is not None:
        #     x_ref, x_ref2 = x_refs

        if args.direction == 'bi':
            full_image_batch_real = torch.cat([x_real, y_real], dim=0)
            full_label_batch_real = torch.cat([x_labels, y_labels], dim=0)
            full_label_batch_fake = torch.cat([y_labels, x_labels], dim=0)
        elif args.direction == 'x2y':
            full_image_batch_real = x_real
            full_label_batch_real = x_labels
            full_label_batch_fake = y_labels
        elif args.direction == 'y2x':
            full_image_batch_real = y_real
            full_label_batch_real = y_labels
            full_label_batch_fake = x_labels
        else:
            raise ValueError(f"")

        # # adversarial loss
        # if z_trgs is not None:
        #     s_trg = nets.mapping_network(z_trg, y_trg)
        # else:

        if args.direction == 'bi':
            s_real = nets.style_encoder(full_image_batch_real, full_label_batch_real)
            style_x_real = s_real[:x_real.shape[0]]
            style_y_real = s_real[x_real.shape[0]:]
            # flip x and y for target styles
            s_target = torch.cat([style_y_real, style_x_real], dim=0)
        elif args.direction == 'x2y':
            style_x_real = nets.style_encoder(x_real, x_labels)
            style_y_real = nets.style_encoder(y_real, y_labels)
            s_target = style_y_real
        elif args.direction == 'y2x':
            style_y_real = nets.style_encoder(y_real, y_labels)
            style_x_real = nets.style_encoder(x_real, x_labels)
            s_target = style_x_real
        else:
            raise ValueError(f"")

        full_image_batch_fake = nets.generator(full_image_batch_real, s_target, masks=masks)
        out = nets.discriminator(full_image_batch_fake, full_label_batch_fake)
        loss_adv = adv_loss(out, 1)

        # style reconstruction loss

        # style(G(x)) == style(y)
        # style(x) == style(G(y))
        s_fake = nets.style_encoder(full_image_batch_fake, full_label_batch_fake) #style of translated fake images

        # style_x_real = s_real[:x_real.shape[0]]
        # style_y_real = s_real[x_real.shape[0]:]

        if args.direction == 'bi':
            style_x_fake = s_fake[x_real.shape[0]:]
            style_y_fake = s_fake[:x_real.shape[0]]
            style_real = torch.cat([style_x_real, style_y_real], dim=0)
            style_fake = torch.cat([style_x_fake, style_y_fake], dim=0)
        elif args.direction == 'x2y':
            style_y_fake = s_fake
            style_fake = s_fake
            style_real = style_y_real
        elif args.direction == 'y2x':
            style_x_fake = s_fake
            style_fake = s_fake
            style_real = style_x_real
        else:
            raise ValueError(f"")

        loss_sty = torch.mean(torch.abs(style_real - style_fake))

        # half-cycle reconstruction loss
        if args.direction == 'bi':
            x_fake = full_image_batch_fake[x_real.shape[0]:]
            y_fake = full_image_batch_fake[:x_real.shape[0]]
            half_cycle_x = torch.mean(torch.abs(x_fake - x_real))
            half_cycle_y = torch.mean(torch.abs(y_fake - y_real))
            half_cycle = half_cycle_x + half_cycle_y

            if self.loss_nets.vgg_loss is not None:
                half_cycle_x_vgg, _ = self.loss_nets.vgg_loss(x_fake, x_real)
                half_cycle_y_vgg, _ = self.loss_nets.vgg_loss(y_fake, y_real)
                half_cycle_vgg = half_cycle_x_vgg + half_cycle_y_vgg

            if self.loss_nets.facerec_loss is not None:
                half_cycle_x_face_rec = self.loss_nets.facerec_loss(x_fake, x_real)
                half_cycle_y_face_rec = self.loss_nets.facerec_loss(y_fake, y_real)
                half_cycle_face_rec = half_cycle_x_face_rec + half_cycle_y_face_rec

            if self.loss_nets.emorec_loss is not None:
                half_cycle_x_emo_rec = self.loss_nets.emorec_loss.compute_loss(x_fake, x_real)[1]
                half_cycle_y_emo_rec = self.loss_nets.emorec_loss.compute_loss(y_fake, y_real)[1]
                half_cycle_emo_rec = half_cycle_x_emo_rec + half_cycle_y_emo_rec

        elif args.direction == 'x2y':
            y_fake = full_image_batch_fake
            half_cycle_y = torch.mean(torch.abs(y_fake - y_real))
            half_cycle = half_cycle_y

            if self.loss_nets.vgg_loss is not None:
                half_cycle_y_vgg, _ = self.loss_nets.vgg_loss(y_fake, y_real)
                half_cycle_vgg = half_cycle_y_vgg

            if self.loss_nets.facerec_loss is not None:
                half_cycle_y_face_rec = self.loss_nets.facerec_loss(y_fake, y_real)
                half_cycle_face_rec = half_cycle_y_face_rec

            if self.loss_nets.emorec_loss is not None:
                half_cycle_y_emo_rec = self.loss_nets.emorec_loss.compute_loss(y_fake, y_real)[1]
                half_cycle_emo_rec = half_cycle_y_emo_rec

        elif args.direction == 'y2x':
            x_fake = full_image_batch_fake
            half_cycle_x = torch.mean(torch.abs(x_fake - x_real))
            half_cycle = half_cycle_x

            if self.loss_nets.vgg_loss is not None:
                half_cycle_x_vgg, _ = self.loss_nets.vgg_loss(x_fake, x_real)
                half_cycle_vgg = half_cycle_x_vgg

            if self.loss_nets.facerec_loss is not None:
                half_cycle_x_face_rec = self.loss_nets.facerec_loss(x_fake, x_real)
                half_cycle_face_rec = half_cycle_x_face_rec

            if self.loss_nets.emorec_loss is not None:
                half_cycle_x_emo_rec = self.loss_nets.emorec_loss.compute_loss(x_fake, x_real)[1]
                half_cycle_emo_rec = half_cycle_x_emo_rec

        else:
            raise ValueError(f"")

        # diversity sensitive loss # does not apply here, no random style
        # if z_trgs is not None:
        #     s_trg2 = nets.mapping_network(z_trg2, y_trg)
        # else:
        #     s_trg2 = nets.style_encoder(x_ref2, y_trg)
        # x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
        # x_fake2 = x_fake2.detach()
        # loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

        # cycle-consistency loss
        if args.direction == 'bi':
            masks = nets.fan.get_heatmap(full_image_batch_fake) if args.w_hpf > 0 else None

            full_image_batch_rec_real_style = nets.generator(full_image_batch_fake, s_real, masks=masks)
            full_image_batch_rec_fake_style = nets.generator(full_image_batch_fake, style_fake, masks=masks)

            loss_cyc_real_style = torch.mean(torch.abs(full_image_batch_rec_real_style - full_image_batch_real))
            loss_cyc_fake_style = torch.mean(torch.abs(full_image_batch_rec_fake_style - full_image_batch_real))

            if self.loss_nets.vgg_loss is not None:
                loss_cyc_vgg_real_style, _ = self.loss_nets.vgg_loss(full_image_batch_rec_real_style, full_image_batch_real)
                loss_cyc_vgg_fake_style, _ = self.loss_nets.vgg_loss(full_image_batch_rec_fake_style, full_image_batch_real)

            if self.loss_nets.facerec_loss is not None:
                loss_cyc_facerec_real_style = self.loss_nets.facerec_loss(full_image_batch_rec_real_style, full_image_batch_real)
                loss_cyc_facerec_fake_style = self.loss_nets.facerec_loss(full_image_batch_rec_fake_style, full_image_batch_real)

            if self.loss_nets.emorec_loss is not None:
                loss_cyc_emorec_real_style = self.loss_nets.emorec_loss.compute_loss(full_image_batch_rec_real_style, full_image_batch_real)[1]
                loss_cyc_emorec_fake_style = self.loss_nets.emorec_loss.compute_loss(full_image_batch_rec_fake_style, full_image_batch_real)[1]

        else:
            loss_cyc_real_style = 0
            loss_cyc_fake_style = 0

        loss = loss_adv + args.lambda_sty * loss_sty\
               + args.lambda_cyc_real_style * loss_cyc_real_style \
               + args.lambda_cyc_fake_style * loss_cyc_fake_style \
               + args.lambda_sup_photo * half_cycle
               #- args.lambda_ds * loss_ds

        if self.loss_nets.vgg_loss is not None:
            if self.args.direction == 'bi':
                loss_cyc_vgg = 0.5*loss_cyc_vgg_real_style + 0.5*loss_cyc_vgg_fake_style
                loss += self.args.lambda_vgg * loss_cyc_vgg
            loss += self.args.lambda_vgg * half_cycle_vgg


        if self.loss_nets.facerec_loss is not None:
            if self.args.direction == 'bi':
                loss_cyc_facerec = 0.5*loss_cyc_facerec_real_style + 0.5*loss_cyc_facerec_fake_style
                loss += self.args.lambda_face_rec * loss_cyc_facerec
            loss += self.args.lambda_face_rec * half_cycle_face_rec

        if self.loss_nets.emorec_loss is not None:
            if self.args.direction == 'bi':
                loss_cyc_emorec = 0.5*loss_cyc_emorec_real_style + 0.5*loss_cyc_emorec_fake_style
                loss += self.args.lambda_emo_rec * loss_cyc_emorec
            loss += self.args.lambda_emo_rec * half_cycle_emo_rec


        metrics = Munch(adv=loss_adv.item(),
                           sty=loss_sty.item(),
                           )
        if args.direction == 'bi':
            metrics['sup_rec_x2y'] = half_cycle_x.item()
            metrics['sup_rec_y2x'] = half_cycle_y.item()
            metrics['sup_rec'] = half_cycle.item()
            metrics['cyc_real_style'] = loss_cyc_real_style.item()
            metrics['cyc_fake_style'] = loss_cyc_fake_style.item()
            if self.loss_nets.vgg_loss is not None:
                metrics['sup_rec_vgg_x2y'] = half_cycle_x_vgg.item()
                metrics['sup_rec_vgg_y2x'] = half_cycle_y_vgg.item()
                metrics['sup_rec_vgg'] = half_cycle_vgg.item()
                metrics['loss_cyc_vgg_real_style'] = loss_cyc_vgg_real_style.item()
                metrics['loss_cyc_vgg_fake_style'] = loss_cyc_vgg_fake_style.item()

            if self.loss_nets.facerec_loss is not None:
                metrics['sup_rec_facerec_x2y'] = half_cycle_x_face_rec.item()
                metrics['sup_rec_facerec_y2x'] = half_cycle_y_face_rec.item()
                metrics['sup_rec_facerec'] = half_cycle_face_rec.item()
                metrics['loss_cyc_facerec_real_style'] = loss_cyc_facerec_real_style.item()
                metrics['loss_cyc_facerec_fake_style'] = loss_cyc_facerec_fake_style.item()

            if self.loss_nets.emorec_loss is not None:
                metrics['sup_rec_emorec_x2y'] = half_cycle_x_emo_rec.item()
                metrics['sup_rec_emorec_y2x'] = half_cycle_y_emo_rec.item()
                metrics['sup_rec_emorec'] = half_cycle_emo_rec.item()
                metrics['loss_cyc_emorec_real_style'] = loss_cyc_emorec_real_style.item()
                metrics['loss_cyc_emorec_fake_style'] = loss_cyc_emorec_fake_style.item()

        elif args.direction == 'x2y':
            metrics['sup_rec_y2x'] = half_cycle_y.item()
            if self.loss_nets.vgg_loss is not None:
                metrics['sup_vgg_y2x'] = half_cycle_y_vgg.item()

            if self.loss_nets.facerec_loss is not None:
                metrics['sup_facerec_y2x'] = half_cycle_y_face_rec.item()

            if self.loss_nets.emorec_loss is not None:
                metrics['sup_emorec_y2x'] = half_cycle_y_emo_rec.item()

        elif args.direction == 'y2x':
            metrics['sup_rec_x2y'] = half_cycle_x.item()
            if self.loss_nets.vgg_loss is not None:
                metrics['sup_vgg_x2y'] = half_cycle_x_vgg.item()

            if self.loss_nets.facerec_loss is not None:
                metrics['sup_facerec_x2y'] = half_cycle_x_face_rec.item()

            if self.loss_nets.emorec_loss is not None:
                metrics['sup_emorec_x2y'] = half_cycle_x_emo_rec.item()
        else:
            raise ValueError(f"")
        return loss, metrics
