"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import sys
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
from pytorch_lightning.loggers import WandbLogger
from wandb import Image

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from omegaconf import OmegaConf, DictConfig

class SolverBase(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets, self.nets_ema = self._build_model()
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)


        t = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")

        if (not hasattr(self.args, 'run_dir')) or self.args.run_dir == '' or self.args.run_dir == 'todo':
            experiment_name = self._create_experiment_name()
            version = t + "_" + experiment_name
            self.args.run_dir = str(Path(self.args.expr_dir) / version)
        else:
            version = Path(self.args.run_dir).name
            t = version[:len(t)]
            experiment_name = version[len(t)+1 :]

        self.args.checkpoint_dir = str(Path(self.args.run_dir) / "checkpoints")
        self.args.sample_dir = str(Path(self.args.run_dir) / "samples")
        self.args.eval_dir = str(Path(self.args.run_dir) / "eval")

        cfg_path = Path(self.args.run_dir) / "config.yaml"
        if not cfg_path.is_file():
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cfg_path, 'w') as outfile:
                OmegaConf.save(config=vars(self.args), f=outfile)
        else:
            with open(cfg_path, 'r') as infile:
                cfg_old = OmegaConf.to_container(OmegaConf.load(infile))
            cfg_current = vars(self.args)
            # if cfg_old != cfg_current:
            #     print("The old and current configs do not match!")
            #     sys.exit()

        if str(args.logger).lower() == 'wandb':
            short_name = experiment_name[:min(128, len(experiment_name))]
            short_version = version[:min(128, len(version))]
            self.logger = WandbLogger(name=short_name,
                                       project="StarGAN",
                                       config=vars(self.args),
                                       version=short_version,
                                       save_dir=str(Path(args.checkpoint_dir).parent.absolute()))
            self.logger.finalize("")
        else:
            self.logger = None

        if args.mode == 'train':
            self.optims = self._configure_optimizers()
            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=True, **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)]
        else:
            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema)]

        self.to(self.device)
        self._initialize()

    def _configure_optimizers(self):
        raise NotImplementedError()

    def _initialize(self):
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

    def _create_experiment_name(self):
        raise NotImplementedError()

    def _build_model(self):
        raise NotImplementedError()

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step=None):
        if step is not None:
            self.args.resume_iter = step

        if self.args.resume_iter is None:
            return
        if isinstance(self.args.resume_iter, str):
            if self.args.resume_iter == 'latest':
                path = Path(self.ckptios[0].fname_template)
                ckpts = list(path.parent.glob("*.ckpt"))
                ckpts.sort(reverse=True)
                num = ckpts[0].name.split("_")[0]
                self.args.resume_iter = int(num)
            else:
                raise ValueError(f"Invalid resume_iter value: {self.args.resume_iter}")

        if self.args.resume_iter is not None and not isinstance(self.args.resume_iter, int):
            raise ValueError(f"Invalid resume_iter value: {self.args.resume_iter}")

        step = self.args.resume_iter
        if step > 0:
            for ckptio in self.ckptios:
                ckptio.load(step)


    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def _training_step(self, inputs, step):
        raise NotImplementedError()

    def _evaluate(self, step):
        raise NotImplementedError()

    def _generate_images(self, inputs, step):
        raise NotImplementedError()

    def test_paired_images(self, loaders):
        self._load_checkpoint()
        step = self.args.resume_iter
        fetcher_val = InputFetcher(loaders.val, None, self.args.latent_dim, 'val')
        inputs_val = next(fetcher_val)
        out_path = Path(self.args.sample_dir) / "test"
        out_path.mkdir(exist_ok=True, parents=True)
        print("Generating paired cycle consistency image")
        image_fname_dict = utils.debug_image_paired(self.nets_ema, self.args, inputs=inputs_val, step=step,
                                                    outdir=str(out_path))
        print("Image generated")
        if self.logger is not None:
            if isinstance(self.logger, WandbLogger):
                image_dict = {}
                for name, path in image_fname_dict.items():
                    image_dict["test_images/" + name] = Image(path)
                self.logger.log_metrics(image_dict)
                print("Image saved to wandb")

    @torch.no_grad()
    def test_on_validation_set(self, loaders):
        from core.utils import translate_and_reconstruct_no_image
        from metrics.EmoNetLoss import EmoNetLoss
        from metrics.FRNet import VGGFace2Loss
        from metrics.vgg import VGG19Loss

        self._load_checkpoint()
        # fetcher_val = InputFetcher(loaders.val, None, self.args.latent_dim, 'val')
        out_path = Path(self.args.sample_dir) / "test"
        out_path.mkdir(exist_ok=True, parents=True)
        print("Testing on validation set ")

        # fetcher_val = InputFetcher(loaders.val, None, self.args.latent_dim, 'val')
        # inputs= next(fetcher_val)

        pixel_loss_x2y = []
        vgg_loss_x2y = []
        fr_loss_x2y = []
        emonet_f1_loss_x2y = []
        emonet_f2_loss_x2y = []
        emonet_v_loss_x2y = []
        emonet_a_loss_x2y = []
        emonet_exp_loss_x2y = []
        adv_loss_x2y = []

        pixel_loss_x2y2x = []
        vgg_loss_x2y2x = []
        fr_loss_x2y2x = []
        emonet_f1_loss_x2y2x = []
        emonet_f2_loss_x2y2x = []
        emonet_v_loss_x2y2x = []
        emonet_a_loss_x2y2x = []
        emonet_exp_loss_x2y2x = []
        adv_loss_x2y2x = []

        pixel_loss_y2x = []
        vgg_loss_y2x = []
        fr_loss_y2x = []
        emonet_f1_loss_y2x = []
        emonet_f2_loss_y2x = []
        emonet_v_loss_y2x = []
        emonet_a_loss_y2x = []
        emonet_exp_loss_y2x = []
        adv_loss_y2x = []

        pixel_loss_y2x2y = []
        vgg_loss_y2x2y = []
        fr_loss_y2x2y = []
        emonet_f1_loss_y2x2y = []
        emonet_f2_loss_y2x2y = []
        emonet_v_loss_y2x2y = []
        emonet_a_loss_y2x2y = []
        emonet_exp_loss_y2x2y = []
        adv_loss_y2x2y = []

        vgg_loss = VGG19Loss(dict(zip(self.args.vgg_loss_layers, self.args.lambda_vgg_layers)))
        facerec_loss = VGGFace2Loss(metric='cos', unnormalize=True)
        emorec_loss = EmoNetLoss(unnormalize=True, feat_metric=self.args.metric_emo_rec)

        vgg_loss.cuda()
        facerec_loss.cuda()
        emorec_loss.cuda()

        fetcher_val = InputFetcher(loaders.val, None, self.args.latent_dim, 'val')

        from tqdm import auto

        for bi in auto.tqdm(range(len(loaders.val) // loaders.val.batch_size)):
            if bi == 10:
                break
            inputs = next(fetcher_val)
            x_src_, y_src_ = inputs.x_src
            x_ref_, y_ref_ = inputs.x_ref

            x_src_label, y_src_label = inputs.y_src
            x_ref_label, y_ref_label = inputs.y_ref

            # 1 ) one way
            if self.args.direction in ['bi', 'x2y']:
                x_all_src = torch.cat([x_src_, x_ref_, ], dim=0)
                x_all_ref = torch.cat([y_src_, y_ref_, ], dim=0)
                labels_all_src = torch.cat([x_src_label, x_ref_label, ], dim=0)
                labels_all_ref = torch.cat([y_src_label, y_ref_label, ], dim=0)

                # translate and reconstruct (reference-guided)
                x_src, x_ref, x_fake, x_rec = translate_and_reconstruct_no_image(
                    self.nets, self.args, x_all_src, labels_all_src, x_all_ref, labels_all_ref)

                # SUPERVISED
                # photometric
                half_cycle_x = torch.mean(torch.abs(x_fake - x_ref))
                pixel_loss_x2y += [half_cycle_x.item()]

                # vgg feature loss
                if vgg_loss is not None:
                    half_cycle_x_vgg, _ = vgg_loss(x_fake, x_ref)
                    half_cycle_vgg = half_cycle_x_vgg
                    vgg_loss_x2y += [half_cycle_x_vgg.item()]

                # facerec loss
                if facerec_loss is not None:
                    half_cycle_x_face_rec = facerec_loss(x_fake, x_ref)
                    half_cycle_face_rec = half_cycle_x_face_rec
                    fr_loss_x2y += [half_cycle_face_rec.item()]

                # emonet loss
                if emorec_loss is not None:
                    emo_feat_loss_1, emo_feat_loss_2, valence_loss, arousal_loss, expression_loss = \
                        emorec_loss.compute_loss(x_fake, x_ref)
                    emonet_f1_loss_x2y += [emo_feat_loss_1.item()]
                    emonet_f2_loss_x2y += [emo_feat_loss_2.item()]
                    emonet_v_loss_x2y += [valence_loss.item()]
                    emonet_a_loss_x2y += [arousal_loss.item()]
                    emonet_exp_loss_x2y += [expression_loss.item()]
                # adv loss

                out = F.sigmoid(self.nets.discriminator(x_fake, labels_all_ref)).mean().item()
                adv_loss_x2y += [out]

                # out = self.nets.discriminator(x_fake, labels_all_ref)
                # adv_loss_x2y += [adv_loss(out, 1).item()]

                # CYCLE
                # photometric
                half_cycle_x = torch.mean(torch.abs(x_src - x_rec))
                pixel_loss_x2y2x += [half_cycle_x.item()]

                # vgg feature loss
                if vgg_loss is not None:
                    half_cycle_x_vgg, _ = vgg_loss(x_src, x_rec)
                    half_cycle_vgg = half_cycle_x_vgg
                    vgg_loss_x2y2x += [half_cycle_x_vgg.item()]

                # facerec loss
                if facerec_loss is not None:
                    half_cycle_x_face_rec = facerec_loss(x_src, x_rec)
                    half_cycle_face_rec = half_cycle_x_face_rec
                    fr_loss_x2y2x += [half_cycle_face_rec.item()]

                # emonet loss
                if emorec_loss is not None:
                    emo_feat_loss_1, emo_feat_loss_2, valence_loss, arousal_loss, expression_loss = \
                        emorec_loss.compute_loss(x_src, x_rec)

                    emonet_f1_loss_x2y2x += [emo_feat_loss_1.item()]
                    emonet_f2_loss_x2y2x += [emo_feat_loss_2.item()]
                    emonet_v_loss_x2y2x += [valence_loss.item()]
                    emonet_a_loss_x2y2x += [arousal_loss.item()]
                    emonet_exp_loss_x2y2x += [expression_loss.item()]

                # adv loss
                out = F.sigmoid(self.nets.discriminator(x_rec, labels_all_src)).mean().item()
                adv_loss_x2y2x += [out]

                # out = self.nets.discriminator(x_rec, labels_all_src)
                # adv_loss_x2y2x += [adv_loss(out, 1).item()]

            # 2) other way
            if self.args.direction in ['bi', 'y2x']:
                x_all_src = torch.cat([y_src_, y_ref_, ], dim=0)
                x_all_ref = torch.cat([x_src_, x_ref_, ], dim=0)
                labels_all_src = torch.cat([y_src_label, y_ref_label, ], dim=0)
                labels_all_ref = torch.cat([x_src_label, x_ref_label, ], dim=0)

                # translate and reconstruct (reference-guided)
                x_src, x_ref, x_fake, x_rec = translate_and_reconstruct_no_image(
                    self.nets, self.args, x_all_src, labels_all_src, x_all_ref, labels_all_ref)

                # SUPERVISED
                # photometric
                half_cycle_y = torch.mean(torch.abs(x_fake - x_ref))
                pixel_loss_y2x += [half_cycle_y.item()]

                # vgg feature loss
                if vgg_loss is not None:
                    half_cycle_y_vgg, _ = vgg_loss(x_fake, x_ref)
                    half_cycle_vgg = half_cycle_y_vgg
                    vgg_loss_y2x += [half_cycle_y_vgg.item()]

                # facerec loss
                if facerec_loss is not None:
                    half_cycle_y_face_rec = facerec_loss(x_fake, x_ref)
                    half_cycle_face_rec = half_cycle_y_face_rec
                    fr_loss_y2x += [half_cycle_face_rec.item()]

                # emonet loss
                if emorec_loss is not None:
                    emo_feat_loss_1, emo_feat_loss_2, valence_loss, arousal_loss, expression_loss = \
                        emorec_loss.compute_loss(x_fake, x_ref)

                    emonet_f1_loss_y2x += [emo_feat_loss_1.item()]
                    emonet_f2_loss_y2x += [emo_feat_loss_2.item()]
                    emonet_v_loss_y2x += [valence_loss.item()]
                    emonet_a_loss_y2x += [arousal_loss.item()]
                    emonet_exp_loss_y2x += [expression_loss.item()]

                # adv loss
                out = F.sigmoid(self.nets.discriminator(x_fake, labels_all_ref)).mean().item()
                adv_loss_y2x += [out]

                # out = self.nets.discriminator(x_fake, labels_all_ref)
                # adv_loss_y2x += [adv_loss(out, 1).item()]

                # CYCLE
                # photometric
                half_cycle_y = torch.mean(torch.abs(x_src - x_rec))
                pixel_loss_y2x2y += [half_cycle_y.item()]
                # vgg feature loss
                if vgg_loss is not None:
                    half_cycle_y_vgg, _ = vgg_loss(x_src, x_rec)
                    half_cycle_vgg = half_cycle_y_vgg
                    vgg_loss_y2x2y += [half_cycle_y_vgg.item()]

                # facerec loss
                if facerec_loss is not None:
                    half_cycle_y_face_rec = facerec_loss(x_src, x_rec)
                    half_cycle_face_rec = half_cycle_y_face_rec
                    fr_loss_y2x2y += [half_cycle_face_rec.item()]

                # emonet loss
                if emorec_loss is not None:
                    emo_feat_loss_1, emo_feat_loss_2, valence_loss, arousal_loss, expression_loss = \
                        emorec_loss.compute_loss(x_src, x_rec)

                    emonet_f1_loss_y2x2y += [emo_feat_loss_1.item()]
                    emonet_f2_loss_y2x2y += [emo_feat_loss_2.item()]
                    emonet_v_loss_y2x2y += [valence_loss.item()]
                    emonet_a_loss_y2x2y += [arousal_loss.item()]
                    emonet_exp_loss_y2x2y += [expression_loss.item()]

                # adv loss
                out = F.sigmoid(self.nets.discriminator(x_rec, labels_all_src)).mean().item()
                adv_loss_y2x2y += [out]
                # out = self.nets.discriminator(x_rec, labels_all_src)
                # adv_loss_y2x2y += [adv_loss(out, 1).item()]

        import pickle as pkl

        if self.args.direction in ['bi', 'x2y']:
            names_x2y = ["pixel_loss_x2y" ,"vgg_loss_x2y", "fr_loss_x2y", "emonet_f1_loss_x2y", "emonet_f2_loss_x2y", "emonet_v_loss_x2y", "emonet_a_loss_x2y", "emonet_exp_loss_x2y", "adv_loss_x2y"]
            losses_x2y = [pixel_loss_x2y ,vgg_loss_x2y, fr_loss_x2y, emonet_f1_loss_x2y, emonet_f2_loss_x2y, emonet_v_loss_x2y, emonet_a_loss_x2y, emonet_exp_loss_x2y, adv_loss_x2y]
            n2l_x2y = dict(zip(names_x2y, losses_x2y))
            n2l_x2y_mean = {}
            for loss, value in n2l_x2y.items():
                n2l_x2y[loss] = np.array(value)
                n2l_x2y_mean[loss] = n2l_x2y[loss].mean()
            self.logger.log_metrics({f"test/{key}": value for key, value in n2l_x2y_mean.items()})
            self.logger.log_metrics({f"test/{key}_items": value for key, value in n2l_x2y.items()})

            with open(out_path / "losses_x2y.pkl", "wb") as f:
                pkl.dump(n2l_x2y, f)
                pkl.dump(n2l_x2y_mean, f)

            names_x2y2x = ["pixel_loss_x2y2x" , "vgg_loss_x2y2x" , "fr_loss_x2y2x", "emonet_f1_loss_x2y2x", "emonet_f2_loss_x2y2x", "emonet_v_loss_x2y2x" , "emonet_a_loss_x2y2x", "emonet_exp_loss_x2y2x", "adv_loss_x2y2x"]
            losses_x2y2x = [pixel_loss_x2y2x , vgg_loss_x2y2x , fr_loss_x2y2x, emonet_f1_loss_x2y2x, emonet_f2_loss_x2y2x, emonet_v_loss_x2y2x , emonet_a_loss_x2y2x, emonet_exp_loss_x2y2x, adv_loss_x2y2x]
            n2l_x2y2x = dict(zip(names_x2y2x, losses_x2y2x))
            n2l_x2y2x_mean = {}

            for loss, value in n2l_x2y2x.items():
                n2l_x2y2x[loss] = np.array(value)
                n2l_x2y2x_mean[loss] = n2l_x2y2x[loss].mean()
            self.logger.log_metrics({f"test/{key}": value for key, value in n2l_x2y2x_mean.items()})
            self.logger.log_metrics({f"test/{key}_items": value for key, value in n2l_x2y2x.items()})

            with open(out_path / "losses_x2y2x.pkl", "wb") as f:
                pkl.dump(n2l_x2y2x, f)
                pkl.dump(n2l_x2y2x_mean, f)


        if self.args.direction in ['bi', 'y2x']:
            names_y2x = ["pixel_loss_y2x" ,"vgg_loss_y2x", "fr_loss_y2x", "emonet_f1_loss_y2x", "emonet_f2_loss_y2x", "emonet_v_loss_y2x", "emonet_a_loss_y2x", "emonet_exp_loss_y2x", "adv_loss_y2x"]
            losses_y2x = [pixel_loss_y2x ,vgg_loss_y2x, fr_loss_y2x, emonet_f1_loss_y2x, emonet_f2_loss_y2x, emonet_v_loss_y2x, emonet_a_loss_y2x, emonet_exp_loss_y2x, adv_loss_y2x]
            n2l_y2x = dict(zip(names_y2x, losses_y2x))
            n2l_y2x_mean = {}
            for loss, value in n2l_y2x.items():
                n2l_y2x[loss] = np.array(value)
                n2l_y2x_mean[loss] = n2l_y2x[loss].mean()
            self.logger.log_metrics({f"test/{key}": value for key, value in n2l_y2x_mean.items()})
            self.logger.log_metrics({f"test/{key}_items": value for key, value in n2l_y2x.items()})


            with open(out_path / "losses_y2x.pkl", "wb") as f:
                pkl.dump(n2l_y2x, f)
                pkl.dump(n2l_y2x_mean, f)

            names_y2x2y = ["pixel_loss_y2x2y" , "vgg_loss_y2x2y" , "fr_loss_y2x2y", "emonet_f1_loss_y2x2y", "emonet_f2_loss_y2x2y", "emonet_v_loss_y2x2y" , "emonet_a_loss_y2x2y", "emonet_exp_loss_y2x2y", "adv_loss_y2x2y"]
            losses_y2x2y = [pixel_loss_y2x2y , vgg_loss_y2x2y , fr_loss_y2x2y, emonet_f1_loss_y2x2y, emonet_f2_loss_y2x2y, emonet_v_loss_y2x2y , emonet_a_loss_y2x2y, emonet_exp_loss_y2x2y, adv_loss_y2x2y]
            n2l_y2x2y =  dict(zip(names_y2x2y, losses_y2x2y))
            n2l_y2x2y_mean = {}
            for loss, value in n2l_y2x2y.items():
                n2l_y2x2y[loss] = np.array(value)
                n2l_y2x2y_mean[loss] = n2l_y2x2y[loss].mean()
            self.logger.log_metrics({f"test/{key}": value for key, value in n2l_y2x2y_mean.items()})
            self.logger.log_metrics({f"test/{key}_items": value for key, value in n2l_y2x2y.items()})


            with open(out_path / "losses_y2x2y.pkl", "wb") as f:
                pkl.dump(n2l_y2x2y, f)
                pkl.dump(n2l_y2x2y_mean, f)



    def fit(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, loaders.ref, args.latent_dim, 'train')
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'val')
        inputs_val = next(fetcher_val)

        # resume training if necessary
        self._load_checkpoint()

        # remember the initial value of ds weight
        self.initial_lambda_ds = args.lambda_ds

        self.aggregated_losses = {}

        print('Start training...')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            # fetch images and labels
            inputs = next(fetcher)

            losses = self._training_step(inputs, i)

            all_losses = dict()
            for key, value in losses.items():
                all_losses[key] = value
            all_losses['G/lambda_ds'] = args.lambda_ds

            for key, value in losses.items():
                if key not in self.aggregated_losses.keys():
                    self.aggregated_losses[key] = 0
                self.aggregated_losses[key] += all_losses[key]


            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)

            if (i + 1) % args.log_every == 0:
                for key in self.aggregated_losses.keys():
                    self.aggregated_losses[key] /= args.log_every
                if self.logger is not None:
                    self.logger.log_metrics(self.aggregated_losses, i+1)
                for key in self.aggregated_losses.keys():
                    self.aggregated_losses[key] = 0

            # generate images for debugging
            if (i+1) % args.sample_every == 0:
                os.makedirs(args.sample_dir, exist_ok=True)
                image_fname_list = self._generate_images(inputs_val, i+1)
                if self.logger is not None:
                    if isinstance(self.logger, WandbLogger):
                        image_dict = {}
                        for name, path in image_fname_list.items():
                            image_dict[name] = Image(path)
                        self.logger.log_metrics(image_dict, i+1)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)

            # compute FID and LPIPS if necessary
            if (i+1) % args.eval_every == 0:
                val_metrics = self._evaluate(i+1)
                self.logger.log_metrics(val_metrics, i + 1)


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


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg