"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import argparse

from munch import Munch
from torch.backends import cudnn
import torch

from core.data_loader import get_train_loader
from core.data_loader import get_test_loader
from core.solver_dualstar import SolverDualStar


def str2bool(v):
    return v.lower() in ('true')


def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]


def main(args):
    print(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    domains_to_split = []
    if args.split_x_domain:
        domains_to_split += [0]
    if args.split_y_domain:
        domains_to_split += [1]


    if args.mode == 'train':
        if not hasattr(args, 'domain_names') or len(args.domain_names) == 0:
            assert len(subdirs(args.train_img_dir)) == args.num_domains
            assert len(subdirs(args.val_img_dir)) == args.num_domains
        else:
            assert len(args.domain_names) == args.num_domains

        loaders = Munch(src=get_train_loader(root= args.train_img_dir,
                                             which='correspondence',
                                             domain_names=args.domain_names,
                                             domains_to_split=domains_to_split,
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             shuffle=True,
                                             num_workers=args.num_workers),
                        ref=None,
                        val=get_test_loader(root=args.val_img_dir,
                                            which='correspondence',
                                            domain_names=args.domain_names,
                                             domains_to_split=domains_to_split,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers))
        args.num_domains = loaders.src.dataset.num_final_domains()
        solver = SolverDualStar(args)
        solver.fit(loaders)
    elif args.mode == 'sample':
        assert len(subdirs(args.src_dir)) == args.num_domains
        assert len(subdirs(args.ref_dir)) == args.num_domains
        loaders = Munch(src=get_test_loader(root=args.src_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers),
                        ref=get_test_loader(root=args.ref_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers))
        args.num_domains = loaders.src.dataset.num_final_domains()
        solver = SolverDualStar(args)
        solver.sample(loaders)
    elif args.mode == 'eval':
        if args.split_x_domain or args.split_y_domain:
            raise NotImplementedError("Splitting domains and eval options had not been tested and "
                                      "likely break due to the number of domains expected")
        solver = SolverDualStar(args)
        solver.evaluate()
    elif args.mode == 'paired_images':
        loaders = Munch(val=get_test_loader(root=args.val_img_dir,
                                            which='correspondence',
                                            domain_names=args.domain_names,
                                            domains_to_split=domains_to_split,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers))
        args.num_domains = loaders.val.dataset.num_final_domains()

        solver = SolverDualStar(args)
        solver.test_paired_images(loaders)
    elif args.mode == 'paired_test':
        loaders = Munch(val=get_test_loader(root=args.val_img_dir,
                                            which='correspondence',
                                            domain_names=args.domain_names,
                                            domains_to_split=domains_to_split,
                                            img_size=args.img_size,
                                            # batch_size=args.val_batch_size,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=args.num_workers),
                                            )
        args.num_domains = loaders.val.dataset.num_final_domains()

        solver = SolverDualStar(args)
        solver.test_on_validation_set(loaders)

    elif args.mode == 'align':
        from core.wing import align_faces
        align_faces(args, args.inp_dir, args.out_dir)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--arch_type', type=str, default='star', choices=['star', 'starskip', 'starskipcat'],
                        help='Architecture type')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=2,
                        help='Number of domains')
    parser.add_argument('--domain_names', nargs="*", default=[],
                        help='Specify domain subfolder by name instead')
    parser.add_argument('--split_x_domain', action='store_true', default=False,
                        help='Split the fist domain into multiple ones')
    parser.add_argument('--split_y_domain', action='store_true', default=False,
                        help='Split the second domain into multiple ones')
    parser.add_argument('--latent_dim', type=int, default=0,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=0,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')
    parser.add_argument('--direction', type=str, default='bi',
                        help='Translation from one domain to another or '
                             'bidirectional. Accepted values: "bi", "x2y","y2x" ')

    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 regularization')
    # parser.add_argument('--lambda_cyc', type=float, default=1,
    #                     help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_sup_photo', type=float, default=0.0,
                        help='Weight for one was reconstruction loss')
    parser.add_argument('--lambda_cyc_real_style', type=float, default=0.5,
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_cyc_fake_style', type=float, default=0.5,
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_d_real', type=float, default=1.0,
                    help='Weight for discriminator real loss')
    parser.add_argument('--lambda_d_fake', type=float, default=1.0,
                    help='Weight for discriminator fake loss')
    parser.add_argument('--lambda_vgg', type=float, default=0.0,
                        help='Weight for vgg perceptual losses')
    parser.add_argument('--vgg_loss_layers', type=list, default=[4, 9],
                        help='VGG layers to use for perceptual loss.')
    parser.add_argument('--lambda_vgg_layers', type=list, default=[0.5, 1],
                        help='Weights for VGG layers specified specified with --vgg_loss_layers')
    parser.add_argument('--lambda_face_rec', type=float, default=0.0,
                        help='Weight for face recognition loss')
    parser.add_argument('--lambda_emo_rec', type=float, default=0.0,
                        help='Weight for emotion recognition loss')
    parser.add_argument('--metric_emo_rec', type=str, default='l1', choices=['l1', 'l2', 'cos'] ,
                        help='Weight for emotion recognition loss')
    parser.add_argument('--lambda_sty', type=float, default=1,
                        help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_ds', type=float, default=1,
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--ds_iter', type=int, default=100000,
                        help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--w_hpf', type=float, default=1,
                        help='weight for high-pass filtering')

    # training arguments
    parser.add_argument('--randcrop_prob', type=float, default=0.5,
                        help='Probabilty of using random-resized cropping')
    parser.add_argument('--total_iters', type=int, default=100000,
                        help='Number of total iterations')
    parser.add_argument('--resume_iter', type=str, default="0",
                        help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for D, E and G')
    parser.add_argument('--f_lr', type=float, default=1e-6,
                        help='Learning rate for F')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=10,
                        help='Number of generated images per domain during sampling')

    # misc
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'sample', 'eval', 'align', 'paired_images', 'paired_test'],
                        help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')
    parser.add_argument('--logger', type=str, default='wandb',
                        help='Seed for random number generator')

    # directory for training
    parser.add_argument('--train_img_dir', type=str, default='data/celeba_hq/train',
                        help='Directory containing training images')
    parser.add_argument('--val_img_dir', type=str, default='data/celeba_hq/val',
                        help='Directory containing validation images')
    parser.add_argument('--run_dir', type=str, default='todo',
                        help='Directory for the experiment checkpoint and results')
    parser.add_argument('--expr_dir', type=str, default='experiments',
                        help='Directory for the experiment checkpoint and results')
    # parser.add_argument('--sample_dir', type=str, default='expr/samples',
    #                     help='Directory for saving generated images')
    # parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints',
    #                     help='Directory for saving network checkpoints')
    # directory for calculating metrics
    # parser.add_argument('--eval_dir', type=str, default='expr/eval',
    #                     help='Directory for saving metrics, i.e., FID and LPIPS')

    # directory for testing
    parser.add_argument('--result_dir', type=str, default='expr/results',
                        help='Directory for saving generated images and videos')
    parser.add_argument('--src_dir', type=str, default='assets/representative/celeba_hq/src',
                        help='Directory containing input source images')
    parser.add_argument('--ref_dir', type=str, default='assets/representative/celeba_hq/ref',
                        help='Directory containing input reference images')
    parser.add_argument('--inp_dir', type=str, default='assets/representative/custom/female',
                        help='input directory when aligning faces')
    parser.add_argument('--out_dir', type=str, default='assets/representative/celeba_hq/src/female',
                        help='output directory when aligning faces')

    # face alignment
    parser.add_argument('--wing_path', type=str, default='expr/checkpoints/wing.ckpt')
    parser.add_argument('--lm_path', type=str, default='expr/checkpoints/celeba_lm_mean.npz')

    # step size
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=5000)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=25000)
    # parser.add_argument('--eval_every', type=int, default=50000)
    # parser.add_argument('--sample_every', type=int, default=20)
    # parser.add_argument('--save_every', type=int, default=60)
    # parser.add_argument('--eval_every', type=int, default=10)

    parser.add_argument('--config_file', type=str)
    args = parser.parse_args()

    if hasattr(args, 'config_file') and args.config_file is not None:
        import yaml
        print(f"A config file was specified. Reading settings from '{args.config_file}'")
        with open(args.config_file, 'r') as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)
        iter = args.resume_iter
        mode = args.mode
        d = vars(args)
        d.update(opt)
        # args = opt
        args.resume_iter = iter
        args.mode = mode

    if isinstance(args.resume_iter, str) and args.resume_iter.isdigit():
        args.resume_iter = int(args.resume_iter)
    if args.resume_iter != 0:
        print(f"Resuming training from step {args.resume_iter}")

    main(args)
