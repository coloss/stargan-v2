"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from pathlib import Path
from itertools import chain
import os
import random

from munch import Munch
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
import glob
import albumentations as A
import albumentations.pytorch as Ap


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None, recursive=False):
        if not recursive:
            self.samples = listdir(root)
        else:
            self.samples = glob.glob(os.path.join(root, "**", "*.*"), recursive=recursive)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = np.array(Image.open(fname).convert('RGB'))
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img

    def __len__(self):
        return len(self.samples)


class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform

    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, fnames2, labels = [], [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]
        img = np.array(Image.open(fname).convert('RGB'))
        img2 = np.array(Image.open(fname2).convert('RGB'))
        if self.transform is not None:
            img = self.transform(image=img)['image']
            img2 = self.transform(image=img2)['image']
        return img, img2, label

    def __len__(self):
        return len(self.targets)


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


class MultiFolderImageDataset(data.Dataset):

    def __init__(self, root, subfolders, transform=None, recursive=True, reference=False):
        super().__init__()
        self.transform = transform
        self.samples = []
        self.reference = reference
        if reference:
            self.samples2 = []
        else:
            self.samples2 = None
        self.labels = []
        self.targets = []
        from torchvision.datasets import folder as df

        for folder in subfolders:
            self.labels += [folder]
            samples_in_folder = []
            for ext in df.IMG_EXTENSIONS:
                if recursive:
                    # samples_in_folder += sorted(list(Path(root).rglob(f"**/{folder}/**/*" + ext)))
                    samples_in_folder += sorted(list(glob.glob(str(Path(root) / (f"**/{folder}/**/*" + ext)), recursive=True)))
                else:
                    # samples_in_folder += sorted(list(Path(root).glob(f"{folder}/*" + ext)))
                    samples_in_folder += sorted(list(glob.glob(str(Path(root) / (f"{folder}/*" + ext)), recursive=False)))
            self.samples += samples_in_folder
            if self.samples2 is not None:
                self.samples2 += random.sample(samples_in_folder, len(samples_in_folder))
            self.targets += [len(self.labels) - 1] * len(samples_in_folder)
        self.loader = df.default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        target = self.targets[index]
        sample = np.array(self.loader(path))
        if self.transform is not None:
            sample = self.transform(image=sample)['image']

        if not self.reference:
            return sample, target

        path2 = self.samples[index]
        sample2 = np.array(self.loader(path2))
        if self.transform is not None:
            sample2 = self.transform(image=sample2)['image']
        return sample, sample2, target

class ImageFolder2(ImageFolder):

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(image=sample)['image']
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class MultiFolderCorrespondenceImageDataset(data.Dataset):

    def __init__(self, root, subfolders, transform=None, recursive=True, reference=False, domains_to_split=None):
        super().__init__()
        self.transform = transform
        self.samples = {}
        self.labels = {}
        self.sample_labels = {}
        self.targets = []
        self.domains_to_split = domains_to_split
        self.subfolders = subfolders

        from torchvision.datasets import folder as df

        for i, folder in enumerate(subfolders):
            self.labels[folder] = i
            samples_in_folder = []
            for ext in df.IMG_EXTENSIONS:
                if recursive:
                    # samples_in_folder += sorted(list(Path(root).rglob(f"**/{folder}/**/*" + ext)))
                    samples_in_folder += sorted(list(glob.glob(str(Path(root) / (f"**/{folder}/**/*" + ext)), recursive=True)))
                else:
                    # samples_in_folder += sorted(list(Path(root).glob(f"{folder}/*" + ext)))
                    samples_in_folder += sorted(list(glob.glob(str(Path(root) / (f"{folder}/*" + ext)), recursive=False)))
            self.samples[folder] = samples_in_folder
            self.sample_labels[folder] = [i] * len(samples_in_folder)

        N = len(self.samples[folder])

        filenames = [Path(p).stem for p in samples_in_folder]
        for domain in self.samples.keys():
            if len(self.samples[domain]) != N:
                raise RuntimeError("All domains need to be in perfect correspondence")
            filenames2 = [Path(p).stem for p in self.samples[domain]]
            if filenames != filenames2:
                raise RuntimeError("All domains need to be in perfect correspondence")
        self.N = N
        self.loader = df.default_loader

        new_labels = []
        low_index = 999999
        label_idx = 0
        for i, di in enumerate(domains_to_split):
            domain_name = subfolders[di]
            domain_samples = self.samples[domain_name]
            subdomains = sorted(list(set([Path(self.samples[domain_name][i]).parents[0].name for i in range(N)])))
            # num_subdomains =
            print(f"Spliting domain '{domain_name}' into subdomains: {' '.join(list(subdomains))}")

            for j in range(len(subdomains)):
                new_labels += [low_index]
                for k in range(len(self.samples[domain_name])):
                    fname = self.samples[domain_name][k]
                    if subdomains[j] == Path(fname).parent.name:
                        self.sample_labels[domain_name][k] = low_index
                low_index -= 1

        old_labels = [i for i in range(len(subfolders)) if i not in domains_to_split]

        new_labels = sorted(old_labels + new_labels)
        final_labels = list(range(len(new_labels)))
        self.new2final = dict(zip(new_labels, final_labels))

        for domain_name, label_list in self.sample_labels.items():
            for li, label in enumerate(label_list):
                self.sample_labels[domain_name][li] = self.new2final[label]

    def num_initial_domains(self):
        return len(self.subfolders)

    def num_final_domains(self):
        return len(self.new2final)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample_d = {}
        label = []
        for domain in self.samples.keys():
            im = np.array(self.loader(self.samples[domain][index]))
            # if self.transform is not None:
            #     im = self.transform(im)
            sample_d[domain] = im
            # label += [self.labels[domain]]
            label += [self.sample_labels[domain][index]]
        if self.transform is not None:
            sample_d = self.transform(image=im, **sample_d)
        sample = []
        for domain in self.samples.keys():
            sample += [sample_d[domain]]
        return sample, label


def get_train_loader(root, which='source', img_size=256,
                     batch_size=8, prob=0.5, num_workers=4,
                     domain_names=None,
                     domains_to_split=None):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    # crop = transforms.RandomResizedCrop(
    #     img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    # rand_crop = transforms.Lambda(
    #     lambda x: crop(x) if random.random() < prob else x)
    #
    # transform = transforms.Compose([
    #     rand_crop,
    #     transforms.Resize([img_size, img_size]),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                          std=[0.5, 0.5, 0.5]),
    # ])

    rand_crop = A.RandomResizedCrop(
        img_size, img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=prob)

    additional_targets = {name : "image" for name in domain_names} if domain_names is not None else {}

    transform = A.Compose([
        rand_crop,
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
        A.pytorch.transforms.ToTensorV2(),
        # A.Normalize(mean=[0.5, 0.5, 0.5],
        #                      std=[0.5, 0.5, 0.5])
    ],
        additional_targets=additional_targets
    )

    if which == 'source':
        if domain_names is None or len(domain_names) == 0:
            dataset = ImageFolder2(root, transform)
        else:
            dataset = MultiFolderImageDataset(root, domain_names, transform, recursive=True)
    elif which == 'reference':
        if domain_names is None or len(domain_names) == 0:
            dataset = ReferenceDataset(root, transform)
        else:
            dataset = MultiFolderImageDataset(root, domain_names, transform, recursive=True, reference=True)
    elif which == 'correspondence':
        dataset = MultiFolderCorrespondenceImageDataset(root, domain_names, transform, recursive=True,
                                             domains_to_split=domains_to_split)
    else:
        raise NotImplementedError
    print(f"Dataset has {len(dataset)} samples")
    if which == 'correspondence': # or not hasattr(dataset, 'targets'):
        sampler = None
    else:
        sampler = _make_balanced_sampler(dataset.targets)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           sampler=sampler,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)


def get_eval_loader(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False, recursive=False, domain_names=None):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    # transform = transforms.Compose([
    #     transforms.Resize([img_size, img_size]),
    #     transforms.Resize([height, width]),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=mean, std=std)
    # ])

    additional_targets = {name : "image" for name in domain_names} if domain_names is not None else {}
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Resize(height, width),
        A.Normalize(mean=mean, std=std),
        A.pytorch.transforms.ToTensorV2(),
        # A.Normalize(mean=mean, std=std),
    ],
        additional_targets=additional_targets)

    dataset = DefaultDataset(root, transform=transform, recursive=recursive)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


def get_test_loader(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4,
                    domain_names=None, which=None,
                    domains_to_split=None):
    print('Preparing DataLoader for the generation phase...')
    # transform = transforms.Compose([
    #     transforms.Resize([img_size, img_size]),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                          std=[0.5, 0.5, 0.5]),
    # ])
    additional_targets = {name : "image" for name in domain_names} if domain_names is not None else {}
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        A.pytorch.transforms.ToTensorV2(),
        # A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ], additional_targets=additional_targets)

    if which == 'correspondence':
        dataset = MultiFolderCorrespondenceImageDataset(root, domain_names, transform, recursive=True,
                                                        domains_to_split=domains_to_split)
    else:
        if domain_names is None or len(domain_names) == 0:
            dataset = ImageFolder2(root, transform)
        else:
            dataset = MultiFolderImageDataset(root, domain_names, transform, recursive=True)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        x, y = self._fetch_inputs()
        if self.mode == 'train':
            inputs = Munch(x_src=x, y_src=y)
            if self.loader_ref is not None:
                x_ref, x_ref2, y_ref = self._fetch_refs()
                inputs['x_ref'] = x_ref
                inputs['x_ref2'] = x_ref2
                inputs['y_ref'] = y_ref
            if self.latent_dim > 0:
                z_trg = torch.randn(x.size(0), self.latent_dim)
                z_trg2 = torch.randn(x.size(0), self.latent_dim)
                inputs['z_trg'] = z_trg
                inputs['z_trg2'] = z_trg2

        elif self.mode == 'val':
            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y,
                           x_ref=x_ref, y_ref=y_ref)
        elif self.mode == 'test':
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError

        return Munch({k: to(v, self.device)
                      for k, v in inputs.items()})


def to(what, device):
    if isinstance(what, list):
        for i in range(len(what)):
            what[i] = to(what[i], device)
    elif isinstance(what, torch.Tensor):
        what = what.to(device)
    else:
        raise ValueError(f"Unsupported type: {type(what)}'")
    return what