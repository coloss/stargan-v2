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

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import numpy as np
from torchvision import models
from scipy import linalg
from core.data_loader import get_eval_loader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x


class VGG19Loss(nn.Module):

    def __init__(self, layer_activation_indices_weights, diff=torch.nn.functional.l1_loss):
        super().__init__()
        self.vgg19 = VGG19(sorted(layer_activation_indices_weights.keys()))
        self.layer_activation_indices_weights = layer_activation_indices_weights
        self.diff = diff

    def forward(self, x, y):
        feat_x = self.vgg19(x)
        feat_y = self.vgg19(y)

        out = {}
        loss = 0
        for idx, weight in self.layer_activation_indices_weights.items():
            d = self.diff(feat_x[idx], feat_y[idx])
            out[idx] = d
            loss += feat_x[idx]*weight
        return loss, out


class VGG19(nn.Module):

    def __init__(self, layer_activation_indices):
        super().__init__()
        self.layer_activation_indices = layer_activation_indices
        self.blocks = _vgg('vgg19', 'E', batch_norm=False, pretrained=True, progress=True)
        self.conv_block_indices = []

        self.layers = []
        for bi, block in enumerate(self.blocks):
            for layer in block:
                self.layers += [layer]
                if isinstance(layer, nn.Conv2d):
                    self.conv_block_indices += [bi]

        if len(self.layer_activation_indices) != len(set(layer_activation_indices).intersection(set(self.conv_block_indices))):
            raise ValueError("The specified layer indices are not of a conv block")

        self.net = nn.Sequential(*self.layers)
        self.net.eval()
        self.net.requires_grad_(False)

    def forward(self, x):
        layer_outputs = {}
        for bi, block in enumerate(self.blocks):
            for layer in block:
                x = layer(x)
            if bi in self.layer_activation_indices:
                layer_outputs[bi] = x
        layer_outputs['final'] = x
        return layer_outputs


## COPY-PASTED and modified from torchvision
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [[nn.MaxPool2d(kernel_size=2, stride=2)]]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [[conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]]
            else:
                layers += [[conv2d, nn.ReLU(inplace=True)]]
            in_channels = v
    return layers


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = make_layers(cfgs[cfg], batch_norm=batch_norm)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}
