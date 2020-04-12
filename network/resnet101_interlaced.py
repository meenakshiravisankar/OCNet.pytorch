##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import os
import sys
import pdb
import numpy as np
from torch.autograd import Variable
import functools
affine_par = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from resnet_block import conv3x3, Bottleneck
sys.path.append(os.path.join(BASE_DIR, '../oc_module'))
from base_oc_block import BaseOC_Module
from pyramid_oc_block import Pyramid_OC_Module

torch_ver = torch.__version__[:3]

if torch_ver == '0.4':
    sys.path.append(os.path.join(BASE_DIR, '../inplace_abn'))
    from bn import InPlaceABNSync
    BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
    
elif torch_ver == '0.3':
    sys.path.append(os.path.join(BASE_DIR, '../inplace_abn_03'))
    from modules import InPlaceABNSync
    BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')    

class InterlacedSparseAttention(nn.Module):
    def __init__(self, P_h, P_w):
        super(InterlacedSparseAttention, self).__init__()
        self.P_h = P_h
        self.P_w = P_w
        self.attention = BaseOC_Module(in_channels=512, out_channels=512, key_channels=256, value_channels=256, 
                                       dropout=0.05, sizes=([1]))
        
    def forward(self, x):
        N, C, H, W = x.size()
        Q_h, Q_w = H // self.P_h, W // self.P_w
        pad_h, pad_w = self.P_h - (H - self.P_h * Q_h), self.P_w - (W - self.P_w * Q_w)
        pad_top, pad_bottom = pad_h//2, pad_h-pad_h//2
        pad_left, pad_right = pad_w//2, pad_w-pad_w//2
        pad = nn.ZeroPad2d((pad_left, pad_right, pad_top, pad_bottom))
        x = pad(x)
        if pad_left + pad_right != 0:
            Q_w += 1
        if pad_top + pad_bottom != 0:
            Q_h += 1
        N, C, H, W = x.size()
        x = x.reshape(N, C, Q_h, self.P_h, Q_w, self.P_w)
        # Long-range Attention
        x = x.permute(0,3,5,1,2,4)
        x = x.reshape(N * self.P_h * self.P_w, C, Q_h, Q_w)
        x = self.attention(x)
        x = x.reshape(N, self.P_h, self.P_w, C, Q_h, Q_w)

        # Short-range Attention
        x = x.permute(0,4,5,3,1,2)
        x = x.reshape(N * Q_h * Q_w, C, self.P_h, self.P_w)
        x = self.attention(x)
        x = x.reshape(N, Q_h, Q_w, C, self.P_h, self.P_w)
        x = x.permute(0,3,1,4,2,5)
        return x.reshape(N,C,H,W)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1,1,1)) # we do not apply multi-grid method here

        # extra added layers
        self.context = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            InterlacedSparseAttention(P_h=8, P_w=8)
            )
        self.cls = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion,affine = affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_dsn = self.dsn(x)
        x = self.layer4(x)
        x = self.context(x)
        x = self.cls(x)
        return [x_dsn, x]


def get_resnet101_interlaced_dsn(num_classes=21):
    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes)
    return model
