##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret
## Modified from: https://github.com/AlexHex7/Non-local_pytorch
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import math
import os
import sys
import pdb
import numpy as np
from torch import nn
from torch.nn import functional as F
import functools

torch_ver = torch.__version__[:3]

if torch_ver == '0.4':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, '../inplace_abn'))
    from bn import InPlaceABNSync
    BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
    
elif torch_ver == '0.3':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, '../inplace_abn_03'))
    from modules import InPlaceABNSync
    BatchNorm2d = functools.partial(InPlaceABNSync, activation='none') 

class SelfAttentionBlock2D(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(SelfAttentionBlock2D, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            InPlaceABNSync(self.key_channels),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            InPlaceABNSync(self.key_channels),
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
            kernel_size=1, stride=1, padding=0, bias=False),
            InPlaceABNSync(self.out_channels)
        )


    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, h, w)
        context = self.W(context)
        if self.scale > 1:
            if torch_ver == '0.4':
                context = F.upsample(input=context, size=(h, w), mode='bilinear', align_corners=True)
            elif torch_ver == '0.3':
                context = F.upsample(input=context, size=(h, w), mode='bilinear')
        return context


class ISA_Block(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, out_channels, down_factor=[8,8]):
        super(ISA_Block, self).__init__()
        self.out_channels = out_channels
        assert isinstance(down_factor, (tuple, list)) and len(down_factor) == 2
        self.down_factor = down_factor
        self.long_range_sa = SelfAttentionBlock2D(in_channels, key_channels, value_channels, out_channels)
        self.short_range_sa = SelfAttentionBlock2D(out_channels, key_channels, value_channels, out_channels)
    
    def forward(self, x):
        n, c, h, w = x.size()
        dh, dw = self.down_factor       # down_factor for h and w, respectively
        
        out_h, out_w = math.ceil(h / dh), math.ceil(w / dw)
        # pad the feature if the size is not divisible
        pad_h, pad_w = out_h * dh - h, out_w * dw - w
        if pad_h > 0 or pad_w > 0:  # padding in both left&right sides
            feats = F.pad(x, (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2))
        else:
            feats = x
        
        # long range attention
        feats = feats.view(n, c, out_h, dh, out_w, dw)
        feats = feats.permute(0, 3, 5, 1, 2, 4).contiguous().view(-1, c, out_h, out_w)
        feats = self.long_range_sa(feats)
        c = self.out_channels

        # short range attention
        feats = feats.view(n, dh, dw, c, out_h, out_w)
        feats = feats.permute(0, 4, 5, 3, 1, 2).contiguous().view(-1, c, dh, dw)
        feats = self.short_range_sa(feats)
        feats = feats.view(n, out_h, out_w, c, dh, dw).permute(0, 3, 1, 4, 2, 5)
        feats = feats.contiguous().view(n, c, dh * out_h, dw * out_w)

        # remove padding
        if pad_h > 0 or pad_w > 0:
            feats = feats[:, :, pad_h//2:pad_h//2 + h, pad_w//2:pad_w//2 + w]
        
        return feats


class ISA_Module(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, out_channels, down_factors=[[8,8]], dropout=0):
        super(ISA_Module, self).__init__()

        assert isinstance(down_factors, (tuple, list))
        self.down_factors = down_factors

        self.stages = nn.ModuleList([
            ISA_Block(in_channels, key_channels, value_channels, out_channels, d) for d in down_factors
        ])

        concat_channels = in_channels + out_channels
        if len(self.down_factors) > 1:
            self.up_conv = nn.Sequential(
                nn.Conv2d(in_channels, len(self.down_factors) * out_channels, kernel_size=1, padding=0, bias=False),
                InPlaceABNSync(len(self.down_factors) * out_channels),
            )
            concat_channels = out_channels * len(self.down_factors) * 2
        
        self.conv_bn = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=1, bias=False),
            InPlaceABNSync(out_channels),
            nn.Dropout2d(dropout),
        )
    
    def forward(self, x):
        priors = [stage(x) for stage in self.stages]
        if len(self.down_factors) == 1:
            context = priors[0]
        else:
            context = torch.cat(priors, dim=1)
            x = self.up_conv(x)
        # residual connection
        return self.conv_bn(torch.cat([x, context], dim=1))