import pdb
import torch.nn as nn
import math
import os
import sys
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from loss import OhemCrossEntropy2d, CrossEntropy2d
import scipy.ndimage as nd

torch_ver = torch.__version__[:3]

class CriterionCrossEntropy(nn.Module):
    def __init__(self, ignore_index=26):
        super(CriterionCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        weight = torch.FloatTensor([0.846,0.907, 0.987, 0.986, 1.025, 1.009, 0.988, 1.235, 0.995, 0.925, 0.965, 0.976, 1.079,0.983, 0.943, 1.021, 1.133, 0.965, 1.156, 1.334, 0.99,  0.924, 0.896, 1.009, 0.858,0.867])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear')
        loss = self.criterion(scale_pred, target)
        return loss
    

class CriterionDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=26, use_weight=True, dsn_weight=0.4):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        weight = torch.FloatTensor([0.846,0.907, 0.987, 0.986, 1.025, 1.009, 0.988, 1.235, 0.995, 0.925, 0.965, 0.976, 1.079,0.983, 0.943, 1.021, 1.133, 0.965, 1.156, 1.334, 0.99,  0.924, 0.896, 1.009, 0.858,0.867])
        if use_weight:
            print("w/ class balance")
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            print("w/o class balance")
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear')
        loss1 = self.criterion(scale_pred, target)

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear')
        loss2 = self.criterion(scale_pred, target)
        return self.dsn_weight*loss1 + loss2


class CriterionOhemDSN(nn.Module):
    '''
    DSN + OHEM : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=26, thres=0.7, min_kept=100000, dsn_weight=0.4, use_weight=True):
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        self.criterion = OhemCrossEntropy2d(ignore_index, thres, min_kept, use_weight=use_weight)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear')
        loss1 = self.criterion(scale_pred, target)
        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear')
        loss2 = self.criterion(scale_pred, target)
        return self.dsn_weight*loss1 + loss2


class CriterionOhemDSN_single(nn.Module):
    '''
    DSN + OHEM : we find that use hard-mining for both supervision harms the performance.
                Thus we choose the original loss for the shallow supervision
                and the hard-mining loss for the deeper supervision
    '''
    def __init__(self, ignore_index=26, thres=0.7, min_kept=100000, dsn_weight=0.4):
        super(CriterionOhemDSN_single, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        weight = torch.FloatTensor([0.846,0.907, 0.987, 0.986, 1.025, 1.009, 0.988, 1.235, 0.995, 0.925, 0.965, 0.976, 1.079,0.983, 0.943, 1.021, 1.133, 0.965, 1.156, 1.334, 0.99,  0.924, 0.896, 1.009, 0.858,0.867])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.criterion_ohem = OhemCrossEntropy2d(ignore_index, thres, min_kept, use_weight=True)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear')
        loss1 = self.criterion(scale_pred, target)

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear')
        loss2 = self.criterion_ohem(scale_pred, target)
        return self.dsn_weight*loss1 + loss2
