import numpy as np
import torch.nn as nn
import torch
import os
import PIL.Image
from torch.utils import data
from dataset import get_segmentation_dataset
from torch.autograd import Variable
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
from utils import viewer3 as viewer
import os
import time
import numpy.ma as ma
import math

BatchNorm2d = nn.BatchNorm2d
affine_par = True

# Settings
method = "resnet101_interlaced_dsn"
restore_from = "mlruns/6/train_40_80/artifacts/CS_scenes_80000.pth"
list_path="dataset/list/idd/prune.lst"
batch_size = 8

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def _sum_each(self, x, y):
        assert(len(x)==len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i]+y[i])
        return z

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)
        return out

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
            BatchNorm2d(self.key_channels),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(self.key_channels),
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(self.key_channels),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(self.key_channels),
        )
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
            kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(self.out_channels)
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
                BatchNorm2d(len(self.down_factors) * out_channels),
            )
            concat_channels = out_channels * len(self.down_factors) * 2
        
        self.conv_bn = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=1, bias=False),
            BatchNorm2d(out_channels),
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
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1,1,1))
        
        # extra added layers
        self.context = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(512),
            ISA_Module(in_channels=512, key_channels=256, value_channels=512,
                       out_channels=512, down_factors=[[8,8]], dropout=0.05),
            )
        self.cls = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(512),
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
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
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
        x = self.layer4(x)
        x = self.context(x)
        x = self.cls(x)
        return x

def get_resnet101_interlaced(num_classes=21):
    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes)
    return model


def computeTime(model, size=(2048,128,128), device='cuda'):
    inputs = torch.randn(1, size[0], size[1], size[2])
    if device == 'cuda':
        model = model.cuda()
        inputs = inputs.cuda()

    model.eval()
    i = 0
    time_spent = []
    while i < 100:
        print(i)
        start_time = time.time()
        with torch.no_grad():
            output = model(inputs)
        if device == 'cuda':
            # wait for cuda to finish
            torch.cuda.synchronize()
        if i != 0:
            time_spent.append(time.time() - start_time)
        i += 1
        del output
        
    print('Average execution time (s): {:.3f}'.format(np.mean(time_spent)))

def get_confusion_matrix(gt_label, pred_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """
        index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

        return confusion_matrix

def get_result(model, method, image):
    N_, C_, H_, W_ = image.shape
    full_prediction = model(Variable(torch.from_numpy(image), volatile=True).cuda(), )
    full_prediction = F.upsample(input=full_prediction, size=(H_, W_), mode='bilinear', align_corners=True)
    result = full_prediction.cpu().data.numpy().transpose(0,2,3,1)
    return result

def eval_model(model, method="resnet101_interlaced_dsn", visualize=False, list_path="dataset/list/idd/train_dummy.lst"):
    # Load dataset
    testloader = data.DataLoader(get_segmentation_dataset("idd_train", root="dataset/idd", list_path=list_path, 
                                crop_size=(320,320), scale=False, mirror=False, network="resnet101"),
                                batch_size=batch_size, shuffle=False, pin_memory=True)
    print("Images loaded for evaluation")
    image_id = 0
    confusion_matrix = np.zeros((7, 7))
    for index, batch in enumerate(testloader):
        image, label, _ , name = batch
        image = np.array(image)

        result = get_result(model, method, image)
        seg_pred = np.asarray(np.argmax(result, axis=3), dtype=np.uint8)
        m_seg_pred = ma.masked_array(seg_pred, mask=torch.eq(label, 7))
        ma.set_fill_value(m_seg_pred, 7) 
        seg_pred = m_seg_pred

        if visualize:
            # visualize result
            output_im = viewer.view_image(seg_pred[0])
            output_path = "./visualize"
            dir_name, img_name = os.path.split(name[0])
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            output_im.save(output_path+'/'+img_name)
        
        # ignore labels
        seg_gt = np.asarray(label, dtype=np.int)
        ignore_index = seg_gt != 7 # dataset specific
        #print(seg_gt.shape, seg_pred.shape, ignore_index.shape)
        seg_gt = seg_gt[ignore_index]
        seg_pred = seg_pred[ignore_index]
        
        # cf matrix
        confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, 7)
        image_id+=batch_size
        print("Images done ", image_id)
    # compute iou and cf matrix
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IU = IU_array.mean()

    print({'meanIU':mean_IU, 'IU_array':IU_array})
    print("confusion matrix\n")
    print(confusion_matrix)

def load_model(method="resnet101_interlaced_dsn", restore_from="checkpoint/ft7/CS_scenes_80000.pth"):
    torch_model = get_resnet101_interlaced(num_classes=7)
    saved_state_dict = torch.load(restore_from)
    torch_model.load_state_dict(saved_state_dict)
    torch_model.cuda()
    torch_model.eval()
    print("Loaded torch model")
    return torch_model

# get model
model = load_model(method=method, restore_from=restore_from)
# evaluate miou of the model
eval_model(model, method=method, visualize=False, list_path=list_path)
# hardware benchamrk of the model
computeTime(model, size=(3,320,320))

