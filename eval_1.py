##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: speedinghzl02
## updated by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import matplotlib
matplotlib.use('Agg')
import argparse
import scipy
from scipy import ndimage
import torch, cv2
import numpy as np
import numpy.ma as ma
import sys
import pdb
import torch
import PIL.Image
import PIL
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data
from dataset import get_segmentation_dataset
from network import get_segmentation_model
from config import Parameters
from collections import OrderedDict
import os
import scipy.ndimage as nd
from math import ceil
from PIL import Image as PILImage

import mlflow
import matplotlib.pyplot as plt
import torch.nn as nn

from collections import namedtuple
from utils import viewer2 as viewer
torch_ver = torch.__version__[:3]

# TODO : change to idd
def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    palette = [0] * (num_cls * 3)
    palette[0:3] = (128, 64, 128)       # 0: 'road' 
    palette[3:6] = (244, 35,232)        # 1 'sidewalk'
    palette[6:9] = (70, 70, 70)         # 2''building'
    palette[9:12] = (102,102,156)       # 3 wall
    palette[12:15] =  (190,153,153)     # 4 fence
    palette[15:18] = (153,153,153)      # 5 pole
    palette[18:21] = (250,170, 30)      # 6 'traffic light'
    palette[21:24] = (220,220, 0)       # 7 'traffic sign'
    palette[24:27] = (107,142, 35)      # 8 'vegetation'
    palette[27:30] = (152,251,152)      # 9 'terrain'
    palette[30:33] = ( 70,130,180)      # 10 sky
    palette[33:36] = (220, 20, 60)      # 11 person
    palette[36:39] = (255, 0, 0)        # 12 rider
    palette[39:42] = (0, 0, 142)        # 13 car
    palette[42:45] = (0, 0, 70)         # 14 truck
    palette[45:48] = (0, 60,100)        # 15 bus
    palette[48:51] = (0, 80,100)        # 16 train
    palette[51:54] = (0, 0,230)         # 17 'motorcycle'
    palette[54:57] = (119, 11, 32)      # 18 'bicycle'
    palette[57:60] = (105, 105, 105)
    return palette


def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img


def predict_sliding(net, image, tile_size, classes, method, scale=1):
    if scale != 1:
        scaled_img = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)
    else:
        scaled_img = image

    N_, C_, H_, W_ = scaled_img.shape

    # if torch_ver == '0.4':
    #     interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    # else:
    #     interp = nn.Upsample(size=tile_size, mode='bilinear')

    full_probs = np.zeros((N_, H_, W_, classes))
    count_predictions = np.zeros((N_, H_, W_, classes))
    overlap = 0
    stride_h = ceil(tile_size[0] * (1 - overlap))
    stride_w = ceil(tile_size[1] * (1 - overlap))
    tile_rows = int(ceil((H_ - tile_size[0]) / stride_h) + 1)  # strided convolution formula
    tile_cols = int(ceil((W_ - tile_size[1]) / stride_w) + 1)
    print("Need %i x %i prediction tiles @ stride %i px, %i py" % (tile_cols, tile_rows, stride_h, stride_w))

    tile_counter = 0

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride_w)
            y1 = int(row * stride_h)
            x2 = min(x1 + tile_size[1], W_)
            y2 = min(y1 + tile_size[0], H_)
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

            img = scaled_img[:, :, y1:y2, x1:x2]
            padded_img = pad_image(img, tile_size)
            tile_counter += 1
            print("Predicting tile %i" % tile_counter)
            padded_prediction_ = net(Variable(torch.from_numpy(padded_img), volatile=True).cuda(), )
    
            if 'dsn' in method or 'center' in method:
                padded_prediction = padded_prediction_[-1]
            else:
                padded_prediction = padded_prediction_
            # pdb.set_trace()
            # padded_prediction = nn.functional.softmax(padded_prediction, dim=1)
            padded_prediction = F.upsample(input=padded_prediction, size=tile_size, mode='bilinear', align_corners=True)
            padded_prediction = padded_prediction.cpu().data.numpy().transpose(0,2,3,1)
            prediction = padded_prediction[:, 0:img.shape[2], 0:img.shape[3], :]
            count_predictions[:, y1:y2, x1:x2] += 1
            full_probs[:, y1:y2, x1:x2] += prediction 

    full_probs /= count_predictions
    full_probs = ndimage.zoom(full_probs, (1., 1./scale, 1./scale, 1.),
        order=1, prefilter=False)
    return full_probs


def predict_whole_img(net, image, classes, method, scale):
    """
         Predict the whole image w/o using multiple crops.
         The scale specify whether rescale the input image before predicting the results.
    """
    N_, C_, H_, W_ = image.shape
    if torch_ver == '0.4':
        interp = nn.Upsample(size=(H_, W_), mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=(H_, W_), mode='bilinear')
    if scale != 1:
        scaled_img = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)
    else:
        scaled_img = image
    
    full_prediction_ = net(Variable(torch.from_numpy(scaled_img), volatile=True).cuda(), )
    if 'dsn' in method or 'center' in method or 'fuse' in method:
        full_prediction = full_prediction_[-1]
    else:
        full_prediction = full_prediction_

    if torch_ver == '0.4':
        full_prediction = F.upsample(input=full_prediction, size=(H_, W_), mode='bilinear', align_corners=True)
    else:
        full_prediction = F.upsample(input=full_prediction, size=(H_, W_), mode='bilinear')
    result = full_prediction.cpu().data.numpy().transpose(0,2,3,1)
    return result


def predict_whole_img_w_label(net, image, classes, method, scale, label):
    """
         Predict the whole image w/o using multiple crops.
         The scale specify whether rescale the input image before predicting the results.
    """
    N_, C_, H_, W_ = image.shape
    if torch_ver == '0.4':
        interp = nn.Upsample(size=(H_, W_), mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=(H_, W_), mode='bilinear')

#     bug
#     if scale > 1:
#         scaled_img = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)
#     else:
#         scaled_img = image

    scaled_img = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)
    
    full_prediction_ = net(Variable(torch.from_numpy(scaled_img), volatile=True).cuda(), label)
    if 'dsn' in method or 'center' in method or 'fuse' in method:
        full_prediction = full_prediction_[-1]
    else:
        full_prediction = full_prediction_

    full_prediction = F.upsample(input=full_prediction, size=(H_, W_), mode='bilinear', align_corners=True)
    result = full_prediction.cpu().data.numpy().transpose(0,2,3,1)
    return result


def predict_multi_scale(net, image, scales, tile_size, classes, flip_evaluation, method):
    """
    Predict an image by looking at it with different scales.
        We choose the "predict_whole_img" for the image with less than the original input size,
        for the input of larger size, we would choose the cropping method to ensure that GPU memory is enough.
    """
    
    N_, C_, H_, W_ = image.shape
    full_probs = np.zeros((N_, H_, W_, classes))  
    for scale in scales:
        scale = float(scale)
        print("Predicting image scaled by %f" % scale)
        sys.stdout.flush()
        if scale <= 1.0:
            scaled_probs = predict_whole_img(net, image, classes, method, scale=scale)
        else:        
            scaled_probs = predict_sliding(net, image, (720,1280), classes, method, scale=scale)
        if flip_evaluation == 'True':
            if scale <= 1.0:
                flip_scaled_probs = predict_whole_img(net, image[:,:,:,::-1].copy(), classes, method, scale=scale)
            else:
                flip_scaled_probs = predict_sliding(net, image[:,:,:,::-1].copy(), (720,1280), classes, method, scale=scale)
            scaled_probs = 0.5 * (scaled_probs + flip_scaled_probs[:,:,::-1])
        full_probs += scaled_probs
    full_probs /= len(scales)
    return full_probs


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


def id2trainId(label, id_to_trainid, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy


def main():
    """Create the model and start the evaluation process."""
    args = Parameters().parse()

    # mlflow to log
    exp_id = mlflow.set_experiment(args.experiment_name)
    mlflow.start_run(experiment_id=exp_id)
    mlflow.log_param("train_configs", vars(args))

    print("Input arguments:")
    sys.stdout.flush()
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    deeplab = get_segmentation_model("_".join([args.network, args.method]), num_classes=args.num_classes)

    ignore_label = args.ignore_label
    id_to_trainid = {-1: ignore_label, 255: ignore_label}
    for i in range(args.num_classes):
        id_to_trainid[i] = i


    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    
    saved_state_dict = torch.load(args.restore_from)
    deeplab.load_state_dict(saved_state_dict)

    model = nn.DataParallel(deeplab)
    model.eval()
    model.cuda()


    testloader = data.DataLoader(get_segmentation_dataset(args.dataset, root=args.data_dir, list_path=args.data_list, 
                                    crop_size=input_size, scale=False, mirror=False, network=args.network),
                                    batch_size=args.batch_size, shuffle=False, pin_memory=True)

    data_list = []
    confusion_matrix = np.zeros((args.num_classes,args.num_classes))
    # confusion_matrix = np.eye((args.num_classes))
    
    # level 2
    confusion_matrix_2 = np.zeros((16, 16))
    # confusion_matrix_2 = np.eye((16))
    
    # level 1
    confusion_matrix_1 = np.zeros((7, 7))
    # confusion_matrix_1 = np.eye((7))
    
    
    Label = namedtuple( 'Label' , [

    'name'        , 
    'id'          ,

    'csId'        ,

    'csTrainId'   ,    

    'level4Id'    , 
    'level3Id'    , 
    'level2IdName', 
    'level2Id'    , 
    'level1Id'    , 

    'hasInstances', 
    'ignoreInEval', 
    'color'       , 
    ] )
    labels = [
    #       name                     id    csId     csTrainId level4id        level3Id  category           level2Id      level1Id  hasInstances   ignoreInEval   color
    Label(  'road'                 ,  0   ,  7 ,     0 ,       0   ,     0  ,   'drivable'            , 0           , 0      , False        , False        , (128, 64,128)  ),
    Label(  'parking'              ,  1   ,  9 ,   255 ,       1   ,     1  ,   'drivable'            , 1           , 0      , False        , False         , (250,170,160)  ),
    Label(  'drivable fallback'    ,  2   ,  255 ,   255 ,     2   ,       1  ,   'drivable'            , 1           , 0      , False        , False         , ( 81,  0, 81)  ),
    Label(  'sidewalk'             ,  3   ,  8 ,     1 ,       3   ,     2  ,   'non-drivable'        , 2           , 1      , False        , False        , (244, 35,232)  ),
    Label(  'rail track'           ,  4   , 10 ,   255 ,       3   ,     3  ,   'non-drivable'        , 3           , 1      , False        , False         , (230,150,140)  ),
    Label(  'non-drivable fallback',  5   , 255 ,     9 ,      4   ,      3  ,   'non-drivable'        , 3           , 1      , False        , False        , (152,251,152)  ),
    Label(  'person'               ,  6   , 24 ,    11 ,       5   ,     4  ,   'living-thing'        , 4           , 2      , True         , False        , (220, 20, 60)  ),
    Label(  'animal'               ,  7   , 255 ,   255 ,      6   ,      4  ,   'living-thing'        , 4           , 2      , True         , True        , (246, 198, 145)),
    Label(  'rider'                ,  8   , 25 ,    12 ,       7   ,     5  ,   'living-thing'        , 5           , 2      , True         , False        , (255,  0,  0)  ),
    Label(  'motorcycle'           ,  9   , 32 ,    17 ,       8   ,     6  ,   '2-wheeler'           , 6           , 3      , True         , False        , (  0,  0,250)  ),
    Label(  'bicycle'              , 10   , 33 ,    18 ,       9   ,     7  ,   '2-wheeler'           , 6           , 3      , True         , False        , (119, 11, 32)  ),
    Label(  'autorickshaw'         , 11   , 255 ,   255 ,     10   ,      8  ,   'autorickshaw'        , 7           , 3      , True         , False        , (255, 204, 54) ),
    Label(  'car'                  , 12   , 26 ,    13 ,      11   ,     9  ,   'car'                 , 7           , 3      , True         , False        , (  0,  0,120)  ),
    Label(  'truck'                , 13   , 27 ,    14 ,      12   ,     10 ,   'large-vehicle'       , 8           , 3      , True         , False        , (  0,  0, 70)  ),
    Label(  'bus'                  , 14   , 28 ,    15 ,      13   ,     11 ,   'large-vehicle'       , 8           , 3      , True         , False        , (  0, 60,100)  ),
    Label(  'caravan'              , 15   , 29 ,   255 ,      14   ,     12 ,   'large-vehicle'       , 8           , 3      , True         , True         , (  0,  0, 90)  ),
    Label(  'trailer'              , 16   , 30 ,   255 ,      15   ,     12 ,   'large-vehicle'       , 8           , 3      , True         , True         , (  0,  0,110)  ),
    Label(  'train'                , 17   , 31 ,    16 ,      15   ,     12 ,   'large-vehicle'       , 8           , 3      , True         , True        , (  0, 80,100)  ),
    Label(  'vehicle fallback'     , 18   , 355 ,   255 ,     15   ,      12 ,   'large-vehicle'       , 8           , 3      , True         , False        , (136, 143, 153)),  
    Label(  'curb'                 , 19   ,255 ,   255 ,      16   ,     13 ,   'barrier'             , 9           , 4      , False        , False        , (220, 190, 40)),
    Label(  'wall'                 , 20   , 12 ,     3 ,      17   ,     14 ,   'barrier'             , 9           , 4      , False        , False        , (102,102,156)  ),
    Label(  'fence'                , 21   , 13 ,     4 ,      18   ,     15 ,   'barrier'             , 10           , 4      , False        , False        , (190,153,153)  ),
    Label(  'guard rail'           , 22   , 14 ,   255 ,      19   ,     16 ,   'barrier'             , 10          , 4      , False        , False         , (180,165,180)  ),
    Label(  'billboard'            , 23   , 255 ,   255 ,     20   ,      17 ,   'structures'          , 11           , 4      , False        , False        , (174, 64, 67) ),
    Label(  'traffic sign'         , 24   , 20 ,     7 ,      21   ,     18 ,   'structures'          , 11          , 4      , False        , False        , (220,220,  0)  ),
    Label(  'traffic light'        , 25   , 19 ,     6 ,      22   ,     19 ,   'structures'          , 11          , 4      , False        , False        , (250,170, 30)  ),
    Label(  'pole'                 , 26   , 17 ,     5 ,      23   ,     20 ,   'structures'          , 12          , 4      , False        , False        , (153,153,153)  ),
    Label(  'polegroup'            , 27   , 18 ,   255 ,      23   ,     20 ,   'structures'          , 12          , 4      , False        , False         , (153,153,153)  ),
    Label(  'obs-str-bar-fallback' , 28   , 255 ,   255 ,     24   ,      21 ,   'structures'          , 12          , 4      , False        , False        , (169, 187, 214) ),  
    Label(  'building'             , 29   , 11 ,     2 ,      25   ,     22 ,   'construction'        , 13          , 5      , False        , False        , ( 70, 70, 70)  ),
    Label(  'bridge'               , 30   , 15 ,   255 ,      26   ,     23 ,   'construction'        , 13          , 5      , False        , False         , (150,100,100)  ),
    Label(  'tunnel'               , 31   , 16 ,   255 ,      26   ,     23 ,   'construction'        , 13          , 5      , False        , False         , (150,120, 90)  ),
    Label(  'vegetation'           , 32   , 21 ,     8 ,      27   ,     24 ,   'vegetation'          , 14          , 5      , False        , False        , (107,142, 35)  ),
    Label(  'sky'                  , 33   , 23 ,    10 ,      28   ,     25 ,   'sky'                 , 15          , 6      , False        , False        , ( 70,130,180)  ),
    Label(  'fallback background'  , 34   , 255 ,   255 ,     29   ,      25 ,   'object fallback'     , 15          , 6      , False        , False        , (169, 187, 214)),
    Label(  'unlabeled'            , 35   ,  0  ,     255 ,   255   ,      255 ,   'void'                , 255         , 255    , False        , True         , (  0,  0,  0)  ),
    Label(  'ego vehicle'          , 36   ,  1  ,     255 ,   255   ,      255 ,   'void'                , 255         , 255    , False        , True         , (  0,  0,  0)  ),
    Label(  'rectification border' , 37   ,  2  ,     255 ,   255   ,      255 ,   'void'                , 255         , 255    , False        , True         , (  0,  0,  0)  ),
    Label(  'out of roi'           , 38   ,  3  ,     255 ,   255   ,      255 ,   'void'                , 255         , 255    , False        , True         , (  0,  0,  0)  ),
    Label(  'license plate'        , 39   , 255 ,     255 ,   255   ,      255 ,   'vehicle'             , 255         , 255    , False        , True         , (  0,  0,142)  ),
    
]   
    palette = get_palette(20)

    image_id = 0
    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d processd'%(index))
            sys.stdout.flush()
        image, label, size, name = batch
        size = size[0].numpy()
        if torch_ver == '0.3':
            if args.use_ms == 'True': 
                output = predict_multi_scale(model, image.numpy(), ([0.75, 1, 1.25]), input_size, 
                    args.num_classes, args.use_flip, args.method)
            else:
                if args.use_flip == 'True':
                    output = predict_multi_scale(model, image.numpy(), ([args.whole_scale]), input_size, 
                        args.num_classes, args.use_flip, args.method)
                else:
                    if 'gt' in args.method:
                        label = Variable(label.long().cuda())
                        output = predict_whole_img_w_label(model, image.numpy(), args.num_classes, 
                        args.method, scale=float(args.whole_scale), label=label)
                    else:
                        output = predict_whole_img(model, image.numpy(), args.num_classes, 
                            args.method, scale=float(args.whole_scale))
        else:
            with torch.no_grad():
                if args.use_ms == 'True': 
                    output = predict_multi_scale(model, image.numpy(), ([0.75, 1, 1.25]), input_size, 
                        args.num_classes, args.use_flip, args.method)
                else:
                    if args.use_flip == 'True':
                        output = predict_multi_scale(model, image.numpy(), ([args.whole_scale]), input_size, 
                            args.num_classes, args.use_flip, args.method)
                    else:
                        if 'gt' in args.method:
                            output = predict_whole_img_w_label(model, image.numpy(), args.num_classes, 
                            args.method, scale=float(args.whole_scale), label=Variable(label.long().cuda()))
                        else:
                            output = predict_whole_img(model, image.numpy(), args.num_classes, 
                                args.method, scale=float(args.whole_scale))
        # level 3 predictions
        seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
        m_seg_pred = ma.masked_array(seg_pred, mask=torch.eq(label, 26)) # dataset specific
        ma.set_fill_value(m_seg_pred, 26) # dataset specific
        seg_pred = m_seg_pred

        # level 2 predictions
        seg_pred_2 = np.array(seg_pred)
        for class_label in labels:
            label_id_src = class_label.level3Id
            label_id_dst = class_label.level2Id
            seg_pred_2[seg_pred_2==label_id_src] = label_id_dst
        
        m_seg_pred_2 = ma.masked_array(seg_pred_2, mask=torch.eq(label, 26)) # dataset specific
        ma.set_fill_value(m_seg_pred_2, 26) # dataset specific
        seg_pred_2 = m_seg_pred_2

        # level 1 predictions
        seg_pred_1 = np.array(seg_pred)
        for class_label in labels:
            label_id_src = class_label.level3Id
            label_id_dst = class_label.level1Id
            seg_pred_1[seg_pred_1==label_id_src] = label_id_dst
        
        m_seg_pred_1 = ma.masked_array(seg_pred_1, mask=torch.eq(label, 26)) # dataset specific
        ma.set_fill_value(m_seg_pred_1, 26) # dataset specific
        seg_pred_1 = m_seg_pred_1

        for i in range(image.size(0)): 
            image_id += 1
            print('%d th segmentation map generated ...'%(image_id))
            sys.stdout.flush()
            if args.store_output == 'True':
                output_im = viewer.view_image(seg_pred[i])
                dir_name, img_name = os.path.split(name[i])
                if not os.path.exists(output_path+dir_name):
                    os.makedirs(output_path+dir_name)
                output_im.save(output_path+dir_name+'/'+img_name)
                mlflow.log_artifact(output_path+dir_name+'/'+img_name)
        seg_gt = np.asarray(label.numpy()[:,:,:], dtype=np.int)
        ignore_index = seg_gt!=26
        seg_gt = seg_gt[ignore_index]
        seg_pred = seg_pred[ignore_index]
        seg_pred_2 = seg_pred_2[ignore_index]
        seg_pred_1 = seg_pred_1[ignore_index]
        
        # level 2 gt
        seg_gt_2 = np.array(seg_gt)
        for class_label in labels:
            label_id_src = class_label.level3Id
            label_id_dst = class_label.level2Id
            seg_gt_2[seg_gt_2==label_id_src] = label_id_dst
        
        # level 1 gt
        seg_gt_1 = np.array(seg_gt)
        for class_label in labels:
            label_id_src = class_label.level3Id
            label_id_dst = class_label.level1Id
            seg_gt_1[seg_gt_1==label_id_src] = label_id_dst

        confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, args.num_classes)
        confusion_matrix_2 += get_confusion_matrix(seg_gt_2, seg_pred_2, 16)
        confusion_matrix_1 += get_confusion_matrix(seg_gt_1, seg_pred_1, 7)
    # level 3
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    IU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IU = IU_array.mean()
    
    print({'meanIU':mean_IU, 'IU_array':IU_array})

    print("confusion matrix\n")
    print(confusion_matrix)
    mlflow.log_param("mean_iu", mean_IU)
    mlflow.log_param("iu_array", IU_array)
    mlflow.log_param("confusion_matrix", confusion_matrix)

    # level 2
    pos = confusion_matrix_2.sum(1)
    res = confusion_matrix_2.sum(0)
    tp = np.diag(confusion_matrix_2)

    IU_array_2 = (tp / np.maximum(1.0, pos + res - tp))
    mean_IU_2 = IU_array_2.mean()
    
    print({'meanIU':mean_IU_2, 'IU_array_2':IU_array_2})

    print("confusion matrix\n")
    print(confusion_matrix_2)
    mlflow.log_param("mean_iu_2", mean_IU_2)
    mlflow.log_param("iu_array_2", IU_array_2)
    mlflow.log_param("confusion_matrix_2", confusion_matrix_2)

    # level 1
    pos = confusion_matrix_1.sum(1)
    res = confusion_matrix_1.sum(0)
    tp = np.diag(confusion_matrix_1)

    IU_array_1 = (tp / np.maximum(1.0, pos + res - tp))
    mean_IU_1 = IU_array_1.mean()
    
    print({'meanIU':mean_IU_1, 'IU_array_1':IU_array_1})

    print("confusion matrix\n")
    print(confusion_matrix_1)
    print("#######")
    print("Summary")
    print("#######")
    print("Level 3 mIOU = ", mean_IU)
    print("Level 2 mIOU = ", mean_IU_2)
    print("Level 1 mIOU = ", mean_IU_1)
    
    mlflow.log_param("mean_iu_1", mean_IU_1)
    mlflow.log_param("iu_array_1", IU_array_1)
    mlflow.log_param("confusion_matrix_1", confusion_matrix_1)
    sys.stdout.flush()

if __name__ == '__main__':
    main()
    mlflow.end_run()
