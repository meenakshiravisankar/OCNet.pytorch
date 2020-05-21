import tensorflow as tf
import numpy as np
import torch.nn as nn
import torch
import PIL.Image
from torch.utils import data
from dataset import get_segmentation_dataset
from utils import viewer2 as viewer
import os
import numpy.ma as ma

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

# Load dataset
testloader = data.DataLoader(get_segmentation_dataset("idd_train",                                      root="dataset/idd",                                                        list_path="dataset/list/idd/train_dummy.lst", 
                             crop_size=(320,320), scale=False, mirror=False, network="resnet101"),
                             batch_size=1, shuffle=False, pin_memory=True)

confusion_matrix = np.zeros((7, 7))
    
for index, batch in enumerate(testloader):
    image, label, _ , name = batch
    image = np.array(image)
    # reshape input shape from NCHW to NHWC
    image = np.transpose(image, (0,2,3,1))

    # load model
    interpreter = tf.lite.Interpreter(model_path="/home/meenu/Desktop/ddp/semantic-segmentation/code/android-deploy/model_320_320.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # assign tensors
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    seg_pred = np.asarray(np.argmax(output_data, axis=3), dtype=np.uint8)
    m_seg_pred = ma.masked_array(seg_pred, mask=torch.eq(label, 7))
    ma.set_fill_value(m_seg_pred, 7) 
    seg_pred = m_seg_pred

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
    print(seg_gt.shape, seg_pred.shape, ignore_index.shape)
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    # cf matrix
    confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, 7)
    print(index)
    break

# compute iou and cf matrix
pos = confusion_matrix.sum(1)
res = confusion_matrix.sum(0)
tp = np.diag(confusion_matrix)
IU_array = (tp / np.maximum(1.0, pos + res - tp))
mean_IU = IU_array.mean()

print({'meanIU':mean_IU, 'IU_array':IU_array})
print("confusion matrix\n")
print(confusion_matrix)
