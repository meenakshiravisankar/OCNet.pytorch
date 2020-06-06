import onnxruntime
import onnx
import torch.onnx
from onnx2keras import onnx_to_keras
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import math
import torch.utils.model_zoo as model_zoo
import sys
import pdb
from torch.autograd import Variable
from copy import deepcopy
import functools
import numpy as np
from pytorch2keras.converter import pytorch_to_keras

affine_par = True

BatchNorm2d = nn.BatchNorm2d

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
        self.layer5 = nn.Sequential(
                nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
                BatchNorm2d(512),
                nn.Dropout2d(0.05)
                )
        self.layer6 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

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
        h, w = x.shape[2], x.shape[3]
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x  

def get_resnet101_baseline(num_classes=21):
    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes)
    return model

## random.seed(42)

def check_onnx(x, output, path="temp_320_320.onnx"):
    # Verify the model
    ort_session = onnxruntime.InferenceSession(path)
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(output), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

random.seed(42)

statuses = ["onnx", "keras", "tflite", "all"]
status = "tflite"

if status == statuses[0] or status == statuses[-1]:
    restore_from = "prune_g25.pth"
    torch_model = get_resnet101_baseline(num_classes=7)
    saved_state_dict = torch.load(restore_from, map_location=torch.device('cpu'))
    torch_model.load_state_dict(saved_state_dict)
    torch_model.eval()

    batch_size = 1
    channels = 3
    input_height = 320
    input_width = 320

    # Input to the model
    x = torch.randn(batch_size, channels, input_height, input_width)
    output = torch_model(x)
    torch.onnx.export(torch_model, x, "./temp_320_320.onnx", opset_version=11, 
                      input_names=["input"], output_names=["output"])
    print("Onnx exported")
    check_onnx(x, output, "temp_320_320.onnx")

if status == statuses[1] or status==statuses[-1]:
    # Convert to keras
    onnx_model = onnx.load("./temp_320_320.onnx")
    keras_model = onnx_to_keras(onnx_model=onnx_model, input_names=["input"], change_ordering=True, verbose=False)
    new_model = tf.keras.Sequential()
    new_model.add(keras_model)
    new_model.add(tf.keras.layers.UpSampling2D(size=(8,8), interpolation="bilinear"))
    new_model.save("keras_320_320")
    print("Saved keras model!")

if status == statuses[2] or status==statuses[-1]:
    # Convert to tflite
    keras_model = tf.keras.models.load_model("keras_320_320")
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.experimental_new_converter=False
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY] # Performs Quantization
    tflite_model = converter.convert()
    open('model_320_320.tflite', "wb").write(tflite_model)
    print("Saved tflite model!")
