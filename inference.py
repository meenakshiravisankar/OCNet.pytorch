import numpy as np
import time
import os
import torch
from network import get_segmentation_model
import torch.nn as nn


def computeTime(model, size=(2048,128,128), device='cuda'):
    inputs = torch.randn(1, size[0], size[1], size[2])
    if device == 'cuda':
        model = model.cuda()
        inputs = inputs.cuda()

    model.eval()
    i = 0
    time_spent = []
    while i < 10:
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
        
    print('Average execution time (ms): {:.3f}'.format(np.mean(time_spent)))
    print('Max memory usage (MB): {:.3f}'.format(torch.cuda.max_memory_allocated(device=0)//(1024**2)))


model = get_segmentation_model("_".join(["resnet101", "base_oc_dsn_inf"]), num_classes=26)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
computeTime(model)
