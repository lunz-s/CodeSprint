# -*- coding: utf-8 -*-
"""alexnet_3d_pytorch_benchmarking.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RXh_CosiMxz4ycWzlQdyx5zM5NP95fqA
"""

#Run on google colab
#mount gdrive 
import os
# from google.colab import drive
# drive.mount('/content/drive')

#import libraries
import numpy as np
import matplotlib.pyplot as plt
#from kymatio import Scattering2D
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import scipy.misc

import torch
import torch.nn as nn
import torch.nn.functional as F

#Check PyTorch and GPU configuration
print("PyTorch Version %s" % torch.__version__)

n_gpu = torch.cuda.device_count() #no. of GPUs available
print("No. of GPUs = %d" % n_gpu)
print("GPU type %s" % torch.cuda.get_device_name(0))

if n_gpu >= 1:
  device = torch.device("cuda:0")
  Tensor = torch.cuda.FloatTensor
else:
  device = torch.device("cpu")
  Tensor = torch.FloatTensor

#specify the volume size
in_channels = 8
batch_size = 1
image_size = 128 #128^3 volume
num_classes = 2

#num_of_filters_layer1 = 64
mult_factor = 2
feature_map_dim = 55296*mult_factor #modify appropriately 

#3D alexnet
class AlexNet_3d(nn.Module):

    def __init__(self, num_classes=num_classes):
        super(AlexNet_3d, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 64*mult_factor, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(64*mult_factor, 192*mult_factor, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(192*mult_factor, 384*mult_factor, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(384*mult_factor, 256*mult_factor, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256*mult_factor, 256*mult_factor, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(feature_map_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), feature_map_dim)
        x = self.classifier(x)
        return x

#realize the model
alexnet = AlexNet_3d().to(device)

# #implement alexnet: one forward pass
# alexnet = AlexNet_3d().to(device)
# test_in = torch.randn(batch_size,in_channels,image_size,image_size,image_size).to(device)
# test_out = alexnet(test_in)
# print(test_out.size())

#do backprop
n_batches = 5
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(alexnet.parameters(),lr = 1e-3, betas = (0.5, 0.999))

for batch in np.arange(n_batches):
  image_in = torch.randn(batch_size,in_channels,image_size,image_size,image_size).to(device)
  labels = torch.randn(batch_size,num_classes).to(device)
  pred_labels = alexnet(image_in)
  
  loss_fn = criterion(pred_labels, labels)
  
  optimizer.zero_grad()
  loss_fn.backward()
  optimizer.step()

# memory footprint support libraries/code
# !ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi
# !pip install gputil
# !pip install psutil
# !pip install humanize
import psutil
import humanize
import os
import GPUtil as GPU
GPUs = GPU.getGPUs()
# XXX: only one GPU on Colab and isn’t guaranteed
gpu = GPUs[0]
def printm():
 process = psutil.Process(os.getpid())
 print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))
 print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
printm()