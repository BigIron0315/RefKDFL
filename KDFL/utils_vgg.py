from __future__ import absolute_import
from xml.dom.pulldom import START_DOCUMENT
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class VGG(nn.Module):

    def __init__(self, cfg, batch_norm=False, num_classes=1000):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.block0 = self._make_layers(cfg[0], batch_norm, 16)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])
        #print('-----------------batch_norm : ', batch_norm)

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Linear(cfg[4][0], num_classes)
        self._initialize_weights()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.block0)
        feat_m.append(self.pool0)
        feat_m.append(self.block1)
        feat_m.append(self.pool1)
        feat_m.append(self.block2)
        feat_m.append(self.pool2)
        feat_m.append(self.block3)
        feat_m.append(self.pool3)
        feat_m.append(self.block4)
        feat_m.append(self.pool4)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block1[-1]
        bn2 = self.block2[-1]
        bn3 = self.block3[-1]
        bn4 = self.block4[-1]
        return [bn1, bn2, bn3, bn4]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        i_res = x        
        x = F.relu(self.block0(x))
        x = self.pool0(x)
        x = self.block1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = F.relu(x)
        x = self.block4(x)
        x = F.relu(x)
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, i_res

    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v, track_running_stats=False), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        layers = layers[:-1]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def vgg_scale(target_ratio, **kwargs):
    G_m = 3.24
    m = 4
    n = 5
    G_n = ((n / m) ** 2) * G_m # golden r
    basic_depth = 2
    basic_width = 64

    depth = max(int(np.round((basic_depth * (basic_width ** 2) * target_ratio / (G_n ** 2)) ** 0.2)), 1)
    width_ratio = np.sqrt(target_ratio * basic_depth / depth)

    if width_ratio > 1:
        width_ratio = 1
        depth = int(np.ceil(target_ratio * basic_depth))

    if depth > basic_depth:
        depth = basic_depth
        width_ratio = np.sqrt(target_ratio)
    
    t_w = int(np.round(width_ratio*basic_width))
    print(' #### Depth {} Width {} Target ratio {} G_n {}'.format(depth, t_w, target_ratio, G_n))
    return VGG([[t_w]*depth, [t_w*2]*depth, [t_w*4]*depth, [t_w*8]*depth, [t_w*8]*depth], batch_norm=True, num_classes=10)

def vgg13_compound(**kwargs):
    model = VGG([[32], [64], [128], [256], [256]], batch_norm=True, num_classes=10)
    return model

def vgg13_1_8(**kwargs):
    target_ratio = 0.125
    G_m = 3.24
    m = 4
    n = 5
    G_n = ((n / m) ** 2) * G_m # golden ratio
    basic_depth = 2
    basic_width = 64

    depth = max(int(np.round((basic_depth * (basic_width ** 2) * target_ratio / (G_n ** 2)) ** 0.2)), 1)
    width_ratio = np.sqrt(target_ratio * basic_depth / depth)

    if width_ratio > 1:
        width_ratio = 1
        depth = int(np.ceil(target_ratio * basic_depth))

    if depth > basic_depth:
        depth = basic_depth
        width_ratio = np.sqrt(target_ratio)
    
    t_w = int(np.round(width_ratio*basic_width))
    print('Depth {} Width {} Target ratio {}'.format(depth, t_w, target_ratio))
    if depth == 1:
        return VGG([[t_w], [t_w*2], [t_w*4], [t_w*8], [t_w*8]], batch_norm=True, num_classes=10)
    else:
        return VGG([[t_w, t_w], [t_w*2, t_w*2], [t_w*4, t_w*4], [t_w*8, t_w*8], [t_w*8, t_w*8]], batch_norm=True, num_classes=10)


def vgg13(**kwargs):
    model = VGG([[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]], batch_norm=True, num_classes=10)
    #model = VGG([[8, 8], [16, 16], [32, 32], [64, 64], [64, 64]], batch_norm=True)
    return model

def vgg_scale_cifar100(target_ratio, **kwargs):
    G_m = 3.24
    m = 4
    n = 5
    G_n = ((n / m) ** 2) * G_m # golden ratio
    basic_depth = 2
    basic_width = 64

    depth = max(int(np.round((basic_depth * (basic_width ** 2) * target_ratio / (G_n ** 2)) ** 0.2)), 1)
    width_ratio = np.sqrt(target_ratio * basic_depth / depth)

    if width_ratio > 1:
        width_ratio = 1
        depth = int(np.ceil(target_ratio * basic_depth))

    if depth > basic_depth:
        depth = basic_depth
        width_ratio = np.sqrt(target_ratio)
    
    t_w = int(np.round(width_ratio*basic_width))
    print('Depth {} Width {} Target ratio {} G_n {}'.format(depth, t_w, target_ratio, G_n))
    if depth == 1:
        return VGG([[t_w], [t_w*2], [t_w*4], [t_w*8], [t_w*8]], batch_norm=True, num_classes=100)
    else:
        return VGG([[t_w, t_w], [t_w*2, t_w*2], [t_w*4, t_w*4], [t_w*8, t_w*8], [t_w*8, t_w*8]], batch_norm=True, num_classes=100)

def vgg13_compound_cifar100(**kwargs):
    model = VGG([[32], [64], [128], [256], [256]], batch_norm=True, num_classes=100)
    return model

def vgg13_1_8_cifar100(**kwargs):
    target_ratio = 0.125
    G_m = 3.24
    m = 4
    n = 5
    G_n = ((n / m) ** 2) * G_m # golden ratio
    basic_depth = 2
    basic_width = 64

    #depth_float = (basic_depth * (basic_width ** 2) * target_ratio / (G_n ** 2)) ** 0.2
    #depth = int(np.ceil(depth_float))
    depth = max(int(np.round((basic_depth * (basic_width ** 2) * target_ratio / (G_n ** 2)) ** 0.2)), 1)
    width_ratio = np.sqrt(target_ratio * basic_depth / depth)

    if width_ratio > 1:
        width_ratio = 1
        depth = int(np.ceil(target_ratio * basic_depth))

    if depth > basic_depth:
        depth = basic_depth
        width_ratio = np.sqrt(target_ratio)
    
    t_w = int(np.round(width_ratio*basic_width))
    print('Depth {} Width {} Target ratio {}'.format(depth, t_w, target_ratio))
    if depth == 1:
        return VGG([[t_w], [t_w*2], [t_w*4], [t_w*8], [t_w*8]], batch_norm=True, num_classes=100)
    else:
        return VGG([[t_w, t_w], [t_w*2, t_w*2], [t_w*4, t_w*4], [t_w*8, t_w*8], [t_w*8, t_w*8]], batch_norm=True, num_classes=100)

def vgg13_cifar100(**kwargs):
    model = VGG([[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]], batch_norm=True, num_classes=100)
    #model = VGG([[8, 8], [16, 16], [32, 32], [64, 64], [64, 64]], batch_norm=True, num_classes=100)
    return model