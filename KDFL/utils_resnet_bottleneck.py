from __future__ import absolute_import
from xml.dom.pulldom import START_DOCUMENT

import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        out += residual
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, depth, num_filters, block_name='BasicBlock', num_classes=10):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0], track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, num_filters[1], n)
        self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)
        self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(num_filters[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        #if stride != 1 or self.inplanes != planes * block.expansion:
        if stride != 1 or self.inplanes < planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, track_running_stats=False),
            )

        layers = list([])
        layers.append(block(self.inplanes, planes, stride, downsample, is_last=(blocks == 1)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, is_last=(i == blocks-1)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        i_res = x

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, i_res

def resnet_B_scale(target_ratio, **kwargs):
    G_m = 0.32
    m = 3
    n = 3
    G_n = ((n / m) ** 2) * G_m # golden ratio
    basic_depth = 9
    basic_width = 16

    depth = max(int(np.round((basic_depth * (basic_width ** 2) * target_ratio / (G_n ** 2)) ** 0.2)), 1)
    width_ratio = np.sqrt(target_ratio * basic_depth / depth)

    if width_ratio > 1:
        width_ratio = 1
        depth = int(np.ceil(target_ratio * basic_depth))

    if depth > basic_depth:
        depth = basic_depth
        width_ratio = np.sqrt(target_ratio)
    
    target_width = int(np.round(width_ratio*basic_width))
    return ResNet(int(depth*9+2), [basic_width, basic_width, target_width*2, target_width*4], 'bottleneck', **kwargs)

def resnet_B_1_8(**kwargs):
    target_ratio = 0.125
    G_m = 0.32
    m = 3
    n = 3
    G_n = ((n / m) ** 2) * G_m # golden ratio
    basic_depth = 9
    basic_width = 16

    depth = max(int(np.round((basic_depth * (basic_width ** 2) * target_ratio / (G_n ** 2)) ** 0.2)), 1)
    width_ratio = np.sqrt(target_ratio * basic_depth / depth)

    if width_ratio > 1:
        width_ratio = 1
        depth = int(np.ceil(target_ratio * basic_depth))

    if depth > basic_depth:
        depth = basic_depth
        width_ratio = np.sqrt(target_ratio)
    
    target_width = int(np.round(width_ratio*basic_width))
    return ResNet(int(depth*9+2), [basic_width, basic_width, target_width*2, target_width*4], 'bottleneck', **kwargs)


def resnet11_B(**kwargs):
    return ResNet(11, [16, 16, 32, 64], 'bottleneck', **kwargs)

def resnet20_B(**kwargs):
    return ResNet(20, [16, 16, 32, 64], 'bottleneck', **kwargs)

def resnet29_B(**kwargs):
    return ResNet(29, [16, 16, 32, 64], 'bottleneck', **kwargs)

def resnet38_B(**kwargs):
    return ResNet(38, [16, 16, 32, 64], 'bottleneck', **kwargs)

def resnet83_B(**kwargs):
    return ResNet(83, [16, 16, 32, 64], 'bottleneck', **kwargs)

def resnet_B_width(**kwargs):
    return ResNet(83, [6, 6, 13, 26], 'bottleneck', **kwargs)

def resnet_B_compound(**kwargs):
    return ResNet(38, [8, 8, 17, 34], 'bottleneck', **kwargs)


def resnet_B_scale_base(target_ratio, **kwargs):
    if target_ratio <= 0.125:
        model_func = resnet11_B(**kwargs)
    elif target_ratio <= 0.25:
        model_func = resnet20_B(**kwargs)
    elif target_ratio <= 0.35:
        model_func = resnet29_B(**kwargs)
    elif target_ratio <= 0.5:
        model_func = resnet38_B(**kwargs)
    elif target_ratio <= 1:
        model_func = resnet83_B(**kwargs)
    return model_func

def resnet_B_scale_cifar100(target_ratio, num_classes = 100, **kwargs):
    G_m = 0.32
    m = 3
    n = 3
    G_n = ((n / m) ** 2) * G_m # golden ratio
    basic_depth = 9
    basic_width = 16

    depth = max(int(np.round((basic_depth * (basic_width ** 2) * target_ratio / (G_n ** 2)) ** 0.2)), 1)
    width_ratio = np.sqrt(target_ratio * basic_depth / depth)

    if width_ratio > 1:
        width_ratio = 1
        depth = int(np.ceil(target_ratio * basic_depth))

    if depth > basic_depth:
        depth = basic_depth
        width_ratio = np.sqrt(target_ratio)
    
    target_width = int(np.round(width_ratio*basic_width))
    return ResNet(int(depth*9+2), [basic_width, basic_width, target_width*2, target_width*4], 'bottleneck', num_classes, **kwargs)

def resnet_B_1_8_cifar100(**kwargs):
    target_ratio = 0.125
    G_m = 0.32
    m = 3
    n = 3
    G_n = ((n / m) ** 2) * G_m # golden ratio
    basic_depth = 9
    basic_width = 16

    depth = max(int(np.round((basic_depth * (basic_width ** 2) * target_ratio / (G_n ** 2)) ** 0.2)), 1)
    width_ratio = np.sqrt(target_ratio * basic_depth / depth)

    if width_ratio > 1:
        width_ratio = 1
        depth = int(np.ceil(target_ratio * basic_depth))

    if depth > basic_depth:
        depth = basic_depth
        width_ratio = np.sqrt(target_ratio)
    
    target_width = int(np.round(width_ratio*basic_width))
    return ResNet(int(depth*9+2), [basic_width, basic_width, target_width*2, target_width*4], 'bottleneck', num_classes = 100, **kwargs)

def resnet_B_width_cifar100(**kwargs):
    return ResNet(83, [6, 6, 12, 24], 'bottleneck', num_classes = 100, **kwargs)

def resnet_B_compound_cifar100(**kwargs):
    return ResNet(38, [8, 8, 17, 34], 'bottleneck', num_classes = 100, **kwargs)

def resnet11_B_cifar100(**kwargs):
    return ResNet(11, [16, 16, 32, 64], 'bottleneck', num_classes = 100, **kwargs)

def resnet20_B_cifar100(**kwargs):
    return ResNet(20, [16, 16, 32, 64], 'bottleneck', num_classes = 100, **kwargs)

def resnet29_B_cifar100(**kwargs):
    return ResNet(29, [16, 16, 32, 64], 'bottleneck', num_classes = 100, **kwargs)

def resnet38_B_cifar100(**kwargs):
    return ResNet(38, [16, 16, 32, 64], 'bottleneck', num_classes = 100, **kwargs)

def resnet83_B_cifar100(**kwargs):
    return ResNet(83, [16, 16, 32, 64], 'bottleneck', num_classes = 100, **kwargs)

def resnet_B_scale_base_cifar100(target_ratio, **kwargs):
    if target_ratio <= 0.125:
        model_func = resnet11_B_cifar100(**kwargs)
    elif target_ratio <= 0.25:
        model_func = resnet20_B_cifar100(**kwargs)
    elif target_ratio <= 0.35:
        model_func = resnet29_B_cifar100(**kwargs)
    elif target_ratio <= 0.5:
        model_func = resnet38_B_cifar100(**kwargs)
    elif target_ratio <= 1:
        model_func = resnet83_B_cifar100(**kwargs)
    return model_func