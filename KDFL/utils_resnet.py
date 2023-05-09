from __future__ import absolute_import
from xml.dom.pulldom import START_DOCUMENT

import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, depth, num_filters, block_name='BasicBlock', num_classes=10):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
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

def resnet_scale(target_ratio, **kwargs):
    G_m = 2.0
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
    return ResNet(int(depth*6+2), [basic_width, basic_width, target_width*2, target_width*4], 'basicblock', **kwargs)

def resnet_1_8(**kwargs):
    target_ratio = 0.125
    G_m = 2.0
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
    return ResNet(int(depth*6+2), [basic_width, basic_width, target_width*2, target_width*4], 'basicblock', **kwargs)


def resnet8(**kwargs):
    print('resnet8')
    return ResNet(8, [16, 16, 32, 64], 'basicblock', **kwargs)

def resnet14(**kwargs):
    return ResNet(14, [16, 16, 32, 64], 'basicblock', **kwargs)

def resnet20(**kwargs):
    return ResNet(20, [16, 16, 32, 64], 'basicblock', **kwargs)

def resnet26(**kwargs):
    return ResNet(26, [16, 16, 32, 64], 'basicblock', **kwargs)

def resnet56(**kwargs):
    return ResNet(56, [16, 16, 32, 64], 'basicblock', **kwargs)

def resnet_width(**kwargs):
    print('resnet_width')
    return ResNet(56, [6, 6, 13, 26], 'basicblock', **kwargs)

def resnet_compound(**kwargs):
    print('resnet_compound')
    return ResNet(26, [8, 8, 17, 34], 'basicblock', **kwargs)


def resnet_scale_base(target_ratio, **kwargs):
    if target_ratio <= 0.125:
        model_func = resnet8(**kwargs)
    elif target_ratio <= 0.25:
        model_func = resnet14(**kwargs)
    elif target_ratio <= 0.35:
        model_func = resnet20(**kwargs)
    elif target_ratio <= 0.5:
        model_func = resnet26(**kwargs)
    elif target_ratio <= 1:
        model_func = resnet56(**kwargs)
    return model_func

def resnet_scale_cifar100(target_ratio, num_classes = 100, **kwargs):
    G_m = 2.0
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
    return ResNet(int(depth*6+2), [basic_width, basic_width, target_width*2, target_width*4], 'basicblock', num_classes, **kwargs)

def resnet_1_8_cifar100(**kwargs):
    target_ratio = 0.125
    G_m = 2.0
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
    return ResNet(int(depth*6+2), [basic_width, basic_width, target_width*2, target_width*4], 'basicblock', num_classes = 100, **kwargs)

def resnet8_cifar100(**kwargs):
    return ResNet(8, [16, 16, 32, 64], 'basicblock', num_classes = 100, **kwargs)

def resnet14_cifar100(**kwargs):
    return ResNet(14, [16, 16, 32, 64], 'basicblock', num_classes = 100, **kwargs)

def resnet20_cifar100(**kwargs):
    return ResNet(20, [16, 16, 32, 64], 'basicblock', num_classes = 100, **kwargs)

def resnet26_cifar100(**kwargs):
    return ResNet(26, [16, 16, 32, 64], 'basicblock', num_classes = 100, **kwargs)

def resnet56_cifar100(**kwargs):
    return ResNet(56, [16, 16, 32, 64], 'basicblock', num_classes = 100, **kwargs)

def resnet_width_cifar100(**kwargs):
    print('resnet_width')
    return ResNet(56, [6, 6, 13, 26], 'basicblock', num_classes = 100, **kwargs)

def resnet_compound_cifar100(**kwargs):
    print('resnet_compound')
    return ResNet(26, [8, 8, 17, 34], 'basicblock', num_classes = 100, **kwargs)

def resnet_scale_base_cifar100(target_ratio, **kwargs):
    if target_ratio <= 0.125:
        model_func = resnet8_cifar100(**kwargs)
    elif target_ratio <= 0.25:
        model_func = resnet14_cifar100(**kwargs)
    elif target_ratio <= 0.35:
        model_func = resnet20_cifar100(**kwargs)
    elif target_ratio <= 0.5:
        model_func = resnet26_cifar100(**kwargs)
    elif target_ratio <= 1:
        model_func = resnet56_cifar100(**kwargs)
    return model_func