from __future__ import absolute_import
from xml.dom.pulldom import START_DOCUMENT

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = ['resnet']

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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
        preact = out
        out = F.relu(out)
        return out


class Bottleneck_block(nn.Module):

    def __init__(self, depth, num_filters, block_name='BasicBlock', K = 3, num_classes=10):
        super(Bottleneck_block, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'bottleneck':
            assert (depth - 2) % (K*3) == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // (K*3)
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.K = K
        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, num_filters[1], n)
        self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)
        self.avgpool = nn.AvgPool2d(16)
        if self.K > 2:
            self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)
            self.avgpool = nn.AvgPool2d(8)
        if self.K > 3:
            self.layer4 = self._make_layer(block, num_filters[4], n, stride=2)
            self.avgpool = nn.AvgPool2d(4)
        
        self.fc = nn.Linear(num_filters[K] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
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
        x = self.layer1(x)
        x = self.layer2(x)
        if self.K > 2:
            x = self.layer3(x)
        if self.K > 3:
            x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

############
def Bottle_2block_14_32(**kwargs):
    return Bottleneck_block(14, [32, 32, 64], 'bottleneck', K = 2, **kwargs)

def Bottle_2block_20_27(**kwargs):
    return Bottleneck_block(20, [27, 27, 54], 'bottleneck', K = 2, **kwargs)

def Bottle_2block_32_21(**kwargs):
    return Bottleneck_block(32, [21, 21, 42], 'bottleneck', K = 2, **kwargs)

def Bottle_2block_56_16(**kwargs):
    return Bottleneck_block(56, [16, 16, 32], 'bottleneck', K = 2, **kwargs)

def Bottle_2block_110_11(**kwargs):
    return Bottleneck_block(110, [11, 11, 22], 'bottleneck', K = 2, **kwargs)

############
def Bottle_2block_14_42(**kwargs):
    return Bottleneck_block(14, [42, 42, 84], 'bottleneck', K = 2, **kwargs)

def Bottle_2block_20_35(**kwargs):
    return Bottleneck_block(20, [35, 35, 70], 'bottleneck', K = 2, **kwargs)

def Bottle_2block_32_28(**kwargs):
    return Bottleneck_block(32, [28, 28, 56], 'bottleneck', K = 2, **kwargs)

def Bottle_2block_56_21(**kwargs):
    return Bottleneck_block(56, [21, 21, 42], 'bottleneck', K = 2, **kwargs)

def Bottle_2block_110_15(**kwargs):
    return Bottleneck_block(110, [15, 15, 30], 'bottleneck', K = 2, **kwargs)

############
def Bottle_2block_14_60(**kwargs):
    return Bottleneck_block(14, [60, 60, 120], 'bottleneck', K = 2, **kwargs)

def Bottle_2block_20_51(**kwargs):
    return Bottleneck_block(20, [51, 51, 102], 'bottleneck', K = 2, **kwargs)

def Bottle_2block_32_39(**kwargs):
    return Bottleneck_block(32, [39, 39, 78], 'bottleneck', K = 2, **kwargs)

def Bottle_2block_56_29(**kwargs):
    return Bottleneck_block(56, [29, 29, 58], 'bottleneck', K = 2, **kwargs)

def Bottle_2block_110_21(**kwargs):
    return Bottleneck_block(110, [21, 21, 42], 'bottleneck', K = 2, **kwargs)  

################################################################################################
def Bottle_3block_20_32(**kwargs):
    return Bottleneck_block(20, [32, 32, 64, 128], 'bottleneck', K = 3, **kwargs)

def Bottle_3block_29_27(**kwargs):
    return Bottleneck_block(29, [27, 27, 54, 108], 'bottleneck', K = 3, **kwargs)

def Bottle_3block_47_21(**kwargs):
    return Bottleneck_block(47, [21, 21, 42, 84], 'bottleneck', K = 3, **kwargs)

def Bottle_3block_83_16(**kwargs):
    return Bottleneck_block(83, [16, 16, 32, 64], 'bottleneck', K = 3, **kwargs)

def Bottle_3block_174_11(**kwargs):
    return Bottleneck_block(174, [11, 11, 22, 44], 'bottleneck', K = 3, **kwargs)

############
def Bottle_3block_20_42(**kwargs):
    return Bottleneck_block(20, [42, 42, 84, 168], 'bottleneck', K = 3, **kwargs)

def Bottle_3block_29_35(**kwargs):
    return Bottleneck_block(29, [35, 35, 70, 140], 'bottleneck', K = 3, **kwargs)

def Bottle_3block_47_28(**kwargs):
    return Bottleneck_block(47, [28, 28, 56, 112], 'bottleneck', K = 3, **kwargs)

def Bottle_3block_83_21(**kwargs):
    return Bottleneck_block(83, [21, 21, 42, 84], 'bottleneck', K = 3, **kwargs)

def Bottle_3block_174_15(**kwargs):
    return Bottleneck_block(174, [15, 15, 30, 60], 'bottleneck', K = 3, **kwargs)

############
def Bottle_3block_20_60(**kwargs):
    return Bottleneck_block(20, [60, 60, 120, 240], 'bottleneck', K = 3, **kwargs)

def Bottle_3block_29_51(**kwargs):
    return Bottleneck_block(29, [51, 51, 102, 204], 'bottleneck', K = 3, **kwargs)

def Bottle_3block_47_39(**kwargs):
    return Bottleneck_block(47, [39, 39, 78, 156], 'bottleneck', K = 3, **kwargs)

def Bottle_3block_83_29(**kwargs):
    return Bottleneck_block(83, [29, 29, 58, 116], 'bottleneck', K = 3, **kwargs)

def Bottle_3block_174_21(**kwargs):
    return Bottleneck_block(174, [21, 21, 42, 84], 'bottleneck', K = 3, **kwargs)  

################################################################################################

def Bottle_4block_26_32(**kwargs):
    return Bottleneck_block(26, [32, 32, 64, 128, 256], 'bottleneck', K = 4, **kwargs)

def Bottle_4block_38_27(**kwargs):
    return Bottleneck_block(38, [27, 27, 54, 108, 216], 'bottleneck', K = 4, **kwargs)

def Bottle_4block_62_21(**kwargs):
    return Bottleneck_block(62, [21, 21, 42, 84, 168], 'bottleneck', K = 4, **kwargs)

def Bottle_4block_110_16(**kwargs):
    return Bottleneck_block(110, [16, 16, 32, 64, 128], 'bottleneck', K = 4, **kwargs)

def Bottle_4block_218_11(**kwargs):
    return Bottleneck_block(218, [11, 11, 22, 44, 88], 'bottleneck', K = 4, **kwargs)

############
def Bottle_4block_26_42(**kwargs):
    return Bottleneck_block(26, [42, 42, 84, 168, 336], 'bottleneck', K = 4, **kwargs)

def Bottle_4block_38_35(**kwargs):
    return Bottleneck_block(38, [35, 35, 70, 140, 280], 'bottleneck', K = 4, **kwargs)

def Bottle_4block_62_28(**kwargs):
    return Bottleneck_block(62, [28, 28, 56, 112, 224], 'bottleneck', K = 4, **kwargs)

def Bottle_4block_110_21(**kwargs):
    return Bottleneck_block(110, [21, 21, 42, 84, 168], 'bottleneck', K = 4, **kwargs)

def Bottle_4block_218_15(**kwargs):
    return Bottleneck_block(218, [15, 15, 30, 60, 120], 'bottleneck', K = 4, **kwargs)

############
def Bottle_4block_26_60(**kwargs):
    return Bottleneck_block(26, [60, 60, 120, 240, 480], 'bottleneck', K = 4, **kwargs)

def Bottle_4block_38_51(**kwargs):
    return Bottleneck_block(38, [51, 51, 102, 204, 408], 'bottleneck', K = 4, **kwargs)

def Bottle_4block_62_39(**kwargs):
    return Bottleneck_block(62, [39, 39, 78, 156, 312], 'bottleneck', K = 4, **kwargs)

def Bottle_4block_110_29(**kwargs):
    return Bottleneck_block(110, [29, 29, 58, 116, 232], 'bottleneck', K = 4, **kwargs)

def Bottle_4block_218_21(**kwargs):
    return Bottleneck_block(218, [21, 21, 42, 84, 168], 'bottleneck', K = 4, **kwargs)  

################################################################################################
