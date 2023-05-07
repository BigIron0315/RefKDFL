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



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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


class Residual_block(nn.Module):

    def __init__(self, depth, num_filters, block_name='BasicBlock', K = 3, num_classes=10):
        super(Residual_block, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % (K*2) == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // (K*2)
            block = BasicBlock
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
def Residual_2block_10_33(**kwargs):
    return Residual_block(10, [33, 33, 66], 'basicblock', K = 2, **kwargs)

def Residual_2block_14_27(**kwargs):
    return Residual_block(14, [27, 27, 54], 'basicblock', K = 2, **kwargs)

def Residual_2block_22_21(**kwargs):
    return Residual_block(22, [21, 21, 42], 'basicblock', K = 2, **kwargs)

def Residual_2block_38_16(**kwargs):
    return Residual_block(38, [16, 16, 32], 'basicblock', K = 2, **kwargs)

def Residual_2block_74_11(**kwargs):
    return Residual_block(74, [11, 11, 22], 'basicblock', K = 2, **kwargs)

############ 
def Residual_2block_10_44(**kwargs):
    return Residual_block(10, [44, 44, 88], 'basicblock', K = 2, **kwargs)

def Residual_2block_14_36(**kwargs):
    return Residual_block(14, [36, 36, 72], 'basicblock', K = 2, **kwargs)

def Residual_2block_22_28(**kwargs):
    return Residual_block(22, [28, 28, 56], 'basicblock', K = 2, **kwargs)

def Residual_2block_38_21(**kwargs):
    return Residual_block(38, [21, 21, 42], 'basicblock', K = 2, **kwargs)

def Residual_2block_74_15(**kwargs):
    return Residual_block(74, [15, 15, 30], 'basicblock', K = 2, **kwargs)

############ 
def Residual_2block_10_62(**kwargs):
    return Residual_block(10, [62, 62, 124], 'basicblock', K = 2, **kwargs)

def Residual_2block_14_51(**kwargs):
    return Residual_block(14, [51, 51, 102], 'basicblock', K = 2, **kwargs)

def Residual_2block_22_39(**kwargs):
    return Residual_block(22, [39, 39, 78], 'basicblock', K = 2, **kwargs)

def Residual_2block_38_29(**kwargs):
    return Residual_block(38, [29, 29, 58], 'basicblock', K = 2, **kwargs)

def Residual_2block_74_21(**kwargs):
    return Residual_block(74, [21, 21, 42], 'basicblock', K = 2, **kwargs)

################################################################################################

############
def Residual_3block_14_33(**kwargs):
    return Residual_block(14, [33, 33, 66, 132], 'basicblock', K = 3, **kwargs)

def Residual_3block_20_27(**kwargs):
    return Residual_block(20, [27, 27, 54, 108], 'basicblock', K = 3, **kwargs)

def Residual_3block_32_21(**kwargs):
    return Residual_block(32, [21, 21, 42, 84], 'basicblock', K = 3, **kwargs)   

def Residual_3block_56_16(**kwargs):
    return Residual_block(56, [16, 16, 32, 64], 'basicblock', K = 3, **kwargs) 

def Residual_3block_110_11(**kwargs):
    return Residual_block(110, [11, 11, 22, 44], 'basicblock', K = 3, **kwargs)

############

def Residual_3block_14_44(**kwargs):
    return Residual_block(14, [44, 44, 88, 176], 'basicblock', K = 3, **kwargs)

def Residual_3block_20_36(**kwargs):
    return Residual_block(20, [36, 36, 72, 144], 'basicblock', K = 3, **kwargs)

def Residual_3block_32_28(**kwargs):
    return Residual_block(32, [28, 28, 56, 112], 'basicblock', K = 3, **kwargs)

def Residual_3block_56_21(**kwargs):
    return Residual_block(56, [21, 21, 41, 83], 'basicblock', K = 3, **kwargs)

def Residual_3block_110_15(**kwargs):
    return Residual_block(110, [15, 15, 29, 59], 'basicblock', K = 3, **kwargs)


############
def Residual_3block_14_62(**kwargs):
    return Residual_block(14, [62, 62, 125, 249], 'basicblock', K = 3, **kwargs)

def Residual_3block_20_51(**kwargs):
    return Residual_block(20, [51, 51, 102, 204], 'basicblock', K = 3, **kwargs)

def Residual_3block_32_39(**kwargs):
    return Residual_block(32, [39, 39, 78, 156], 'basicblock', K = 3, **kwargs)

def Residual_3block_56_29(**kwargs):
    return Residual_block(56, [29, 29, 59, 118], 'basicblock', K = 3, **kwargs)

def Residual_3block_110_21(**kwargs):
    return Residual_block(110, [21, 21, 42, 84], 'basicblock', K = 3, **kwargs)


################################################################################################

############
def Residual_4block_18_33(**kwargs):
    return Residual_block(18, [33, 33, 66, 132, 264], 'basicblock', K = 4, **kwargs)

def Residual_4block_26_27(**kwargs):
    return Residual_block(26, [27, 27, 54, 108, 216], 'basicblock', K = 4, **kwargs)

def Residual_4block_42_21(**kwargs):
    return Residual_block(42, [21, 21, 42, 84, 168], 'basicblock', K = 4, **kwargs)

def Residual_4block_74_16(**kwargs):
    return Residual_block(74, [16, 16, 32, 64, 128], 'basicblock', K = 4, **kwargs)

def Residual_4block_146_11(**kwargs):
    return Residual_block(146, [11, 11, 23, 45, 90], 'basicblock', K = 4, **kwargs)

############

def Residual_4block_18_44(**kwargs):
    return Residual_block(18, [44, 44, 88, 176, 352], 'basicblock', K = 4, **kwargs)

def Residual_4block_26_36(**kwargs):
    return Residual_block(26, [36, 36, 72, 144, 288], 'basicblock', K = 4, **kwargs)

def Residual_4block_42_28(**kwargs):
    return Residual_block(42, [28, 28, 56, 112, 224], 'basicblock', K = 4, **kwargs)

def Residual_4block_74_21(**kwargs):
    return Residual_block(74, [21, 21, 42, 84, 168], 'basicblock', K = 4, **kwargs)

def Residual_4block_146_15(**kwargs):
    return Residual_block(146, [15, 15, 30, 60, 120], 'basicblock', K = 4, **kwargs)

############
def Residual_4block_18_62(**kwargs):
    return Residual_block(18, [62, 62, 124, 248, 496], 'basicblock', K = 4, **kwargs)

def Residual_4block_26_51(**kwargs):
    return Residual_block(26, [51, 51, 102, 204, 408], 'basicblock', K = 4, **kwargs)

def Residual_4block_42_39(**kwargs):
    return Residual_block(42, [39, 39, 78, 156, 312], 'basicblock', K = 4, **kwargs)

def Residual_4block_74_29(**kwargs):
    return Residual_block(74, [29, 29, 58, 116, 232], 'basicblock', K = 4, **kwargs)

def Residual_4block_146_21(**kwargs):
    return Residual_block(146, [21, 21, 42, 84, 168], 'basicblock', K = 4, **kwargs)