import torch.nn as nn
import torch.nn.functional as F
import math

class Basic_block(nn.Module):

    def __init__(self, cfg, batch_norm=False, K = 5, num_classes=10):
        super(Basic_block, self).__init__()
        self.K = K
        self.block0 = self._make_layers(cfg[0], batch_norm, 3)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
        if self.K > 3:
            self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
        if self.K > 4:
            self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        if self.K == 3:
            self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(cfg[2][0], num_classes)
        elif self.K == 4:
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(cfg[3][0], num_classes)
        elif self.K == 5:
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(cfg[4][0], num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.block0(x))
        x = self.pool0(x)
        x = self.block1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = F.relu(x)
        x = self.pool2(x)
        if self.K > 3:
            x = self.block3(x)
            x = F.relu(x)
            x = self.pool3(x)
        if self.K > 4:
            x = self.block4(x)
            x = F.relu(x)
            x = self.pool4(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
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

################################################################################################
############
def Basic_3block_2_16(**kwargs):
    model = Basic_block([[15]*2, [30]*2, [60]*2], batch_norm=True, K = 3)
    return model

def Basic_3block_3_12(**kwargs):
    model = Basic_block([[12]*3, [24]*3, [48]*3], batch_norm=True, K = 3)
    return model

def Basic_3block_4_10(**kwargs):
    model = Basic_block([[10]*4, [20]*4, [40]*4], batch_norm=True, K = 3)
    return model

def Basic_3block_6_8(**kwargs):
    model = Basic_block([[8]*6, [16]*6, [32]*6], batch_norm=True, K = 3)
    return model

def Basic_3block_8_7(**kwargs):
    model = Basic_block([[7]*8, [14]*8, [29]*8], batch_norm=True, K = 3)
    return model

############
def Basic_3block_2_24(**kwargs):
    model = Basic_block([[23]*2, [46]*2, [92]*2], batch_norm=True, K = 3)
    return model

def Basic_3block_3_19(**kwargs):
    model = Basic_block([[19]*3, [37]*3, [75]*3], batch_norm=True, K = 3)
    return model

def Basic_3block_4_16(**kwargs):
    model = Basic_block([[16]*4, [32]*4, [64]*4], batch_norm=True, K = 3)
    return model

def Basic_3block_6_13(**kwargs):
    model = Basic_block([[13]*6, [25]*6, [50]*6], batch_norm=True, K = 3)
    return model

def Basic_3block_8_11(**kwargs):
    model = Basic_block([[11]*8, [21]*8, [43]*8], batch_norm=True, K = 3)
    return model

############
def Basic_3block_2_48(**kwargs):
    model = Basic_block([[45]*2, [90]*2, [180]*2], batch_norm=True, K = 3)
    return model

def Basic_3block_3_37(**kwargs):
    model = Basic_block([[37]*3, [74]*3, [149]*3], batch_norm=True, K = 3)
    return model

def Basic_3block_4_31(**kwargs):
    model = Basic_block([[31]*4, [63]*4, [126]*4], batch_norm=True, K = 3)
    return model

def Basic_3block_6_25(**kwargs):
    model = Basic_block([[25]*6, [50]*6, [100]*6], batch_norm=True, K = 3)
    return model

def Basic_3block_8_21(**kwargs):
    model = Basic_block([[21]*8, [43]*8, [86]*8], batch_norm=True, K = 3)
    return model

################################################################################################

def Basic_4block_1_28(**kwargs):
    model = Basic_block([[28], [56], [112], [224]], batch_norm=True, K = 4)
    return model

def Basic_4block_1_42(**kwargs):
    model = Basic_block([[42], [84], [168], [336]], batch_norm=True, K = 4)
    return model

def Basic_4block_1_83(**kwargs):
    model = Basic_block([[83], [166], [332], [664]], batch_norm=True, K = 4)
    return model

def Basic_4block_2_16(**kwargs):
    model = Basic_block([[15]*2, [30]*2, [60]*2, [120]*2], batch_norm=True, K = 4)
    return model

def Basic_4block_3_12(**kwargs):
    model = Basic_block([[12]*3, [24]*3, [48]*3, [96]*3], batch_norm=True, K = 4)
    return model

def Basic_4block_4_10(**kwargs):
    model = Basic_block([[10]*4, [21]*4, [42]*4, [84]*4], batch_norm=True, K = 4)
    return model

def Basic_4block_6_8(**kwargs):
    model = Basic_block([[8]*6, [17]*6, [33]*6, [66]*6], batch_norm=True, K = 4)
    return model

def Basic_4block_8_7(**kwargs):
    model = Basic_block([[7]*8, [14]*8, [29]*8, [58]*8], batch_norm=True, K = 4)
    return model

############
def Basic_4block_2_24(**kwargs):
    model = Basic_block([[23]*2, [46]*2, [92]*2, [184]*2], batch_norm=True, K = 4)
    return model

def Basic_4block_3_19(**kwargs):
    model = Basic_block([[19]*3, [37]*3, [75]*3, [150]*3], batch_norm=True, K = 4)
    return model

def Basic_4block_4_16(**kwargs):
    model = Basic_block([[16]*4, [32]*4, [64]*4, [128]*4], batch_norm=True, K = 4)
    return model

def Basic_4block_6_13(**kwargs):
    model = Basic_block([[13]*6, [25]*6, [50]*6, [100]*6], batch_norm=True, K = 4)
    return model

def Basic_4block_8_11(**kwargs):
    model = Basic_block([[11]*8, [21]*8, [43]*8, [86]*8], batch_norm=True, K = 4)
    return model

############
def Basic_4block_2_48(**kwargs):
    model = Basic_block([[45]*2, [90]*2, [180]*2, [360]*2], batch_norm=True, K = 4)
    return model

def Basic_4block_3_37(**kwargs):
    model = Basic_block([[37]*3, [74]*3, [149]*3, [298]*3], batch_norm=True, K = 4)
    return model

def Basic_4block_4_31(**kwargs):
    model = Basic_block([[31]*4, [63]*4, [126]*4, [252]*4], batch_norm=True, K = 4)
    return model

def Basic_4block_6_25(**kwargs):
    model = Basic_block([[25]*6, [50]*6, [100]*6, [200]*6], batch_norm=True, K = 4)
    return model

def Basic_4block_8_21(**kwargs):
    model = Basic_block([[21]*8, [43]*8, [86]*8, [172]*8], batch_norm=True, K = 4)
    return model

##########################################################################################

def Basic_5block_2_16(**kwargs):
    model = Basic_block([[16]*2, [32]*2, [64]*2, [128]*2, [256]*2], batch_norm=True, K = 5)
    return model

def Basic_5block_3_12(**kwargs):
    model = Basic_block([[12]*3, [24]*3, [48]*3, [96]*3, [192]*3], batch_norm=True, K = 5)
    return model

def Basic_5block_4_10(**kwargs):
    model = Basic_block([[10]*4, [21]*4, [42]*4, [84]*4, [168]*4], batch_norm=True, K = 5)
    return model

def Basic_5block_6_8(**kwargs):
    model = Basic_block([[8]*6, [17]*6, [33]*6, [66]*6, [132]*6], batch_norm=True, K = 5)
    return model

def Basic_5block_8_7(**kwargs):
    model = Basic_block([[7]*8, [14]*8, [29]*8, [58]*8, [116]*8], batch_norm=True, K = 5)
    return model

############
def Basic_5block_2_24(**kwargs):
    model = Basic_block([[24]*2, [48]*2, [96]*2, [192]*2, [384]*2], batch_norm=True, K = 5)
    return model

def Basic_5block_3_19(**kwargs):
    model = Basic_block([[19]*3, [37]*3, [75]*3, [150]*3, [300]*3], batch_norm=True, K = 5)
    return model

def Basic_5block_4_16(**kwargs):
    model = Basic_block([[16]*4, [32]*4, [64]*4, [128]*4, [256]*4], batch_norm=True, K = 5)
    return model

def Basic_5block_6_13(**kwargs):
    model = Basic_block([[13]*6, [25]*6, [50]*6, [100]*6, [200]*6], batch_norm=True, K = 5)
    return model

def Basic_5block_8_11(**kwargs):
    model = Basic_block([[11]*8, [21]*8, [43]*8, [86]*8, [172]*8], batch_norm=True, K = 5)
    return model

############
def Basic_5block_2_48(**kwargs):
    model = Basic_block([[48]*2, [96]*2, [192]*2, [384]*2, [768]*2], batch_norm=True, K = 5)
    return model

def Basic_5block_3_37(**kwargs):
    model = Basic_block([[37]*3, [74]*3, [149]*3, [298]*3, [596]*3], batch_norm=True, K = 5)
    return model

def Basic_5block_4_31(**kwargs):
    model = Basic_block([[31]*4, [63]*4, [126]*4, [252]*4, [504]*4], batch_norm=True, K = 5)
    return model

def Basic_5block_6_25(**kwargs):
    model = Basic_block([[25]*6, [50]*6, [100]*6, [200]*6, [400]*6], batch_norm=True, K = 5)
    return model

def Basic_5block_8_21(**kwargs):
    model = Basic_block([[21]*8, [43]*8, [86]*8, [172]*8, [344]*8], batch_norm=True, K = 5)
    return model