from .TA_Resnet import resnet110, resnet110_Cifar100
from .TA_Vgg import vgg19, vgg19_Cifar100
model_dict1 = {
    'TA_Resnet110_CIFAR10' : resnet110,
    'TA_Resnet110_CIFAR100' : resnet110_Cifar100,
    'TA_Vgg_CIFAR10' : vgg19,
    'TA_Vgg_CIFAR100' : vgg19_Cifar100,
}
