# RefKDFL
This repository is the official implementation of RefKDFL:Resource-efficient Knowledge Distilled Federated Learning.

# Requirements
To install requirements:

```setup
pip install -r requirements.txt
```

# Training + Evaluation

To train the model to find a golden ratio, run this command: train cifar10 with residual block where K = 3, and d = 2, and W<sub>1</sub> = 33
```train model
python3 main.py --model Residual_3block_14_33 --dataset cifar10
```

To train a model through federated learning, run these commands
- Train RefKDFL, (Public data 10%, CIFAR-10 dataset, Resnet with residual block, target model size 0.125). If teacher network training is required, TA_train should be 1. fed_a(FedAvg), fed_b(FedProx), fed_c(Scaffold), fed_d(RefKDFL).
```train FL
python3 main_cifar10.py --path_t ./save/models/TA_Resnet110_CIFAR10/PublicRatio_10/TA_Resnet110_CIFAR10.pth --isDistill 1 --dataset cifar10 --model_name resnet --TA_train 0 --target_ratio 0.125 --Dir 0.3 --publicRatio 0.1 --fed_a 0 --fed_b 0 --fed_c 0 --fed_d 1
```

- Train FedAvg, FedProx, Scaffold (Public data 10%, CIFAR-10 dataset, Resnet with residual block, target model size 1(Resnet56))
```train FL
python3 main_cifar10.py --path_t ./save/models/TA_Resnet110_CIFAR10/PublicRatio_10/TA_Resnet110_CIFAR10.pth --isDistill 1 --dataset cifar10 --model_name resnet --TA_train 0 --target_ratio 1 --Dir 0.3 --publicRatio 0.1 --fed_a 1 --fed_b 1 --fed_c 1 --fed_d 0
```


# Results
### - Test accuracy with 3 popular convolutional building blocks ( basic block, residual block, bottleneck block)
<p float="left">
<img src="https://user-images.githubusercontent.com/91996704/236240591-65b5f062-796b-4261-878a-0e28ff89d714.PNG" width="240" height="300">
<img src="https://user-images.githubusercontent.com/91996704/236240614-d84af1d3-931f-4d9b-99fe-5219be0e4655.PNG" width="240" height="300">
<img src="https://user-images.githubusercontent.com/91996704/236240486-75d63bc3-941c-495a-b5c5-bbb7fa4e4941.PNG" width="240" height="300">
</p>


## 1. CIFAR-10 dataset
### - Resnet with bottleneck block
| Method   | MFLOPs  | Model size(MB) |P = 0.1 |P = 0.2 | P = 0.5| 
| ---------|-------- | -------------- |------- |------- |------- |
| FedAvg   |  134    |      3.6       |  78.63 |  78.24 |  68.54 |
| FedProx  |  134    |      3.6       |  79.27 |  79.15 |  71.54 |
| Scaffold |  134    |      3.6       |  62.7  |  62.6  |  52.36 |
| RefKDFL  |   39    |      0.6       |  78.91 |  79.29 |  71.85 |

### - Resnet with residual block
| Method   | MFLOPs  | Model size(MB) |P = 0.1 |P = 0.2 | P = 0.5| 
| ---------|-------- | -------------- |------- |------- |------- |
| FedAvg   |  128    |      3.5       |  80.73 |  81.57 |  72.16 |
| FedProx  |  128    |      3.5       |  81.07 |  80.74 |  72.5  |
| Scaffold |  128    |      3.5       |  73.53 |  74.55 |  66.75 |
| RefKDFL  |   20    |      0.4       |  82.39 |  82.52 |  78.68 |

### - Vgg with basic block
| Method   | MFLOPs  | Model size(MB) |P = 0.1 |P = 0.2 | P = 0.5| 
| ---------|-------- | -------------- |------- |------- |------- |
| FedAvg   |  290    |      37.7      |  86.07 |  84.34 |  82.09 |
| FedProx  |  290    |      37.7      |  85.53 |  84.62 |  82.68 |
| Scaffold |  290    |      37.7      |  83.67 |  83.35 |  79.01 |
| RefKDFL  |   40    |      5         |  85.79 |  85.26 |  83.3  |


## 2. CIFAR-100 dataset
### - Resnet with bottleneck block
| Method   | MFLOPs  | Model size(MB) |P = 0.1 |P = 0.2 | P = 0.5| 
| ---------|-------- | -------------- |------- |------- |------- |
| FedAvg   |  134    |      3.6       |  61.01 |  58.91 |  48.53 |
| FedProx  |  134    |      3.6       |  61.69 |  59.03 |  46.94 |
| Scaffold |  134    |      3.6       |  46.56 |  43.82 |  30.42 |
| RefKDFL  |   55    |      1.1       |  60.67 |  61.34 |  54.79 |

### - Resnet with residual block
| Method   | MFLOPs  | Model size(MB) |P = 0.1 |P = 0.2 | P = 0.5| 
| ---------|-------- | -------------- |------- |------- |------- |
| FedAvg   |  128    |      3.5       |  59.91 |  57.92 |  48.38 |
| FedProx  |  128    |      3.5       |  60.39 |  58.43 |  47.38 |
| Scaffold |  128    |      3.5       |  51.72 |  49.42 |  39.95 |
| RefKDFL  |   35    |      0.9       |  59.93 |  60.03 |  56.62 |

### - Vgg with basic block
| Method   | MFLOPs  | Model size(MB) |P = 0.1 |P = 0.2 | P = 0.5| 
| ---------|-------- | -------------- |------- |------- |------- |
| FedAvg   |  290    |      37.7      |  62.27 |  60.99 |  52.37 |
| FedProx  |  290    |      37.7      |  62.74 |  61.09 |  53.0  |
| Scaffold |  290    |      37.7      |  64.57 |  62.41 |  55.65 |
| RefKDFL  |   77    |      9         |  62.57 |  61.65 |  55.17 |

### - Test accuracy on CIFAR-10 with public data ratio 0.2
<p float="left">
<img src="https://user-images.githubusercontent.com/91996704/236241012-ff19a62b-44dd-4220-951c-2a489c22c3d8.PNG" width="240" height="180">
<img src="https://user-images.githubusercontent.com/91996704/236241056-e0cd8a20-0bfd-43b4-b3b0-8e6620cca393.PNG" width="240" height="180">
<img src="https://user-images.githubusercontent.com/91996704/236241092-d887c037-946f-4240-954b-282acf83241b.PNG" width="240" height="180">
</p>
