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
python3 main.py --model Residual_3block_14_33 --dataset cifar10 --iter 1
```

To train a model through federated learning, run this command: train teacher network with centralized manner(TA_train 1), batch size 50, public data ratio 10%, fed_a(FedAvg), fed_b(FedProx), fed_c(Scaffold), fed_d(RefKDFL). Once teacher network is trained, it doesn't need to be train again.
```train FL
python3 main_cifar10.py --path_t ./save/models/TA_Resnet110_CIFAR10/PublicRatio_10/TA_Resnet110_CIFAR10.pth --isDistill 1 --dataset cifar10 --model_name resnet --TA_train 1 --target_ratio 0.125 --batch 50 --Dir 0.3 --publicRatio 0.1 --fed_a 1 --fed_b 1 --fed_c 1 --fed_d 1
```

# Results
<p float="left">
<img src="https://user-images.githubusercontent.com/91996704/236240591-65b5f062-796b-4261-878a-0e28ff89d714.PNG" width="240" height="300">
<img src="https://user-images.githubusercontent.com/91996704/236240614-d84af1d3-931f-4d9b-99fe-5219be0e4655.PNG" width="240" height="300">
<img src="https://user-images.githubusercontent.com/91996704/236240486-75d63bc3-941c-495a-b5c5-bbb7fa4e4941.PNG" width="240" height="300">
</p>

<p float="left">
<img src="https://user-images.githubusercontent.com/91996704/236241012-ff19a62b-44dd-4220-951c-2a489c22c3d8.PNG" width="240" height="180">
<img src="https://user-images.githubusercontent.com/91996704/236241056-e0cd8a20-0bfd-43b4-b3b0-8e6620cca393.PNG" width="240" height="180">
<img src="https://user-images.githubusercontent.com/91996704/236241092-d887c037-946f-4240-954b-282acf83241b.PNG" width="240" height="180">
</p>
| Method         | MFLOPs  | Model size(MB) | 0.1  | 0.2  | 0.5 | 
| Method         | MFLOPs  | Model size(MB) | 0.1  | 0.2  | 0.5 | 
| -------------- |-------- | -------------- |------- |------- |------- |
| FedAvg   |     134         |      3.6       |78.63|78.24|68.54
| FedProx |     134         |      3.6       |78.63|78.24|68.54
| Scaffold  |       134         |      3.6        |78.63|78.24|68.54
| RefKDFL  |       39         |      0.6        |78.91|79.29|71.85
