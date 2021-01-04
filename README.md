# pytorch-models
A collection of `image-classification`, `detection`, `segmentation` models on Pytorch

## Classification Models
There may be some differences between models for `cifar100` and `imagenet`

if you want to know the information of the network, here is an example:
```bash
$ python counter.py --dataset cifar100 --model resnet18
```

### 1. Classic Network
Classic network for higher performance on cifar100 or imagenet

|dataset|network|params|
|:---:|:---:|:---:
|cifar100|resnet18|11.22M
|cifar100|resnet34|21.33M
|cifar100|resnet50|23.71M
|cifar100|resnet101|42.70M
|cifar100|resnet152|58.34M
|cifar100|resnext50|14.79M
|cifar100|resnext101|25.30M
|cifar100|resnext152|33.34M

|dataset|network|params|
|:---:|:---:|:---:
|imagenet|resnet18|11.69M
|imagenet|resnet34|21.80M
|imagenet|resnet50|25.56M
|imagenet|resnet101|44.55M
|imagenet|resnet152|60.19M

#### Implemented Network
- resnet [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385v1)
- resnext [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431v2)

### 2. Efficient Network
Efficient Network for mobile equipments

|dataset|network|params|
|:---:|:---:|:---:
|cifar100|mobilenet|3.32M
|cifar100|mobilenetV3|1.36M

#### Implemented Network
- mobilenet [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- mobilenetv3 [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf)

### 3. Weight Network
`Condconv` `Weightnet` series

|dataset|network|params|
|:---:|:---:|:---:
|imagenet|cond-mobilenetv2|22.97M

#### Implemented Network
- CondConv [CondConv: Conditionally Parameterized Convolutions for Efficient Inference.](https://arxiv.org/abs/1904.04971)