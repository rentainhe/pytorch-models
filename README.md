# pytorch-models
A collection of `image-classification`, `detection`, `segmentation` models on Pytorch

## Computer Vision Models
There may be some differences between models for `cifar100` and `imagenet`

if you want to know the information of the network, here is an example:
```bash
$ python counter.py --dataset cifar100 --model resnet18
```

The supported net args are:
```
resnet18  resnet34  resnet50  resnet101  resnet152
mobilenet mobilenetv3
resnext50  resnext101  resnext152
Octresnet50  Octresnet101  Octresnet152
```

## Content
Network information list
- [classic network](https://github.com/rentainhe/pytorch-models/blob/master/model_information/classic-networks.md)  (`resnet`, `resnext`... )
- [efficient network](https://github.com/rentainhe/pytorch-models/blob/master/model_information/efficient-networks.md) (`mobilenet`, `shufflenet`...)
- [weight network](https://github.com/rentainhe/pytorch-models/blob/master/model_information/weight-networks.md) (`condconv`, `weightnet`...)


## Learning Notes
|model|paper|conference|year|my notes|original github|
|:---:|:---:|:---:|:---:|:---:|:---:
| __Octave Conv__|[Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution](https://export.arxiv.org/pdf/1904.05049)|ICCV|2019|[notes](https://github.com/rentainhe/pytorch-models/blob/master/notes/efficient-networks/Octave%20Conv.md)|[github](https://github.com/lxtGH/OctaveConv_pytorch)|
|__SPConv__|[Split to Be Slim: An Overlooked Redundancy in Vanilla Convolution](https://arxiv.org/abs/2006.12085)|IJCAI|2020|[notes]()|[github](https://github.com/qiulinzhang/SPConv.pytorch)
