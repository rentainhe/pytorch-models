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

## Structure
- components: put some `module` in here.
- figs: pictures
- models: the main networks
- notes: my learning notes

## Content
Network information list
- [classic network](https://github.com/rentainhe/pytorch-models/blob/master/model_information/classic-networks.md)  (`resnet`, `resnext`... )
- [efficient network](https://github.com/rentainhe/pytorch-models/blob/master/model_information/efficient-networks.md) (`mobilenet`, `shufflenet`...)
- [weight network](https://github.com/rentainhe/pytorch-models/blob/master/model_information/weight-networks.md) (`condconv`, `weightnet`...)


## Learning Notes
### 1. Convolution
|model|paper|conference|year|my notes|original github|
|:---:|:---:|:---:|:---:|:---:|:---:
| __Octave Conv__|[Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution](https://export.arxiv.org/pdf/1904.05049)|ICCV|2019|[notes](https://github.com/rentainhe/pytorch-models/blob/master/notes/efficient-networks/Octave%20Conv.md)|[github](https://github.com/lxtGH/OctaveConv_pytorch)
|__SPConv__|[Split to Be Slim: An Overlooked Redundancy in Vanilla Convolution](https://arxiv.org/abs/2006.12085)|IJCAI|2020|[notes](https://github.com/rentainhe/pytorch-models/blob/master/notes/efficient-networks/SPConv.md)|[github](https://github.com/qiulinzhang/SPConv.pytorch)
| __Deformable ConvNet__ |[Deformable Convolutional Networks](https://arxiv.org/abs/1703.06211)|ICCV(oral)|2017|...|[github](https://github.com/msracver/Deformable-ConvNets)
| __Deformable ConvNet v2__ |[Deformable ConvNets v2: More Deformable, Better Results](https://arxiv.org/abs/1811.11168)|arxiv|2018|...|[original](https://github.com/msracver/Deformable-ConvNets) [pytorch](https://github.com/4uiiurz1/pytorch-deform-conv-v2)

### 2. Pooling
|pool|paper|conference|year|my notes|original github|
|:---:|:---:|:---:|:---:|:---:|:---:
| __SoftPool__|[Refining activation downsampling with SoftPool](https://arxiv.org/abs/2101.00440)|arxiv|2021|...|[github](https://github.com/alexandrosstergiou/SoftPool#)

There are some naive tests on different pooling method, you can see my another [repo](https://github.com/rentainhe/pytorch-pooling)
## Addition
如需转载，请标明出处