# pytorch-models
A collection of `image-classification`, `detection`, `segmentation` models on Pytorch

## Computer Vision Models
There may be some differences between models for `cifar100` and `imagenet`

if you want to know the information of the network, here is an example:
```bash
$ python counter.py --dataset cifar100 --model resnet18
```

The supported `--model` args are:
- classic networks
```
resnet18 resnet34 resnet50 resnet101 resnet152
resnext50 resnext101 resnext152
```

- efficient networks
```
mobilenet mobilenetv3
Octresnet50 Octresnet101 Octresnet152
spresnet18 spresnet34 spresnet50 spresnet101 spresnet152
```


## Structure
- `/components`: put some `module` in here.
- `/figs`: pictures
- `/models`: the main networks
- `/notes`: my learning notes

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
| __ACNet__ |[ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Contolution Blocks](https://arxiv.org/pdf/1908.03930.pdf)|ICCV|2019|[notes](https://github.com/rentainhe/pytorch-models/blob/master/notes/efficient-networks/ACNet.md)|[github](https://github.com/DingXiaoH/ACNet)
| __RepVGG__ |[RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)|arxiv|2021|...|[github](https://github.com/DingXiaoH/RepVGG)
| __ECANet__ |[ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks](https://arxiv.org/abs/1910.03151)|CVPR|2020| ... |[github](https://github.com/BangguWu/ECANet)
| __MobileNetV2__ |[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)|arxiv|2018| [notes](https://github.com/rentainhe/pytorch-models/blob/master/notes/classic-networks/mobilenetv2.md) | [github](https://github.com/tonylins/pytorch-mobilenet-v2)
| __Triplet Attention__ |[Rotate to Attend: Convolutional Triplet Attention Module](https://arxiv.org/abs/2010.03045)|WACV|2021| [notes](https://zhuanlan.zhihu.com/p/360028132) | [github](https://github.com/LandskapeAI/triplet-attention)

### 2. Vision Transformer
|model|paper|conference|year|my notes|original github|
|:---:|:---:|:---:|:---:|:---:|:---:
| __T2T_ViT__ |[Tokens-to-Token ViT: Training Vision Transformers from Scratch on Imagenet](https://arxiv.org/abs/2101.11986)|arxiv|2021| [notes](https://github.com/rentainhe/pytorch-models/blob/master/notes/vision-transformer/T2T_ViT.md) |[github](https://github.com/yitu-opensource/T2T-ViT)
| __BoTNet__ |[Bottleneck Transformers for Visual Recognition](https://arxiv.org/pdf/2101.11605.pdf)|arxiv|2021| ... |[github](https://github.com/leaderj1001/BottleneckTransformers)


### 3. Pooling
|pool|paper|conference|year|my notes|original github|
|:---:|:---:|:---:|:---:|:---:|:---:
| __SoftPool__|[Refining activation downsampling with SoftPool](https://arxiv.org/abs/2101.00440)|arxiv|2021|[notes](https://github.com/rentainhe/pytorch-models/blob/master/notes/pooling/SoftPool.md)|[github](https://github.com/alexandrosstergiou/SoftPool#)

### 4. Transformer
|pool|paper|conference|year|my notes|original github|
|:---:|:---:|:---:|:---:|:---:|:---:
| __The Sparse Transformer__|[Generating Long Sequences with Sparse Transformers](https://arxiv.org/pdf/1904.10509.pdf)|arxiv|2019| ... |[github](https://github.com/openai/sparse_attention)
| __Informer__|[Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436)|AAAI(best paper)|2021| ... |[github](https://github.com/zhouhaoyi/Informer2020)



There are some naive tests on different pooling method, you can see my another [repo](https://github.com/rentainhe/pytorch-pooling)
## Addition
如需转载，请标明出处