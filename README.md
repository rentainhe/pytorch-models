# pytorch-models
A collection of `image-classification`, `detection`, `segmentation` models on Pytorch

## Computer Vision Models
There may be some differences between models for `cifar100` and `imagenet`

if you want to know the information of the network, here is an example:
```bash
$ python counter.py --dataset cifar100 --model resnet18
```
## Content
- [classic network]() (`resnet`, `resnext`... )
- [efficient network]()(`mobilenet`, `shufflenet`...)
- [weight network]()(`condconv`, `weightnet`...)

## Learning Notes
|model|paper|conference|year|my notes|
|:---:|:---:|:---:|:---:|:---:
|__Octave Conv__|[Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution](https://export.arxiv.org/pdf/1904.05049)|ICCV|2019|[notes]()

