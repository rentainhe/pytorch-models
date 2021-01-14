## Octave Convolution
### Method
文章将自然图像分解为两部分——`低空间频率`部分和`高空间频率`部分
- 低空间频率: 描述图像平稳变化结构, 整体信息量较少
- 高空间频率: 描述图像细节, 整体信息量较大

将输入的`feature map`分解为两组，可以按照`channel`分解为高频部分(high)和低频部分(low), 然后分成`两路`处理, 将低频channel的空间分辨率减半, 分别进行`conv`操作, 两组频率特征会通过`down-sample`和`up-sample`进行交互.

看源码可知, 在 __整个网络之中__ 都存在两路特征, 不断地进行交互, 网络的第一层 ( __first OctConv__ ) 将特征进行分解, 中间层 ( __Middle OcvConv__ ) 用于处理两路特征并完成特征交互, 最后一层 ( __Last OctConv__ ) 用于将特征汇聚.

### Overview
![](../../figs/conv/octave_conv.png)
图中四组操作解析(从上至下):
- high: 高频channel
- low: 低频channel
1. high to high: `Conv2d`
2. high to low: `AvgPool-Conv2d`
3. low to high: `Conv2d-AvgPool`
4. low to low: `Conv2d`

### Code
#### 1. First-Octave-Conv
First `Octave Conv` layer, used in `the first layer` of the network ,change the input feature map into `high frequency sub-channels` and `low frequency sub-channels`, controlled by `alpha`

分为两个path:
- if stride > 2: input -> AvgPool2d
- 产生高频特征图: input -> Conv2d
- 产生低频特征图: input -> AvgPool2d -> Conv2d

```python
import torch
import torch.nn as nn
class FirstOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(FirstOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.avgpool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.high_to_low = torch.nn.Conv2d(in_channels, int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.high_to_high = torch.nn.Conv2d(in_channels, out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        # 通过avgpool来减少特征图大小
        if self.stride ==2:
            x = self.avgpool(x)

        X_high_to_low = self.avgpool(x)
        X_low = self.high_to_low(X_high_to_low)
        
        X_high = x
        X_high = self.high_to_high(X_high)


        return X_high, X_low
```

#### 2. Middle-Octave-Conv
中间层卷积, 低频和高频同时输入, 经过变换后再同时输出低频和高频特征, `低频高频分开处理`,`information changing between high-frequency and low-frequency`

Four paths:
- if stride == 2: input -> AvgPool2d
- `low-frequency` change into `low-frequency`: Conv2d
- `low-frequency` change into `high-frequency`: Conv2d -> Upsample
- `high-frequency` change into `low-frequency`: AvgPool2d -> Conv2d
- `high-frequency` change into `high-frequency`: Conv2d

```python
import torch
import torch.nn as nn
class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        kernel_size = kernel_size[0]
        self.avgpool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride

        # four Conv2d layer
        self.low_to_low = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.low_to_high = torch.nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.high_to_low = torch.nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.high_to_high = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        X_h, X_l = x  # input: [ high_frequency, low_frequency ]
        if self.stride ==2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)
        
        # low-frequency
        X_low_to_high = self.low_to_high(X_l)
        X_low_to_low = self.low_to_low(X_l)
        X_low_to_high = self.upsample(X_low_to_high)

        # high-frequency
        X_high_to_low = self.avgpool(X_h)
        X_high_to_high = self.high_to_high(X_h)
        X_high_to_low = self.high_to_low(X_high_to_low)
        
        # element-wise add output, combine both information
        X_h = X_low_to_high + X_high_to_high
        X_l = X_high_to_low + X_low_to_low

        return X_h, X_l
```

#### 3. Last-Octave-Conv
Combine two paths features into one output
- low-to-high: Conv2d -> Upsample
- high-to-high: Conv2d
```python
import torch
import torch.nn as nn
class LastOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(LastOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.avgpool = nn.AvgPool2d(kernel_size=(2,2), stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')

        self.low_to_high = torch.nn.Conv2d(int(alpha * in_channels), out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.high_to_high = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        

    def forward(self, x):
        X_h, X_l = x

        if self.stride ==2:
            X_h, X_l = self.avgpool(X_h), self.avgpool(X_l)
        
        X_low_to_high = self.low_to_high(X_l)
        X_high_to_high = self.high_to_high(X_h)
        X_low_to_high = self.upsample(X_low_to_high)
        
        X_h = X_high_to_high + X_low_to_high

        return X_h
```

### Implemented Article
- [Octave Learning notes](https://www.yuejianzun.xyz/2019/05/05/Octave%E5%8D%B7%E7%A7%AF%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/)
- [Octave Convolution details](https://www.cnblogs.com/fydeblog/p/11655076.html)