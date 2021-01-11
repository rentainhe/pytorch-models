## Octave Convolution
### Method
文章将自然图像分解为两部分——`低空间频率`部分和`高空间频率`部分
- 低空间频率: 描述图像平稳变化结构, 整体信息量较少
- 高空间频率: 描述图像细节, 整体信息量较大

将输入的`feature map`分解为两组，可以按照`channel`分解为高频部分(high)和低频部分(low), 然后分成`两路`处理, 将低频channel的空间分辨率减半, 分别进行`conv`操作, 两组频率特征会通过`down-sample`和`up-sample`进行交互.

### Overview
![](../../figs/octave_conv.png)
图中四组操作解析(从上至下):
- high: 高频channel
- low: 低频channel
1. high to high: `Conv2d`
2. high to low: `AvgPool-Conv2d`
3. low to high: `Conv2d-AvgPool`
4. low to low: `Conv2d`


### Implemented Article
- [Octave Learning notes](https://www.yuejianzun.xyz/2019/05/05/Octave%E5%8D%B7%E7%A7%AF%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/)