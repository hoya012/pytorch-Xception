# pytorch-Xception
Simple Code Implementation of ["Xception"](https://arxiv.org/abs/1610.02357) architecture using PyTorch.

![](https://github.com/hoya012/pytorch-Xception/blob/master/assets/xception.PNG)

For simplicity, i write codes in `ipynb`. So, you can easliy test my code.

*Last update : 2018/12/19*

## Contributor
* hoya012

## Requirements
Python 3.5
```
numpy
matplotlib
torch=1.0.0
torchvision
```

## Usage
You only run `Xception_pytorch.ipynb`.
For test, i used `CIFAR-10` Dataset and resize image scale from 32x32 to 299x299.
If you want to use own dataset, you can simply resize images.

## depthwise separable convolution impelemtation.
In Xception, there are many depthwise separable convolution operation. This is my simple implemenatation.

```
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
```

