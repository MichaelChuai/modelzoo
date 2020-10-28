import numpy as np
import torch
import torch.nn as nn

'''
Possible ops:
1.  none   
2.  1x3 & 3x1 conv
3.  1x7 & 7x1 conv   
4.  3x3 dconv
5.  3x3 avgpool
6.  3x3 maxpool
7.  1x1 conv
8.  3x3 conv
9.  3x3 sconv
10. 5x5 sconv
11. 7x7 sconv
'''

def ConvOp(in_channels, 
           out_channels,
           kernel_size,
           stride=1,
           padding=0,
           dilation=1,
           groups=1):
           return nn.Sequential(
               nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups),
               nn.BatchNorm2d(out_channels)
           )

class IdentityOp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IdentityOp, self).__init__()

    def forward(self, x):
        return x


class Conv1331Op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1331Op, self).__init__()
        self.conv1 = ConvOp(in_channels, out_channels, kernel_size=[1, 3], padding=[0, 1])
        self.conv2 = ConvOp(out_channels, out_channels, kernel_size=[3, 1], padding=[1, 0])

    def forward(self, x):
        out = self.conv1(x)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        return out

class Conv1771Op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1771Op, self).__init__()
        self.conv1 = ConvOp(in_channels, out_channels, kernel_size=[1, 7], padding=[0, 3])
        self.conv2 = ConvOp(out_channels, out_channels, kernel_size=[7, 1], padding=[3, 0])

    def forward(self, x):
        out = self.conv1(x)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        return out

class DilConvOp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilConvOp, self).__init__()
        self.dilconv = ConvOp(in_channels, out_channels, kernel_size=[3, 3], padding=[2, 2], dilation=2)
    
    def forward(self, x):
        return self.dilconv(x)

class AvgPoolOp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AvgPoolOp, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=[3, 3], stride=[1, 1], padding=[1, 1])
    
    def forward(self, x):
        return self.avgpool(x)

class MaxPoolOp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MaxPoolOp, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=[3, 3], stride=[1, 1], padding=[1, 1])
    
    def forward(self, x):
        return self.maxpool(x)

class Conv11Op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv11Op, self).__init__()
        self.conv = ConvOp(in_channels, out_channels, kernel_size=1, stride=1)
    
    def forward(self, x):
        return self.conv(x)

class Conv33Op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv33Op, self).__init__()
        self.conv = ConvOp(in_channels, out_channels, kernel_size=[3, 3], padding=[1, 1])
    
    def forward(self, x):
        return self.conv(x)

class SepConv33Op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SepConv33Op, self).__init__()
        self.sepconv = ConvOp(in_channels, out_channels, kernel_size=[3, 3], padding=[1, 1], groups=in_channels)
    
    def forward(self, x):
        return self.sepconv(x)

class SepConv55Op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SepConv55Op, self).__init__()
        self.sepconv = ConvOp(in_channels, out_channels, kernel_size=[5, 5], padding=[2, 2], groups=in_channels)
    
    def forward(self, x):
        return self.sepconv(x) 

class SepConv77Op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SepConv77Op, self).__init__()
        self.sepconv = ConvOp(in_channels, out_channels, kernel_size=[7, 7], padding=[3, 3], groups=in_channels)
    
    def forward(self, x):
        return self.sepconv(x)


class FactorizedReduction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FactorizedReduction, self).__init__()
        assert out_channels % 2 == 0, 'out_channels must be even.'
        self.fr_conv1 = nn.Conv2d(in_channels, out_channels//2, kernel_size=1, stride=2, padding=0)
        self.fr_conv2 = nn.Conv2d(in_channels, out_channels//2, kernel_size=1, stride=2, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = nn.ReLU()(x)
        fr_path1 = self.fr_conv1(x)
        fr_path2 = self.fr_conv2(x[:,:,1:,1:])
        fr = torch.cat([fr_path1, fr_path2], dim=1)
        out = self.bn(fr)
        return out
