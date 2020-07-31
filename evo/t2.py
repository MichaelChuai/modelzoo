import torch
import torch.nn as nn

class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.pad2 = nn.ZeroPad2d((0, 1, 0, 1))
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        z1 = self.conv1(x)
        z2 = self.conv2(x[:, :, 1:, 1:])
        out = torch.cat([z1, z2], dim=1)
        out = self.bn(out)
        return out


a = torch.rand(5, 10, 6, 6)
l = FactorizedReduce(10, 20)

b = l(a)