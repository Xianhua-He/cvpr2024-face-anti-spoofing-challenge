'''MbileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from thop import profile

__all__ = ['dwspgnet16', 'dwspgnet25', 'dwspgnet30', 'dwspgnet46']

class HardSwish(nn.Module):
    def forward(self, x):
        return x * torch.clamp((x + 1) / 2, min=0, max=1)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, groups, kernel_size=3, stride=1, act="relu"):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, groups=groups, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

        self.act = None
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "prelu":
            self.act = nn.PReLU()
        elif act == "swish":
            self.act = Swish()
        elif act == "hardswish":
            self.act = HardSwish()

    def forward(self, x):
        out = self.bn(self.conv(x))
        if self.act != None:
            out = self.act(out)
        return out


class SqueezeExcitation(nn.Module):
    def __init__(self, inplanes, se_ratio=0.25):
        super(SqueezeExcitation, self).__init__()
        hidden_dim = int(inplanes * se_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=hidden_dim, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=inplanes, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = torch.sigmoid(out)
        return x * out


class DWSPGModule(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, use_se=False):
        super(DWSPGModule, self).__init__()
        self.use_se = use_se
        if self.use_se:
            self.se = SqueezeExcitation(out_planes * 2)

        self.conv_in = ConvX(in_planes, int(out_planes*0.5), groups=1, kernel_size=1, stride=1, act="relu")
        self.conv1 = ConvX(int(out_planes*0.5), int(out_planes*0.5), groups=int(out_planes*0.5), kernel_size=kernel, stride=stride, act="relu")
        self.conv2 = ConvX(int(out_planes*0.5), int(out_planes*0.5), groups=int(out_planes*0.5), kernel_size=kernel, stride=1, act="relu")
        self.conv_out = ConvX(int(out_planes*1.0), out_planes, groups=1, kernel_size=1, stride=1, act=None)

        self.act = nn.ReLU(inplace=True)
        # self.act = nn.PReLU()

        self.stride = stride
        self.skip = None
        if stride == 1 and in_planes != out_planes:
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes)
            )

        if stride == 2 and in_planes != out_planes:
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        skip = x
        out = self.conv_in(x)
        out1 = self.conv1(out)
        out2 = self.conv2(out1)
        out_cat = torch.cat((out1, out2), dim=1)

        if self.use_se:
            out_cat = self.se(out_cat)
        out = self.conv_out(out_cat)

        if self.skip is not None:
            skip = self.skip(skip)
        out += skip
        return self.act(out)


class DWSPGNet(nn.Module):
    def __init__(self, cfg_first, cfg, cfg_last, num_classes=1000):
        super(DWSPGNet, self).__init__()
        self.first_conv = nn.Sequential(
            ConvX(*cfg_first[0]),
            ConvX(*cfg_first[1])
        )
        self.layers = self._make_layers(cfg=cfg, in_planes=cfg_first[-1][1])
        self.conv_last = ConvX(*cfg_last)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(cfg_last[1], num_classes, bias=False)

        self.init_params()

    def init_params(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if name.endswith("conv_out.bn.weight"):
                    nn.init.constant_(m.weight, 0)
                else:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, in_planes):
        layers = []
        for out_planes, num_blocks, stride, kernel in cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(DWSPGModule(in_planes, out_planes, kernel, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.first_conv(x)
        out = self.layers(out)
        out = self.conv_last(out)
        out = self.gap(out).flatten(1)
        out = self.linear(out)
        return out

def dwspgnet16(num_classes):
    cfg_first = [(3, 6 , 1, 3, 2, "relu"),
                 (6, 12, 1, 3, 2, "relu")]

    cfg = [(24, 3, 2, 3),
           (48, 4, 2, 3),
           (96, 2, 2, 3)]

    cfg_last = (96, 96, 1, 1, 1, "relu")
    return DWSPGNet(cfg_first, cfg, cfg_last, num_classes)

def dwspgnet25(num_classes):
    cfg_first = [(3, 6 , 1, 3, 2, "relu"),
                 (6, 12, 1, 3, 2, "relu")]

    cfg = [(32 , 3, 2, 3),
           (64 , 6, 2, 3),
           (128, 2, 2, 3)]

    cfg_last = (128, 256, 1, 1, 1, "relu")
    return DWSPGNet(cfg_first, cfg, cfg_last, num_classes)


def dwspgnet30(num_classes):
    cfg_first = [(3, 6 , 1, 3, 2, "relu"),
                 (6, 12, 1, 3, 2, "relu")]

    cfg = [(32 , 4, 2, 3),
           (64 , 8, 2, 3),
           (128, 3, 2, 3)]

    cfg_last = (128, 256, 1, 1, 1, "relu")
    return DWSPGNet(cfg_first, cfg, cfg_last, num_classes)

def dwspgnet46(num_classes):
    cfg_first = [(3, 12, 1, 3, 2, "relu"),
                 (12, 24, 1, 3, 2, "relu")]

    cfg = [(48, 3, 2, 3),
           (96, 3, 2, 3),
           (192, 2, 2, 3)]

    cfg_last = (192, 192, 1, 1, 1, "relu")
    return DWSPGNet(cfg_first, cfg, cfg_last, num_classes)


if __name__ == '__main__':

    net = dwspgnet25(num_classes=480)
    x = torch.randn(1, 3, 224, 224)
    flops, params = profile(net, (x,))
    print('flops: ', flops, 'params: ', params)


