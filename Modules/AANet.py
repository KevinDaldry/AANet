import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from torchsummary import summary
from utils import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet18(pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        if in_channels > 3:
            newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels - 3, :, :])

        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # for n, m in self.layer2.named_modules():
        #     if 'conv1' in n or 'downsample.0' in n:
        #         m.stride = (1, 1)
        # for n, m in self.layer3.named_modules():
        #     if 'conv1' in n or 'downsample.0' in n:
        #         m.stride = (1, 1)
        # for n, m in self.layer4.named_modules():
        #     if 'conv1' in n or 'downsample.0' in n:
        #         m.stride = (1, 1)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AANet(nn.Module):
    def __init__(self, in_channels=3, out_channel=2, inner_channel=128, stage=4):
        super(AANet, self).__init__()
        self.FCN = FCN(in_channels, pretrained=True)
        self.fusion = nn.ModuleList()
        self.up = nn.ModuleList()
        self.fpn = WeightRearrangementModule(inner_channel * stage, stage, 256, inner_channel)
        for i in range(stage):
            self.fusion.append(AmbiguityRefinementModule(2**(6+i), inner_channel, upsize=2**(8-i)))
            self.up.append(StdConv(inner_channel, inner_channel, kernel=1))
        self.out = StdConv(inner_channel, out_channel, kernel=1)

    def base_forward(self, x):

        x = self.FCN.layer0(x)
        x = self.FCN.maxpool(x)
        x1 = self.FCN.layer1(x)   # size:(64, 64, 64)
        x2 = self.FCN.layer2(x1)  # size:(128, 64, 64)
        x3 = self.FCN.layer3(x2)  # size:(256, 32, 32)
        x4 = self.FCN.layer4(x3)  # size:(512, 16, 16)

        return [x1, x2, x3, x4]

    def cd_forward(self, x1, x2):
        assert len(x1) == len(x2), f'stage cannot match'
        out = []
        for i in range(len(x1)):
            out.append(self.fusion[i](x1[i], x2[i]))
        return out

    def forward(self, x1, x2):
        x1_list = self.base_forward(x1)
        x2_list = self.base_forward(x2)
        out_list = self.cd_forward(x1_list, x2_list)

        out = self.fpn(out_list)
        out = self.out(out)
        return out


if __name__ == '__main__':
    model = SAGNet(3, 2, inner_channel=128).to('cuda')
    summary(model, input_size=[(3, 256, 256), (3, 256, 256)])
