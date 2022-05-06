# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from module.myTool import Resnet
import torchvision.models as models
import torch.nn.functional as F
from module.attentionBlock import AttentionCha, AttentionSpa


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class GAP(nn.Module):
    def  __init__(self, in_ch, out_ch, k_size=3):
        super(GAP, self).__init__()
        self.deConv = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = BasicConv(in_ch, out_ch, 1, 1, 0, 1)
        self.relu = nn.ReLU(True)
        self.atten_spa = AttentionSpa()
        self.atten_cha = AttentionCha(in_ch, out_ch)


    def forward(self, x):
        x2 = self.atten_spa(x)
        x3 = self.atten_cha(x)
        out = self.relu(x + x2 + x3)
        return out


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class CDC_1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CDC_1, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv1x1 = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        res = self.conv1x1(x)
        x0 = self.branch0(x)
        x1 = self.branch1(x0)
        x2 = self.branch2(x1)
        x3 = self.branch3(x2)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + res)
        return x

class CDC_2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CDC_2, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )

        self.conv_cat = BasicConv2d(3*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        res = self.conv_res(x)
        x0 = self.branch0(x)
        x1 = self.branch1(x0)
        x2 = self.branch2(x1)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2), 1))

        x = self.relu(x_cat + res)
        return x


class PDA(nn.Module):
    def __init__(self, channels_high, channels_low, kernel_size=3, upsample=True):
        super(PDA, self).__init__()
        self.deConv = nn.ConvTranspose2d(channels_high, channels_high, 2, stride=2)
        self.gap = GAP(channels_high // 2, channels_high)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(channels_high, channels_low)
        self.conv2 = conv3x3(channels_high, channels_low)


    def forward(self, fms_high, fms_low):
        x1 = self.deConv(fms_high)
        x1 = self.conv2(x1)
        x2 = self.gap(fms_low)
        x = torch.cat((x1, x2), 1)
        x = self.relu(x)
        out = self.conv1(x)
        return out

class AMP(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(AMP, self).__init__()
        self.pol1 = nn.AvgPool2d(kernel_size=2, stride=2,
                             ceil_mode=True, count_include_pad=False)
        self.pol2 = nn.AvgPool2d(kernel_size=3, stride=2,
                             ceil_mode=True, count_include_pad=False)
        self.pol3 = nn.AvgPool2d(kernel_size=5, stride=2, padding=1,
                                 ceil_mode=True, count_include_pad=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv = conv1x1(channel_in * 3 + channel_out, channel_out)


    def forward(self, x, other):
        # x = self.conv(x)
        x1 = self.pol1(x)
        x2 = self.pol2(x)
        x3 = self.pol3(x)
        out = torch.cat((x1, x2, x3, other), 1)
        out = self.conv(out)
        out = self.relu(out)
        return out


class MyNet(nn.Module):
    def __init__(self, n_class=1):
        super(MyNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = Resnet()
        # ---- Receptive Field Block like module ----

        self.CDC1 = CDC_1(256, 128)
        self.CDC2 = CDC_1(512, 256)
        self.CDC3 = CDC_1(1024, 512)
        self.CDC4 = CDC_2(2048, 1024)

        bottom_ch = 1024
        self.PDA3 = PDA(bottom_ch, 512)
        self.PDA2 = PDA(bottom_ch // 2, 256)
        self.PDA1 = PDA(bottom_ch // 4, 128)

        self.conv1_1 = conv1x1(128, 1)
        self.conv1_2 = conv1x1(256, 1)
        self.conv1_3 = conv1x1(512, 1)
        self.conv1_4 = conv1x1(1024, 1)

        self.pol1 = AMP(64, 256)
        self.pol2 = AMP(256, 512)
        self.pol3 = AMP(512, 1024)


        if self.training:
            self.initialize_weights()
            print('initialize_weights')


    def forward(self, x):
        x = self.resnet.conv1(x)        # 64, 176, 176
        pol = x
        x = self.resnet.bn1(x)

        x = self.resnet.relu(x)

        # ---- low-level features ----
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        x1 = self.pol1(pol, x1)
        pol = x1

        # ---- high-level features ----
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
        x2 = self.pol2(pol, x2)
        pol = x2
        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x3 = self.pol3(pol, x3)
        x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11

        x1_CDC = self.CDC1(x1)        # 256 -> 128
        x2_CDC = self.CDC2(x2)        # 512 -> 256
        x3_CDC = self.CDC3(x3)        # 1024 -> 512
        x4_CDC = self.CDC4(x4)        # 2048 -> 1024

        x3 = self.PDA3(x4_CDC, x3_CDC)  # 1/16
        x2 = self.PDA2(x3, x2_CDC)  # 1/8
        x1 = self.PDA1(x2, x1_CDC)  # 1/4

        map_1 = self.conv1_1(x1)
        map_2 = self.conv1_2(x2)
        map_3 = self.conv1_3(x3)
        map_4 = self.conv1_4(x4_CDC)

        out = F.interpolate(map_1, scale_factor=4, mode='bilinear')

        return out

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True) #If True, the pre-trained resnet50 will be loaded.
        pretrained_dict = res50.state_dict()
        model_dict = self.resnet.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.resnet.load_state_dict(model_dict)



