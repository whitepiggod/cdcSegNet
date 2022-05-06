"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math
from torch.autograd import Variable
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
import numpy as np
import cv2
import torchvision
# from unet import Unet
from module.other_model.SIM import AIM, SIM

#from unet import Unet



def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
            )

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
            )




class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=False):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:

            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

            )
            self.conv = conv3x3_bn_relu(middle_channels, out_channels)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv2x2 = nn.Sequential(
                                nn.Conv2d(in_channels, in_channels//2, kernel_size=2, stride=1),
                                nn.ReLU(inplace=True)
            )
            self.block = nn.Sequential(
                conv3x3_bn_relu(middle_channels, middle_channels),
                conv3x3_bn_relu(middle_channels, out_channels),
            )
        ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            #elif isinstance(m, SynchronizedBatchNorm2d):
            #    m.weight.data.fill_(1)
            #    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, y):

        x = self.up(x)
        x = self.conv2x2(F.pad(x,(0,1,0,1)))
        x = torch.cat([x, y], dim=1)
        out = self.block(x)
        return out



class Downsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(y, 2, stride=2)

        return x, y




class Unet(nn.Module):
    def __init__(self, num_classes=1):
        super(Unet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        nb_filter = [64, 128, 256, 512, 1024]
        self.conv0_0 = VGGBlock(3, nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4])
        self.up4 = DecoderBlock(1024, 512 + 512, 512)
        self.up3 = DecoderBlock(512, 256 + 256, 256)
        self.up2 = DecoderBlock(256, 128 + 128, 128)
        self.up1 = DecoderBlock(128, 64 + 64, 64)
        self.conv = nn.Conv2d(64, 2, 3, padding=1)
        self.outconv = nn.Conv2d(2, num_classes, 1)


    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x = self.up4(x4_0, x3_0)
        x = self.up3(x, x2_0)
        x = self.up2(x, x1_0)
        x = self.up1(x, x0_0)
        x = self.conv(x)
        x1 = self.outconv(x)

        return x1

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGGBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, y):
        input = torch.cat([x,y], dim=1)
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out



class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=5):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class eca_VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(eca_VGGBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.eca = eca_layer(out_channels)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)
        out = self.relu(out)

        return out

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.double_conv(x)
        out = self.relu(out + identity)
        return out

class AtrousConv(nn.Module):
    def __init__(self, middle_channels, out_channels):
        super(AtrousConv, self).__init__()
        # self.conv3x3 = conv3x3_bn_relu(in_channels, middle_channels)
        self.conv1 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=2, stride=1, dilation=2)
        self.conv3 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=3, stride=1, dilation=3)
        self.eca = eca_layer(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = self.conv3x3(x)
        # x = torch.cat([x, y], dim=1)
        out1 = self.conv1(x)
        out1 = self.eca(out1)
        out2 = self.conv2(x)
        out2 = self.eca(out2)
        out3 = self.conv3(x)
        out3 = self.eca(out3)
        out =  1/3 * out1 + 1/3 * out2 + 1/3 * out3
        out = self.bn(out)
        out = self.relu(out)

        return out

class AtrousConv1(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(AtrousConv1, self).__init__()
        self.conv3x3 = conv3x3_bn_relu(in_channels, middle_channels)
        self.conv1x1 = nn.Conv2d(middle_channels, middle_channels//4, 1, 1)
        self.conv1 = nn.Conv2d(middle_channels//4, middle_channels//4, kernel_size=3, padding=1, stride=1, dilation=1)
        self.conv2 = nn.Conv2d(middle_channels//4, middle_channels//4, kernel_size=3, padding=2, stride=1, dilation=2)
        self.conv3 = nn.Conv2d(middle_channels//4, middle_channels//4, kernel_size=3, padding=3, stride=1, dilation=3)
        self.conv4 = nn.Conv2d(middle_channels//4, middle_channels//4, kernel_size=3, padding=4, stride=1, dilation=4)
        self.eca = eca_layer(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv3x3(x)
        x = self.conv1x1(x1)
        out1 = self.conv1(x)
        out1 = self.eca(out1)
        out2 = self.conv2(x)
        out2 = self.eca(out2)
        out3 = self.conv3(x)
        out3 = self.eca(out3)
        out4 = self.conv3(x)
        out4 = self.eca(out4)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        out = out + x1
        return out

class AtrousConv2(nn.Module):
    def __init__(self, middle_channels, out_channels):
        super(AtrousConv2, self).__init__()
        self.conv1 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=2, stride=1, dilation=2)
        self.conv3 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=3, stride=1, dilation=3)
        self.eca = eca_layer(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.eca(out1)
        out2 = self.conv2(x)
        out2 = self.eca(out2)
        out3 = self.conv3(x)
        out3 = self.eca(out3)
        out =  1/3 * out1 + 1/3 * out2 + 1/3 * out3
        out = self.bn(out)
        out = self.relu(out)
        return out

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
    def forward(self, x1, x2):
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1


class AIMUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super(AIMUNet, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv0_0 = AtrousConv(3, 32)
        self.conv1_0 = AtrousConv(32, 64)
        self.conv2_0 = AtrousConv(64, 128)
        self.conv3_0 = AtrousConv(128, 256)
        self.conv4_0 = AtrousConv(256, 512)
        # nb_filter = [64, 128, 256, 512, 1024]
        # self.conv0_0 = ResBlock(3, nb_filter[0])
        # self.conv1_0 = ResBlock(nb_filter[0], nb_filter[1])
        # self.conv2_0 = ResBlock(nb_filter[1], nb_filter[2])
        # self.conv3_0 = ResBlock(nb_filter[2], nb_filter[3])
        # self.conv4_0 = ResBlock(nb_filter[3], nb_filter[4])
        #self.conv4 = DoubleConv(2048, 1024)

        # nb_filter = [64, 128, 256, 512, 1024]
        # self.conv0_0 = ResBlock(input_channels, nb_filter[0])#ResBlock
        # self.conv1_0 = ResBlock(nb_filter[0], nb_filter[1])
        # self.conv2_0 = ResBlock(nb_filter[1], nb_filter[2])
        # self.conv3_0 = ResBlock(nb_filter[2], nb_filter[3])
        # self.conv4_0 = ResBlock(nb_filter[3], nb_filter[4])

        # MID
        self.trans = AIM(iC_list=(32, 64, 128, 256, 512), oC_list=(64, 64, 64, 64, 64))
        # self.conv4 = SIM(1024, 512)
        self.conv3_1 = VGGBlock(64+64, 64)
        self.conv2_2 = VGGBlock(64+64, 64)
        self.conv1_3 = VGGBlock(64+64, 64)
        self.conv0_4 = VGGBlock(64+64, 64)
        self.final = nn.Sequential(
                        nn.Conv2d(64, num_classes, kernel_size=1)
                    )

    def forward(self, input):
        x_size = input.size()

        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        # x2_0 = self.conv2_0(x1_0)
        # x3_0 = self.conv3_0(x2_0)
        # x4_0 = self.conv4_0(x3_0)
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        #x4_0 = self.conv4(x4_0)

        #MID
        mid1, mid2, mid3, mid4, mid5 = self.trans(
            x0_0, x1_0, x2_0, x3_0, x4_0
        )


        # x4 = self.conv4(mid5)
        x3_1 = self.conv3_1(self.up(mid5), mid4)
        x2_2 = self.conv2_2(self.up(x3_1), mid3)
        x1_3 = self.conv1_3(self.up(x2_2), mid2)
        x0_4 = self.conv0_4(self.up(x1_3), mid1)
        # x4 = self.conv4(x4_0)
        # x3_1 = self.conv3_1(self.up(x4)+x3_0)
        # x2_2 = self.conv2_2(self.up(x3_1)+x2_0)
        # x1_3 = self.conv1_3(self.up(x2_2)+x1_0)
        # x0_4 = self.conv0_4(self.up(x1_3)+x0_0)
        #last = self.last(self.up(x0_4))
        output = self.final(x0_4)


        return output




class ModelBuilder():
    # custom weights initialization

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)


    def build_unet(self, num_class=1, arch='edgeunet', weights=''):
        arch = arch.lower()

        if arch == 'edgeunet':
            unet = AIMUNet(num_classes=num_class)
        else:
            raise Exception('Architecture undefined!')



        return unet


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = Variable(torch.randn((1, 3, 256, 256))).cuda()
    net = edgeUNet().cuda()
    #print(net)
    out = net(img)
