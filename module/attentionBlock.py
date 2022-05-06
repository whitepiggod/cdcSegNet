import torch.nn as nn
import torch


class BasicConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class AttentionCha(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3):
        super(AttentionCha, self).__init__()
        self.GAvgPol = nn.AdaptiveAvgPool2d(1)
        self.conv = BasicConv(in_ch, out_ch, 1, 1, 0, 1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.GAvgPol(x)
        x1 = self.conv1(x1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        x1 = self.sigmoid(x1)
        out = x * x1.expand_as(x1)
        return out


class AttentionSpa1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(AttentionSpa, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
        self.softmax = nn.Sigmoid()

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        print(proj_query.size(), '------', proj_key.size())
        energy = torch.bmm(proj_query, proj_key)  # 矩阵乘法
        attention = self.softmax(energy)  # 添加非线性函数
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)  # reshape到原图
        return out


class AttentionSpa(nn.Module):
    def __init__(self):
        super(AttentionSpa, self).__init__()
        self.softmax = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attetion = self.softmax(max_out)
        out = x * attetion
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)