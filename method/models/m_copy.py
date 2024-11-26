"""
卷积自编码器（Auto-encoder）模型架构实现。
基于以下论文：
Marine cloud base height retrieval from MODIS cloud properties using supervised machine learning.

参考来源：
https://github.com/pytorch/vision
"""

import numpy as np
import torch
import torch.nn as nn
import io
from contextlib import redirect_stdout
from torchinfo import summary

# 卷积层定义
def conv3_3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """3x3 卷积"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1_1(in_channels, out_channels, stride=1):
    """1x1 卷积"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

def conv3_3_transpose(in_channels, out_channels, stride=2, groups=1):
    """3x3 转置卷积"""
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=stride, groups=groups)

def conv1_1_transpose(in_channels, out_channels, stride=1):
    """1x1 转置卷积"""
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

# 卷积块定义
class ConvBlock(nn.Module):
    """卷积块 => 下采样"""
    def __init__(self, inplanes, planes, stride=1, norm_layer=None):
        super(ConvBlock, self).__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        self.conv1 = conv3_3(inplanes, planes, stride)
        self.conv2 = conv3_3(planes, planes, stride)
        self.conv3 = conv3_3(planes, planes, stride)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = norm_layer(planes)

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.leakyrelu(self.bn(self.conv3(x)))
        x = self.maxpool(x)
        return x

class ConvTransposeBlock(nn.Module):
    """转置卷积块 => 上采样"""
    def __init__(self, inplanes, planes, stride=1, norm_layer=None):
        super(ConvTransposeBlock, self).__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        self.conv1 = conv3_3(inplanes, inplanes, stride)
        self.conv2 = conv3_3(inplanes, inplanes, stride)
        self.conv3 = conv3_3(inplanes, inplanes, stride)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.upsample = conv3_3_transpose(inplanes, planes, stride=2)
        self.bn = norm_layer(inplanes)

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.leakyrelu(self.bn(self.conv3(x)))
        x = self.upsample(x)
        return x

# 编码器网络
class Encoder(nn.Module):
    """编码器"""
    def __init__(self, n_channels=3, encoding_space=1024):
        super(Encoder, self).__init__()
        self.conv1 = conv3_3(n_channels, n_channels, stride=2)
        self.down1 = ConvBlock(n_channels, 6)
        self.down2 = ConvBlock(6, 32)
        self.down3 = ConvBlock(32, 64)
        self.down4 = ConvBlock(64, 128)
        self.down5 = ConvBlock(128, 256)
        self.down_max = nn.MaxPool2d(kernel_size=2) if encoding_space == 256 else None
        self.flat = nn.Flatten(start_dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        if self.down_max: x = self.down_max(x)
        return self.flat(x)

# 解码器网络
class Decoder(nn.Module):
    """解码器"""
    def __init__(self, n_channels=3, unflat_size=(256, 2, 2), encoding_space=1024):
        super(Decoder, self).__init__()
        self.unflat = nn.Unflatten(dim=1, unflattened_size=unflat_size)
        self.up_sample = conv3_3_transpose(256, 256, stride=2) if encoding_space == 256 else None
        self.conv1 = conv3_3_transpose(256, 256, stride=2)
        self.up1 = ConvTransposeBlock(256, 128)
        self.up2 = ConvTransposeBlock(128, 64)
        self.up3 = ConvTransposeBlock(64, 32)
        self.up4 = ConvTransposeBlock(32, 6)
        self.up5 = ConvTransposeBlock(6, n_channels)

    def forward(self, x):
        x = self.unflat(x)
        if self.up_sample: x = self.up_sample(x)
        x = self.conv1(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.up5(x)

# 卷积自编码器
class ConvAutoEncoder(nn.Module):
    """卷积自编码器模型"""
    def __init__(self, n_channels=3, input_grid_size=128, latent_dim=256):
        super(ConvAutoEncoder, self).__init__()
        self.encoder = Encoder(n_channels=n_channels)
        self.unflat_size = (256, 2, 2)
        self.latent_space = nn.Linear(np.prod(self.unflat_size), latent_dim)
        self.decoder_input = nn.Linear(latent_dim, np.prod(self.unflat_size))
        self.decoder = Decoder(n_channels=n_channels, unflat_size=self.unflat_size)
        self.final_actfunc = None

    def forward(self, x):
        encoded = self.encode(x)
        return encoded, self.decode(encoded)

    def encode(self, input):
        return self.latent_space(torch.flatten(self.encoder(input), start_dim=1))

    def decode(self, z):
        output = self.decoder(self.decoder_input(z))
        return self.final_actfunc(output) if self.final_actfunc else output