import torch
from torch import nn


class block(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=None, stride=1):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.downsample = downsample
        self.stride = stride

        self.conv1 = nn.Conv2d(
            in_channel, out_channel, kernel_size=1, stride=1, padding=0
        )
        # 这个层的输出形状不变
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channel, out_channel, kernel_size=3, stride=stride, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        # conv3的输出形状不变，负责升维
        self.conv3 = nn.Conv2d(
            out_channel, out_channel * 4, kernel_size=1, stride=1, padding=0
        )
        self.bn3 = nn.BatchNorm2d(out_channel * 4)
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = identity + x
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, images_channels, num_classes):
        super().__init__()
        self.in_channel = 64
        self.conv1 = nn.Conv2d(images_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def __make_layer(self, block, num_residual_blocks, out_channels, stride):
        downsample = None
        if stride != 1 or self.in_channel != out_channels * 4:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channel, out_channels * 4, kernel_size=1, stride=stride
                ),
                nn.BatchNorm2d(out_channels * 4),
            )

        layers = []
        layers.append(
            block(self.in_channel, out_channels, downsample=downsample, stride=stride)
        )

        self.in_channel = out_channels * 4

        for i in range(1, num_residual_blocks):
            layers.append(block(self.in_channel, out_channels))

        return nn.Sequential(*layers)
