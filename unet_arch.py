import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from typing import Tuple


def conv_relu(n_filters: int, kernel_size: int = 3) -> nn.Sequential:
    """
    Single Conv->ReLU layer.

    :param n_filters: Number of filter for convolution
    :param kernel_size: Size of kernel.
    :return: Sequential object of conv->relu.
    """
    return nn.Sequential(
        nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, bias=False),  # stride=1, padding=0 by default
        nn.ReLU()
    )


class ConvBlock(nn.Module):
    """
    Convolution down/upsample blocks.
    """

    def __init__(self, n_filters: int, kernel_size: int = 3, scale_factor: int = 2, downsample: bool = True, crs: int = 2) -> None:
        """
        Instantiates ConvBlock object.

        :param n_filters: Number of input/output channels for convolution layer.
        :param kernel_size: Size of kernel.
        :param scale_factor: Stride of convolutions over features.
        :param downsample: Down sample if True.
        :param crs: Number of conv->relus.
        """
        super(ConvBlock, self).__init__()

        self.downsample = downsample
        self.scale_factor = scale_factor

        self.conv = nn.Sequential([conv_relu(n_filters, kernel_size) for _ in range(crs)])

    def forward(self, x: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if self.downsample:
            return nn.MaxPool2d(2, stride=self.scale_factor)(self.conv(x))
        else:
            return x[1] + nn.Upsample(scale_factor=self.scale_factor, mode='bilinear')(self.conv(x[0]))


class UNet(nn.Module):
    """
    Segmentation model.
    """

    def __init__(self, down_up: int, n_filters: int = 64, kernel_size: int = 3, scale_factor: int = 2) -> None:
        """
        Instantiates UNet model object.

        :param down_up: Number of down/upsamples.
        :param n_filters: Number of filters at input and output of network.
        :param kernel_size: Kernel size used by convolutions.
        :param scale_factor:Up/downsample factor.
        """
        super(UNet, self).__init__()

        self.down_up = down_up

        blocks = nn.Squential([ConvBlock(n_filters * (2 ** i), kernel_size, scale_factor) for i in range(self.down_up + 1)]
                              + [ConvBlock(n_filters * (2 ** i), kernel_size, scale_factor) for i in range(self.down_up, 0, -1)])

        self.blocks = nn.Squential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for UNet.

        :param x: Input image.
        :return: Network processed image.
        """
        outputs = []

        for i, block in enumerate(self.blocks):
            if i < self.down_up:
                x = block(x)
                outputs.append(x)
            else:
                x = block((x, outputs[::-1][i%self.down_up]))

        return x
