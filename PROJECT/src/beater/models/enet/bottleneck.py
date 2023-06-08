from __future__ import annotations
from abc import abstractmethod

import torch.nn as nn

class Bottleneck(nn.Module):
    """
    Based on paper: https://arxiv.org/pdf/1606.02147.pdf
    Bottleneck module of the model ENet
    """
    def __init__(self) -> None:
        super(Bottleneck, self).__init__()

    @abstractmethod
    def forward(self):
        pass

class Regular(Bottleneck):
    """
    Based on paper: https://arxiv.org/pdf/1606.02147.pdf
    Regular bottleneck module of the model ENet
    """
    def __init__(self,
                 channels,
                 internal_ratio: int = 4,
                 kernel_size: int = 3,
                 padding: int = 0,
                 dropout_prob: float = 0.1,) -> None:
        super(Regular, self).__init__()

        if internal_ratio <= 1 and internal_ratio > channels:
            raise RuntimeError("Internal ratio must be a positive integer.")

        internal_channels = channels // internal_ratio

        # Using ReLU activation by default
        self.activation = nn.ReLU()

        # Main branch: shortcut connection

        # Ext branch: main connection
        # 1x1 projection convolution
        self.conv_proj = nn.Sequential(
            nn.Conv2d(channels,
                      internal_channels,
                      kernel_size = 1,
                      stride = 1,),
            nn.BatchNorm2d(internal_channels),
            self.activation,)

        # main conv
        self.conv_main = nn.Sequential(
            nn.Conv2d(internal_channels,
                      internal_channels,
                      kernel_size = kernel_size,
                      stride = 1,
                      padding = padding,
                      dilation = 1,),
            nn.BatchNorm2d(internal_channels),
            self.activation,)

        # 1x1 expansion convolution
        self.conv_exp = nn.Sequential(
            nn.Conv2d(internal_channels,
                      channels,
                      kernel_size = 1,
                      stride = 1,),
            nn.BatchNorm2d(channels),
            self.activation,)

        # Regularizer
        self.dropout = nn.Dropout2d(p = dropout_prob)

    def forward(self, x):
        # Main branch: shortcut connection
        main = x

        # Ext branch: main connection
        # 1x1 projection convolution
        ext = self.conv_proj(x)
        # main conv
        ext = self.conv_main(ext)
        # 1x1 expansion convolution
        ext = self.conv_exp(ext)
        # Regularizer
        ext = self.dropout(ext)

        # Merge
        out = main + ext

        # Activation
        out = self.activation(out)

        return out

class Dilated(Bottleneck):
    """
    Based on paper: https://arxiv.org/pdf/1606.02147.pdf
    Dilated bottleneck module of the model ENet
    """
    def __init__(self,
                 channels,
                 internal_ratio: int = 4,
                 kernel_size: int = 3,
                 padding: int = 0,
                 dilation: int = 1,
                 dropout_prob: float = 0.1,) -> None:
        super(Dilated, self).__init__()

        if internal_ratio <= 1 and internal_ratio > channels:
            raise RuntimeError("Internal ratio must be a positive integer.")

        internal_channels = channels // internal_ratio

        # Using ReLU activation by default
        self.activation = nn.ReLU()

        # Main branch: shortcut connection

        # Ext branch: main connection
        # 1x1 projection convolution
        self.conv_proj = nn.Sequential(
            nn.Conv2d(channels,
                      internal_channels,
                      kernel_size = 1,
                      stride = 1,),
            nn.BatchNorm2d(internal_channels),
            self.activation,)

        # main conv
        self.conv_main = nn.Sequential(
            nn.Conv2d(internal_channels,
                      internal_channels,
                      kernel_size = kernel_size,
                      stride = 1,
                      padding = padding,
                      dilation = dilation,),
            nn.BatchNorm2d(internal_channels),
            self.activation,)

        # 1x1 expansion convolution
        self.conv_exp = nn.Sequential(
            nn.Conv2d(internal_channels,
                      channels,
                      kernel_size = 1,
                      stride = 1,),
            nn.BatchNorm2d(channels),
            self.activation,)

        # Regularizer
        self.dropout = nn.Dropout2d(p = dropout_prob)

    def forward(self, x):
        # Main branch: shortcut connection
        main = x

        # Ext branch: main connection
        # 1x1 projection convolution
        ext = self.conv_proj(x)
        # main conv
        ext = self.conv_main(ext)
        # 1x1 expansion convolution
        ext = self.conv_exp(ext)
        # Regularizer
        ext = self.dropout(ext)

        # Merge
        out = main + ext

        # Activation
        out = self.activation(out)

        return out
