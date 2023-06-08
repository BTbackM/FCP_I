from __future__ import annotations
from abc import abstractmethod

import torch
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
        self.ext_conv_proj = nn.Sequential(
            nn.Conv2d(channels,
                      internal_channels,
                      kernel_size = 1,
                      stride = 1,),
            nn.BatchNorm2d(internal_channels),
            self.activation,)

        # main conv
        self.ext_conv_main = nn.Sequential(
            nn.Conv2d(internal_channels,
                      internal_channels,
                      kernel_size = kernel_size,
                      stride = 1,
                      padding = padding,
                      dilation = 1,),
            nn.BatchNorm2d(internal_channels),
            self.activation,)

        # 1x1 expansion convolution
        self.ext_conv_exp = nn.Sequential(
            nn.Conv2d(internal_channels,
                      channels,
                      kernel_size = 1,
                      stride = 1,),
            nn.BatchNorm2d(channels),
            self.activation,)

        # Regularizer
        self.ext_dropout = nn.Dropout2d(p = dropout_prob)

    def forward(self, x):
        # Main branch: shortcut connection
        main = x

        # Ext branch: main connection
        # 1x1 projection convolution
        ext = self.ext_conv_proj(x)
        # main conv
        ext = self.ext_conv_main(ext)
        # 1x1 expansion convolution
        ext = self.ext_conv_exp(ext)
        # Regularizer
        ext = self.ext_dropout(ext)

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
        self.ext_conv_proj = nn.Sequential(
            nn.Conv2d(channels,
                      internal_channels,
                      kernel_size = 1,
                      stride = 1,),
            nn.BatchNorm2d(internal_channels),
            self.activation,)

        # main conv
        self.ext_conv_main = nn.Sequential(
            nn.Conv2d(internal_channels,
                      internal_channels,
                      kernel_size = kernel_size,
                      stride = 1,
                      padding = padding,
                      dilation = dilation,),
            nn.BatchNorm2d(internal_channels),
            self.activation,)

        # 1x1 expansion convolution
        self.ext_conv_exp = nn.Sequential(
            nn.Conv2d(internal_channels,
                      channels,
                      kernel_size = 1,
                      stride = 1,),
            nn.BatchNorm2d(channels),
            self.activation,)

        # Regularizer
        self.ext_dropout = nn.Dropout2d(p = dropout_prob)

    def forward(self, x):
        # Main branch: shortcut connection
        main = x

        # Ext branch: main connection
        # 1x1 projection convolution
        ext = self.ext_conv_proj(x)
        # main conv
        ext = self.ext_conv_main(ext)
        # 1x1 expansion convolution
        ext = self.ext_conv_exp(ext)
        # Regularizer
        ext = self.ext_dropout(ext)

        # Merge
        out = main + ext

        # Activation
        out = self.activation(out)

        return out

class Asymmetric(Bottleneck):
    """
    Based on paper: https://arxiv.org/pdf/1606.02147.pdf
    Asymmetric bottleneck module of the model ENet
    """
    def __init__(self,
                 channels,
                 internal_ratio: int = 4,
                 kernel_size: int = 3,
                 padding: int = 0,
                 dropout_prob: float = 0.1,) -> None:
        super(Asymmetric, self).__init__()

        if internal_ratio <= 1 and internal_ratio > channels:
            raise RuntimeError("Internal ratio must be a positive integer.")

        internal_channels = channels // internal_ratio

        # Using ReLU activation by default
        self.activation = nn.ReLU()

        # Main branch: shortcut connection

        # Ext branch: main connection
        # 1x1 projection convolution
        self.ext_conv_proj = nn.Sequential(
            nn.Conv2d(channels,
                      internal_channels,
                      kernel_size = 1,
                      stride = 1,),
            nn.BatchNorm2d(internal_channels),
            self.activation,)

        # main conv: asymmetric convolution
        # divide into two convolutions:
        # 5x1 asymmetric convolution
        # 1x5 asymmetric convolution
        self.ext_conv_main = nn.Sequential(
            nn.Conv2d(internal_channels,
                      internal_channels,
                      kernel_size = (kernel_size, 1),
                      stride = 1,
                      padding = (padding, 0),
                      dilation = 1,),
            nn.BatchNorm2d(internal_channels),
            self.activation,
            nn.Conv2d(internal_channels,
                      internal_channels,
                      kernel_size = (1, kernel_size),
                      stride = 1,
                      padding = (0, padding),
                      dilation = 1,),
            nn.BatchNorm2d(internal_channels),
            self.activation,)

        # 1x1 expansion convolution
        self.ext_conv_exp = nn.Sequential(
            nn.Conv2d(internal_channels,
                      channels,
                      kernel_size = 1,
                      stride = 1,),
            nn.BatchNorm2d(channels),
            self.activation,)

        # Regularizer
        self.ext_dropout = nn.Dropout2d(p = dropout_prob)

    def forward(self, x):
        # Main branch: shortcut connection
        main = x

        # Ext branch: main connection
        # 1x1 projection convolution
        ext = self.ext_conv_proj(x)
        # main conv: asymmetric convolution
        ext = self.ext_conv_main(ext)
        # 1x1 expansion convolution
        ext = self.ext_conv_exp(ext)
        # Regularizer
        ext = self.ext_dropout(ext)

        # Merge
        out = main + ext

        # Activation
        out = self.activation(out)

        return out

class Downsampling(Bottleneck):
    """
    Based on paper: https://arxiv.org/pdf/1606.02147.pdf
    Downsampling bottleneck module of the model ENet
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio: int = 4,
                 return_indices: bool = False,
                 dropout_prob: float = 0.1,) -> None:
        super(Downsampling, self).__init__()

        self.return_indices = return_indices

        if internal_ratio <= 1 and internal_ratio > in_channels:
            raise RuntimeError("Internal ratio must be a positive integer.")

        internal_channels = in_channels // internal_ratio

        # Using ReLU activation by default
        self.activation = nn.ReLU()

        # Main branch: shortcut connection
        # Max pooling
        self.main_max_pool = nn.MaxPool2d(kernel_size = 2,
                                     stride = 2,
                                     return_indices = self.return_indices,)

        # Ext branch: main connection
        # 2x2 projection convolution with stride 2
        self.ext_conv_proj = nn.Sequential(
            nn.Conv2d(in_channels,
                      internal_channels,
                      kernel_size = 2,
                      stride = 2,),
            nn.BatchNorm2d(internal_channels),
            self.activation,)

        # main conv
        self.ext_conv_main = nn.Sequential(
            nn.Conv2d(internal_channels,
                      internal_channels,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1,),
            nn.BatchNorm2d(internal_channels),
            self.activation,)

        # 1x1 expansion convolution
        self.ext_conv_exp = nn.Sequential(
            nn.Conv2d(internal_channels,
                      out_channels,
                      kernel_size = 1,
                      stride = 1,),
            nn.BatchNorm2d(out_channels),
            self.activation,)

        # Regularizer
        self.ext_dropout = nn.Dropout2d(p = dropout_prob)

    def forward(self, x):
        # Main branch: shortcut connection
        main, max_indices = self.main_max_pool(x)

        # Ext branch: main connection
        # 2x2 projection convolution with stride 2
        ext = self.ext_conv_proj(x)
        # main conv
        ext = self.ext_conv_main(ext)
        # 1x1 expansion convolution
        ext = self.ext_conv_exp(ext)
        # Regularizer
        ext = self.ext_dropout(ext)

        # Main branch channel padding
        n, ch_ext, h_ext, w_ext = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h_ext, w_ext)

        if main.is_cuda:
            padding = padding.cuda()

        main = torch.cat((main, padding), 1)

        # Merge
        out = main + ext

        # Activation
        out = self.activation(out)

        if self.return_indices:
            return out, max_indices
        else:
            return out

class Upsampling(Bottleneck):
    """
    Based on paper: https://arxiv.org/pdf/1606.02147.pdf
    Upsampling bottleneck module of the model ENet
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio: int = 4,
                 dropout_prob: float = 0.1,) -> None:
        super(Upsampling, self).__init__()

        if internal_ratio <= 1 and internal_ratio > in_channels:
            raise RuntimeError("Internal ratio must be a positive integer.")

        internal_channels = in_channels // internal_ratio

        # Using ReLU activation by default
        self.activation = nn.ReLU()

        # Main branch: shortcut connection
        # Convolutional
        self.main_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size = 1,),
            nn.BatchNorm2d(out_channels),)

        self.main_max_unpool = nn.MaxUnpool2d(kernel_size = 2,)

        # Ext branch: main connection
        # 1x1 projection convolution
        self.ext_conv_proj = nn.Sequential(
            nn.Conv2d(in_channels,
                      internal_channels,
                      kernel_size = 1,),
            nn.BatchNorm2d(internal_channels),
            self.activation,)

        # main conv: Transposed convolution
        self.ext_trans = nn.ConvTranspose2d(internal_channels,
                                                internal_channels,
                                                kernel_size = 2,
                                                stride = 2,)
        self.ext_trans_bnorm = nn.BatchNorm2d(internal_channels)
        self.ext_trans_activation = self.activation

        # 1x1 expansion convolution
        self.ext_conv_exp = nn.Sequential(
            nn.Conv2d(internal_channels,
                      out_channels,
                      kernel_size = 1,),
            nn.BatchNorm2d(out_channels),)

        # Regularizer
        self.ext_dropout = nn.Dropout2d(p = dropout_prob)

    def forward(self, x, max_indices, output_size):
        # Main branch: shortcut connection
        main = self.main_conv(x)
        main = self.main_max_unpool(main,
                                    max_indices,
                                    output_size = output_size)

        # Ext branch: main connection
        # 1x1 projection convolution
        ext = self.ext_conv_proj(x)
        # main conv: Transposed convolution
        ext = self.ext_trans(ext,
                             output_size = output_size)
        ext = self.ext_trans_bnorm(ext)
        ext = self.ext_trans_activation(ext)
        # 1x1 expansion convolution
        ext = self.ext_conv_exp(ext)
        # Regularizer
        ext = self.ext_dropout(ext)

        # Merge
        out = main + ext

        # Activation
        out = self.activation(out)

        return out
