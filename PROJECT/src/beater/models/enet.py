import torch
import torch.nn as nn
import torch.nn.functional as F

class InitialBlock(nn.Module):
    """
    Based on paper: https://arxiv.org/pdf/1606.02147.pdf
    Initial block of the model ENet
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ) -> None:
        super(InitialBlock, self).__init__()

        # Main branch (left side): performs regular convolution
        self.main_branch = nn.Conv2d(in_channels = in_channels,
                                     out_channels = out_channels - 3,
                                     kernel_size = 3,
                                     stride = 2,
                                     padding = 1,)

        # Ext branch (right side): performs max pooling
        self.ext_branch = nn.MaxPool2d(kernel_size = 3,
                                       stride = 2,
                                       padding = 1,)

        # Batch normalization after concatenation
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # Using ReLU activation by default
        self.activation = nn.ReLU()

    def forward(self, x):
        main_branch = self.main_branch(x)
        ext_branch = self.ext_branch(x)
        # Concatenate the two branches as seen in Figure 2
        out = torch.cat((main_branch, ext_branch), dim = 1)
        out = self.batch_norm(out)
        out = self.activation(out)

        return out

class Bottleneck(nn.Module):
    """
    Based on paper: https://arxiv.org/pdf/1606.02147.pdf
    Bottleneck module of the model ENet
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilated = False,
                 asymmetric = False,
                 downsample = False,
                 upsample = False,
                 p = 0.01,
                 ) -> None:
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.upsample = upsample
        # Dilated bottleneck params
        if dilated:
            dilation = 2
            padding = 2
            stride = 1
        # Asymmetric bottleneck params
        elif asymmetric:
            dilation = 1
            padding = 4
            stride = 1
        # Upsample bottleneck params
        elif upsample:
            dilation = 1
            padding = 0
            stride = 1
        # Downsample bottleneck params
        else:
            dilation = 1
            padding = 1
            stride = 2

        # Conv 1x1 projection layer
        self.bottleneck_projection = nn.Sequential(
            nn.Conv2d(in_channels = in_channels,
                      out_channels = out_channels,
                      kernel_size = 1,
                      stride = 1,
                      padding = 0,),
            nn.BatchNorm2d(out_channels),)

        # Main conv layer
        self.bottleneck_main = nn.Sequential(
            nn.Conv2d(in_channels = out_channels,
                      out_channels = out_channels,
                      kernel_size = 3,
                      stride = stride,
                      padding = padding,
                      dilation = dilation,),
            nn.BatchNorm2d(out_channels),)

        # Conv 1x1 expansion layer
        self.bottleneck_expansion = nn.Sequential(
            nn.Conv2d(in_channels = out_channels,
                      out_channels = out_channels,
                      kernel_size = 1,
                      stride = 1,
                      padding = 0,),
            nn.BatchNorm2d(out_channels),)

        # Regularizer layer: dropout
        self.dropout = nn.Dropout2d(p = p)
