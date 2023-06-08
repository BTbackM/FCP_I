import torch
import torch.nn as nn

class InitialBlock(nn.Module):
    """
    Based on paper: https://arxiv.org/pdf/1606.02147.pdf
    Initial block of the model ENet
    """
    def __init__(self,
                 in_channels,
                 out_channels,) -> None:
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
