import torch
import torch.nn as nn
import torch.nn.functional as F

from enet.initial_block import InitialBlock
from enet.bottleneck import (
    Downsampling,
    Upsampling,
    Regular,
    Dilated,
    Asymmetric,)

class ENet(nn.Module):
    """
    Based on paper: https://arxiv.org/pdf/1606.02147.pdf
    ENet model architecture.
    """
    def __init__(self, num_classes):
        super(ENet, self).__init__()
        # Initial block
        self.initial_block = InitialBlock(3, 16)

        # Block 1
        self.bottleneck1_0 = Downsampling(in_channels = 16,
                                          out_channels = 64,
                                          return_indices = True,
                                          dropout_prob = 0.01)
        self.bottleneck1_1 = Regular(channels = 64,
                                     padding = 1,
                                     dropout_prob = 0.01)
        self.bottleneck1_2 = Regular(channels = 64,
                                     padding = 1,
                                     dropout_prob = 0.01)
        self.bottleneck1_3 = Regular(channels = 64,
                                     padding = 1,
                                     dropout_prob = 0.01)
        self.bottleneck1_4 = Regular(channels = 64,
                                     padding = 1,
                                     dropout_prob = 0.01)

        # Block 2
        self.bottleneck2_0 = Downsampling(in_channels = 64,
                                          out_channels = 128,
                                          return_indices = True,
                                          dropout_prob = 0.1)
        self.bottleneck2_1 = Regular(channels = 128,
                                     padding = 1,
                                     dropout_prob = 0.1)
        self.bottleneck2_2 = Dilated(channels = 128,
                                     dilation = 2,
                                     padding = 2,
                                     dropout_prob = 0.1)
        self.bottleneck2_3 = Asymmetric(channels = 128,
                                        kernel_size = 5,
                                        padding = 2,
                                        dropout_prob = 0.1)
        self.bottleneck2_4 = Dilated(channels = 128,
                                     dilation = 4,
                                     padding = 4,
                                     dropout_prob = 0.1)
        self.bottleneck2_5 = Regular(channels = 128,
                                     padding = 1,
                                     dropout_prob = 0.1)
        self.bottleneck2_6 = Dilated(channels = 128,
                                     dilation = 8,
                                     padding = 8,
                                     dropout_prob = 0.1)
        self.bottleneck2_7 = Asymmetric(channels = 128,
                                        kernel_size = 5,
                                        padding = 2,
                                        dropout_prob = 0.1)
        self.bottleneck2_8 = Dilated(channels = 128,
                                     dilation = 16,
                                     padding = 16,
                                     dropout_prob = 0.1)

        # Block 3
        self.bottleneck3_1 = Regular(channels = 128,
                                     padding = 1,
                                     dropout_prob = 0.1)
        self.bottleneck3_2 = Dilated(channels = 128,
                                     dilation = 2,
                                     padding = 2,
                                     dropout_prob = 0.1)
        self.bottleneck3_3 = Asymmetric(channels = 128,
                                        kernel_size = 5,
                                        padding = 2,
                                        dropout_prob = 0.1)
        self.bottleneck3_4 = Dilated(channels = 128,
                                     dilation = 4,
                                     padding = 4,
                                     dropout_prob = 0.1)
        self.bottleneck3_5 = Regular(channels = 128,
                                     padding = 1,
                                     dropout_prob = 0.1)
        self.bottleneck3_6 = Dilated(channels = 128,
                                     dilation = 8,
                                     padding = 8,
                                     dropout_prob = 0.1)
        self.bottleneck3_7 = Asymmetric(channels = 128,
                                        kernel_size = 5,
                                        padding = 2,
                                        dropout_prob = 0.1)
        self.bottleneck3_8 = Dilated(channels = 128,
                                     dilation = 16,
                                     padding = 16,
                                     dropout_prob = 0.1)

        # Block 4
        self.bottleneck4_0 = Upsampling(in_channels = 128,
                                        out_channels = 64,
                                        dropout_prob = 0.1)
        self.bottleneck4_1 = Regular(channels = 64,
                                     padding = 1,
                                     dropout_prob = 0.1)
        self.bottleneck4_2 = Regular(channels = 64,
                                     padding = 1,
                                     dropout_prob = 0.1)

        # Block 5
        self.bottleneck5_0 = Upsampling(in_channels = 64,
                                        out_channels = 16,
                                        dropout_prob = 0.1)
        self.bottleneck5_1 = Regular(channels = 16,
                                     padding = 1,
                                     dropout_prob = 0.1)

        self.fullconv = nn.ConvTranspose2d(16,
                                           num_classes,
                                           kernel_size = 3,
                                           stride = 2,
                                           padding = 1,)
        
    def forward(self, x):
        # Initial block
        input_size = x.size()
        x = self.initial_block(x)

        # Block 1: Encoder
        block1_input_size = x.size()
        x, max_indices1 = self.bottleneck1_0(x)
        x = self.bottleneck1_1(x)
        x = self.bottleneck1_2(x)
        x = self.bottleneck1_3(x)
        x = self.bottleneck1_4(x)

        # Block 2: Encoder
        block2_input_size = x.size()
        x, max_indices2 = self.bottleneck2_0(x)
        x = self.bottleneck2_1(x)
        x = self.bottleneck2_2(x)
        x = self.bottleneck2_3(x)
        x = self.bottleneck2_4(x)
        x = self.bottleneck2_5(x)
        x = self.bottleneck2_6(x)
        x = self.bottleneck2_7(x)
        x = self.bottleneck2_8(x)

        # Block 3: Encoder
        x = self.bottleneck3_1(x)
        x = self.bottleneck3_2(x)
        x = self.bottleneck3_3(x)
        x = self.bottleneck3_4(x)
        x = self.bottleneck3_5(x)
        x = self.bottleneck3_6(x)
        x = self.bottleneck3_7(x)
        x = self.bottleneck3_8(x)

        # Block 4: Decoder
        x = self.bottleneck4_0(x, max_indices2, output_size = block2_input_size)
        x = self.bottleneck4_1(x)
        x = self.bottleneck4_2(x)

        # Block 5: Decoder
        x = self.bottleneck5_0(x, max_indices1, output_size = block1_input_size)
        x = self.bottleneck5_1(x)

        x = self.fullconv(x, output_size = input_size)

        return x
