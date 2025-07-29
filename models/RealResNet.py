"""
Author: Liam Laidlaw
Filename: RealResNet.py
Purpose: A real valued varient of the complex valued resnet (ComplexResNet.py)
Based on the model presetned in "Deep Complex Neural Networks", Trablesi et al. 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import orthogonal_
import math

__all__ = ['RealResNet']

def init_weights(m):
    """
    Applies the paper's weight initialization to the model's layers.
    Initializes real convolutions with scaled orthogonal matrices and
    complex convolutions with scaled unitary matrices.
    """
    if isinstance(m, nn.Conv2d):
        # Orthogonal initialization for real-valued convolutions
        fan_in = nn.init._calculate_fan_in_and_fan_out(m.weight)[0]
        
        flat_shape = (m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1])
        random_matrix = torch.randn(flat_shape)
        orthogonal_matrix = orthogonal_(random_matrix)
        reshaped_matrix = orthogonal_matrix.reshape(m.weight.shape)
        
        he_variance = 2.0 / fan_in
        scaling_factor = math.sqrt(he_variance * m.out_channels)
        
        with torch.no_grad():
            m.weight.copy_(reshaped_matrix * scaling_factor)


class RealResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout_rate: float = 0.2):
        """Single real-valued residual block.

        Defines one real valued residual block consisting of the following operations: input -> BN -> ReLU -> Conv2d -> BN -> ReLU -> Conv2d + input (original input tensor)

        Args:
            channels: The number of channels in the input data.
        """
        super(RealResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = out + identity
        return out

class RealResNet(nn.Module):
    # MODIFIED LINE: Add dropout_rate to the constructor
    def __init__(self, architecture_type: str, input_channels: int, num_classes: int, dropout_rate: float = 0.2):
        """Real-Valued Convolutional Residual Network.

        RVCNN Based on the network presented in "Deep Complex Networks", Trabelsi et al. 2018.
        Meant to be used for comparison with its complex varient.

        Args:
            architecture_type: The the width and depth of the residual stages of the network. Options are:
                - 'WS' (wide shallow) | 18 convolutional filters with 14 blocks per stage.
                - 'DN' (deep narrow) | 14 convolutional filters with 23 blocks per stage.
                - 'IB' (in-between) | 16 convolutional filters with 18 blocks per stage.
            input_channels: The number of input channels the network should expect. Defaults to 3.
            num_classes The number of classes to classify into. Defaults to 10.
        """
        super(RealResNet, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.dropout_rate = dropout_rate

        configs = {
            'WS': {'filters': 18, 'blocks_per_stage': [4, 4, 4]},
            'DN': {'filters': 16, 'blocks_per_stage': [5, 5, 5]}, 
            'IB': {'filters': 17, 'blocks_per_stage': [4, 4, 4]}, 
        }

        config = configs[architecture_type]
        self.initial_filters = config['filters']
        self.blocks_per_stage = config['blocks_per_stage']
        self.initial_op = nn.Sequential(
            nn.Conv2d(input_channels, self.initial_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.initial_filters),
            nn.ReLU(inplace=False)
        )
        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        current_channels = self.initial_filters
        for i, num_blocks in enumerate(self.blocks_per_stage):
            stage = nn.Sequential(*[RealResidualBlock(current_channels, dropout_rate=self.dropout_rate) for _ in range(num_blocks)])
            self.stages.append(stage)
            
            if i < len(self.blocks_per_stage) - 1:
                self.downsample_layers.append(nn.Conv2d(current_channels, current_channels, kernel_size=1, stride=1, bias=False))
            current_channels *= 2
        self.final_channels = self.initial_filters * (2**(len(self.blocks_per_stage) - 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.final_channels, num_classes)
        self.apply(init_weights)


    def set_input(self, input_channels: int, num_classes:int):
        self.num_classes = num_classes
        self.initial_op = nn.Sequential(
            nn.Conv2d(input_channels, self.initial_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.initial_filters),
            nn.ReLU(inplace=False)
        )

        self.fc = nn.Linear(self.final_channels, num_classes)


    def forward(self, x):
        x = self.initial_op(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:
                projection_conv = self.downsample_layers[i]
                projected_x = projection_conv(x)
                x = torch.cat([x, projected_x], dim=1)
                x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
