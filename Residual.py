"""
Author: Liam Laidlaw
Purpose: Classes for defining real and complex valued residual blocks.
Based on the architecture presented in "Deep Complex Networks", Trabelsi et al., 2018. 
View original paper here: https://openreview.net/forum?id=H1T2hmZAb

This version is configured for use with torch.nn.DataParallel and includes
the specific weight initialization from the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import orthogonal_
from complexPyTorch.complexLayers import  ComplexConv2d, ComplexLinear
from Activations import CReLU, ZReLU, ModReLU, ComplexCardioid
# Import the DataParallel-compatible complex batch norm layer
from CustomComplexLayers import ComplexBatchNorm2d
import math

# --- WEIGHT INITIALIZATION (as per Paper Sec. 3.6) ---

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

    elif isinstance(m, ComplexConv2d):
        # Unitary initialization for complex convolutions using SVD
        real_conv = m.conv_r
        fan_in = real_conv.in_channels * real_conv.kernel_size[0] * real_conv.kernel_size[1]
        
        weight_shape = real_conv.weight.shape
        flat_shape = (weight_shape[0], weight_shape[1] * weight_shape[2] * weight_shape[3])
        
        random_matrix = torch.randn(flat_shape, dtype=torch.complex64)
        
        # --- FIX: Use SVD to create a unitary matrix of the correct shape ---
        U, _, Vh = torch.linalg.svd(random_matrix, full_matrices=False)
        unitary_matrix_flat = U @ Vh
        unitary_matrix = unitary_matrix_flat.reshape(weight_shape)
        
        he_variance = 2.0 / fan_in
        scaling_factor = math.sqrt(he_variance * weight_shape[0])
        
        scaled_unitary = unitary_matrix * scaling_factor

        with torch.no_grad():
            m.conv_r.weight.copy_(scaled_unitary.real)
            m.conv_i.weight.copy_(scaled_unitary.imag)

    elif isinstance(m, ComplexLinear):
        # Unitary initialization for complex linear layers using SVD
        real_fc = m.fc_r
        fan_in = real_fc.in_features
        
        random_matrix = torch.randn(real_fc.weight.shape, dtype=torch.complex64)
        
        U, _, Vh = torch.linalg.svd(random_matrix, full_matrices=False)
        unitary_matrix = U @ Vh
        
        he_variance = 2.0 / fan_in
        scaling_factor = math.sqrt(he_variance * real_fc.out_features)
        
        scaled_unitary = unitary_matrix * scaling_factor
        
        with torch.no_grad():
            m.fc_r.weight.copy_(scaled_unitary.real)
            m.fc_i.weight.copy_(scaled_unitary.imag)


# MODULE: UTILITY & INITIALIZATION
# =================================

class ZeroImag(nn.Module):
    def __init__(self):
        super(ZeroImag, self).__init__()
    def forward(self, x):
        return torch.zeros_like(x)

class ImaginaryComponentLearner(nn.Module):
    def __init__(self, channels):
        super(ImaginaryComponentLearner, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        )
    def forward(self, x):
        return self.layers(x)

# MODULE: RESIDUAL BLOCKS
# ========================

class ComplexResidualBlock(nn.Module):
    def __init__(self, channels, activation_fn_class):
        super(ComplexResidualBlock, self).__init__()
        self.bn1 = ComplexBatchNorm2d(channels)
        self.relu1 = activation_fn_class()
        self.conv1 = ComplexConv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = ComplexBatchNorm2d(channels)
        self.relu2 = activation_fn_class()
        self.conv2 = ComplexConv2d(channels, channels, kernel_size=3, padding=1, bias=False)
    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = out + identity
        return out

class RealResidualBlock(nn.Module):
    def __init__(self, channels):
        super(RealResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = out + identity
        return out

# MODULE: NETWORK ARCHITECTURES
# ==============================

class ComplexResNet(nn.Module):
    def __init__(self, block_class, architecture_type, activation_function, learn_imaginary_component, input_channels=3, num_classes=10):
        super(ComplexResNet, self).__init__()
        configs = {'WS': {'filters': 12, 'blocks_per_stage': [16, 16, 16]}, 'DN': {'filters': 10, 'blocks_per_stage': [23, 23, 23]}, 'IB': {'filters': 11, 'blocks_per_stage': [19, 19, 19]}}
        config = configs[architecture_type]
        self.initial_filters = config['filters']
        self.blocks_per_stage = config['blocks_per_stage']
        activation_map = {'crelu': CReLU, 'zrelu': ZReLU, 'modrelu': ModReLU, 'complex_cardioid': ComplexCardioid}
        self.activation_fn_class = activation_map.get(activation_function.lower())
        if self.activation_fn_class is None:
            raise ValueError(f"Unknown activation function: {activation_function}")
        if learn_imaginary_component:
            self.imag_handler = ImaginaryComponentLearner(input_channels) 
        else:
            self.imag_handler = ZeroImag()
        self.initial_complex_op = nn.Sequential(
            ComplexConv2d(input_channels, self.initial_filters, kernel_size=3, stride=1, padding=1, bias=False),
            ComplexBatchNorm2d(self.initial_filters),
            self.activation_fn_class()
        )
        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        current_channels = self.initial_filters
        for i, num_blocks in enumerate(self.blocks_per_stage):
            self.stages.append(self._make_stage(block_class, current_channels, num_blocks))
            if i < len(self.blocks_per_stage) - 1:
                self.downsample_layers.append(ComplexConv2d(current_channels, current_channels, kernel_size=1, stride=1, bias=False))
            current_channels *= 2
        final_channels = self.initial_filters * (2**(len(self.blocks_per_stage) - 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # conversion to real happens here (right before linear)
        self.fc = nn.Linear(final_channels*2, num_classes)
        self.apply(init_weights)
    def _make_stage(self, block_class, channels, num_blocks):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(block_class(channels, self.activation_fn_class))
        return nn.Sequential(*blocks)
    def forward(self, x_real):
        x_imag = self.imag_handler(x_real)
        x = torch.complex(x_real, x_imag)
        x = self.initial_complex_op(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:
                projection_conv = self.downsample_layers[i]
                projected_x = projection_conv(x)
                x = torch.cat([x, projected_x], dim=1)
                pooled_real = F.avg_pool2d(x.real, kernel_size=2, stride=2)
                pooled_imag = F.avg_pool2d(x.imag, kernel_size=2, stride=2)
                x = torch.complex(pooled_real, pooled_imag)
        pooled_real = self.avgpool(x.real)
        pooled_imag = self.avgpool(x.imag)
        x = torch.complex(pooled_real, pooled_imag)
        x = torch.flatten(x, 1)
        # converts tensor to a flattened real_valued tensor containing both the phase and magnitude values
        x = torch.cat([x.abs(), x.angle()], dim=1)
        logits = self.fc(x)
        return logits

class RealResNet(nn.Module):
    def __init__(self, block_class, architecture_type, input_channels=3, num_classes=10):
        super(RealResNet, self).__init__()
        configs = {'WS': {'filters': 18, 'blocks_per_stage': [14, 14, 14]}, 'DN': {'filters': 14, 'blocks_per_stage': [23, 23, 23]}, 'IB': {'filters': 16, 'blocks_per_stage': [18, 18, 18]}}
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
            self.stages.append(self._make_stage(block_class, current_channels, num_blocks))
            if i < len(self.blocks_per_stage) - 1:
                self.downsample_layers.append(nn.Conv2d(current_channels, current_channels, kernel_size=1, stride=1, bias=False))
            current_channels *= 2
        final_channels = self.initial_filters * (2**(len(self.blocks_per_stage) - 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(final_channels, num_classes)
        self.apply(init_weights)
    def _make_stage(self, block_class, channels, num_blocks):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(block_class(channels))
        return nn.Sequential(*blocks)
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
