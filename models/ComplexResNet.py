"""
Author: Liam Laidlaw
Filename: ComplexResNet.py
Purpose: A complex valued resnet based on the model presetned in "Deep Complex Neural Networks", Trablesi et al. 2018.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from complexPyTorch.complexLayers import  ComplexConv2d, ComplexLinear

__all__ = ['ComplexResNet']

# MODULE: Activation Functions (functional and classes)
# =================================

def modrelu(x, bias: float=1, epsilon: float=1e-8):
    """ModReLU activation function.

    Performs ModReLU over the input tensor. 

    Args:
        x: The input tensor. Must be complex. 
    
    Returns:
        The activated tensor. 

    Raises:
        TypeError: Raised if the input tensor is not complex valued
    """
    if not input.is_complex():
        raise TypeError(f"Input must be a complex tensor. Got type {input.dtype}")
    magnitude = x.abs()
    activated_magnitude = F.relu(magnitude + bias)
    nonzero_magnitude = magnitude + epsilon
    return activated_magnitude * (magnitude / nonzero_magnitude)


def zrelu(x: torch.tensor) -> torch.tensor:
    """zReLU activation function.

    Performs zReLU over the input tensor. 

    Args:
        x: The input tensor. Must be complex. 
    
    Returns:
        The activated tensor. 

    Raises:
        TypeError: Raised if the input tensor is not complex valued
    """
    if not x.is_complex():
        raise TypeError(f"Input must be a complex tensor. Got type {input.dtype}")
    # binary mask is faster than direct angle calculation
    mask = (x.real >= 0) & (x.imag >= 0)
    return x * mask.to(x.dtype)


def crelu(x: torch.tensor) -> torch.tensor:
    """Complex ReLU activation function.

    Performs complex relu over the input tensor.

    Args:
        x: The input tensor. If real valued, traditional relu is performed. 

    Returns:
        The activated tensor. 
    """
    return torch.complex(F.relu(x.real), F.relu(x.imag)).to(x.dtype)


def complex_cardioid(x: torch.tensor) -> torch.tensor:
    """Complex Cardioid activation function.

    Performs complex cardioid over the input tensor.

    Args:
        x: The input tensor. Input must be complex valued.

    Returns:
        The activated tensor.
    
    Raises:
        TypeError: Raised if the input tensor is not complex valued
    """
    
    if not x.is_complex():
        raise TypeError(f"Input must be a complex tensor. Got type {input.dtype}")
    angle = torch.angle(x)
    return 0.5 * (1 + torch.cos(angle)) * x


def abs_softmax(input: torch.tensor) -> torch.tensor:
    """Magnitude based softmax.

    Performs softmax on the magnitude of each value in the input tensor. 

    Args:
        input: The input tensor. If the tensor is real valued, regular softmax is applied. 

    Returns:
        The activated tensor. 
    """
    return F.softmax(input.abs())


# A simple complex dropout module
class ComplexDropout(nn.Module):
    def __init__(self, p=0.3):
        super(ComplexDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        # Create a mask and apply it to both real and imaginary parts
        mask = torch.ones_like(x.real)
        mask = F.dropout(mask, self.p, self.training)
        return torch.complex(x.real * mask, x.imag * mask)

class ModReLU(nn.Module):
    def __init__(self, bias: float = 1.0, inplace: bool = False):
        """ModReLU module. 

        Args:
            bias: Initial value for the learnable bias.
            inplace: Whether to perform the operation in-place. Defaults to False.
        """
        super(ModReLU, self).__init__()
        # The bias is added to the magnitude, so it must be a real-valued parameter
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        self.inplace = inplace

    def forward(self, input):
        if self.inplace:
            magnitude = input.abs()
            # The scaling factor is broadcastable to the complex tensor
            scaling_factor = F.relu(magnitude + self.bias) / (magnitude + 1e-8)
            input.real.mul_(scaling_factor)
            input.imag.mul_(scaling_factor)
            return input
        return modrelu(input, self.bias)


class ZReLU(nn.Module):
    def __init__(self, inplace: bool = False):
        """zReLU module. 

        Args:
            inplace: Whether to perform the operation in-place. Defaults to False.
        """
        super(ZReLU, self).__init__()
        self.inplace = inplace
    
    def forward(self, input):
        if self.inplace:
            mask = (input.real >= 0) & (input.imag >= 0)
            input.mul_(mask.to(input.dtype))
            return input
        return zrelu(input)


class ComplexCardioid(nn.Module):
    def __init__(self, inplace: bool = False):
        """ComplexCardioid module. 

        Args:
            inplace: Whether to perform the operation in-place. Defaults to False.
        """
        super(ComplexCardioid, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        if self.inplace:
            angle = torch.angle(input)
            scaling_factor = 0.5 * (1 + torch.cos(angle))
            input.real.mul_(scaling_factor)
            input.imag.mul_(scaling_factor)
            return input
        return complex_cardioid(input)


class CReLU(nn.Module):
    def __init__(self, inplace: bool = False):
        """Complex ReLU module. 

        Performs complex ReLU activation as a module. Can be used in a module list like traditional ReLU.
        """
        super(CReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        if self.inplace:
            F.relu(input.real, inplace=True)
            F.relu(input.imag, inplace=True)
            return input
        else:
            return crelu(input)


class Abs(nn.Module):
    def __init__(self):
        """Abs (magnitude) module. 

        Performs magnitude calculation as a module. Can be used in a module list like other torch modules.
        """
        super(Abs, self).__init__()

    def forward(self, input):
        return input.abs()


class AbsSoftmax(nn.Module):
    def __init__(self):
        """Abs (magnitude) softmax module. 

        Performs magnitude-based softmax activation as a module. Can be used in a module list like traditional softmax.
        """
        super(AbsSoftmax, self).__init__()

    def forward(self, input):
        return abs_softmax(input)


# MODULE: UTILITY & INITIALIZATION
# =================================

def init_weights(m):
    """
    Applies the paper's weight initialization to the model's layers.
    Initializes real convolutions with scaled orthogonal matrices and
    complex convolutions with scaled unitary matrices.
    """
    if isinstance(m, ComplexConv2d):
        # Unitary initialization for complex convolutions using SVD
        real_conv = m.conv_r
        fan_in = real_conv.in_channels * real_conv.kernel_size[0] * real_conv.kernel_size[1]
        
        weight_shape = real_conv.weight.shape
        flat_shape = (weight_shape[0], weight_shape[1] * weight_shape[2] * weight_shape[3])
        
        random_matrix = torch.randn(flat_shape, dtype=torch.complex64)
        
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

class ComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        super(ComplexDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        # Create a mask and apply it to both real and imaginary parts
        mask = torch.ones_like(x.real)
        mask = F.dropout(mask, self.p, self.training)
        return torch.complex(x.real * mask, x.imag * mask)

# MODULE: RESIDUAL BLOCKS
# ========================

class ComplexResidualBlock(nn.Module):
    def __init__(self, channels, activation_fn_class, dropout_rate: float = 0.5):
        super(ComplexResidualBlock, self).__init__()
        self.bn1 = ComplexBatchNorm2d(channels)
        self.relu1 = activation_fn_class()
        self.conv1 = ComplexConv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = ComplexBatchNorm2d(channels)
        self.relu2 = activation_fn_class()
        self.dropout = ComplexDropout(p=dropout_rate)
        self.conv2 = ComplexConv2d(channels, channels, kernel_size=3, padding=1, bias=False)

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

# MODULE: NETWORK ARCHITECTURES
# ==============================


class ComplexBatchNorm2d(nn.Module):
    """
    A DataParallel-compatible implementation of the complex batch normalization
    described in "Deep Complex Networks" (Trabelsi et al., 2018).

    This layer performs a 2D whitening operation that decorrelates the real
    and imaginary parts of the complex-valued activations. It is designed
    to work with torch.nn.DataParallel by performing calculations directly
    in the forward pass.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, track_running_stats=True):
        super(ComplexBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats

        self.weight = Parameter(torch.Tensor(num_features, 3)) 
        self.bias = Parameter(torch.Tensor(num_features, 2))   

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, 2))
            self.register_buffer('running_cov', torch.zeros(num_features, 3))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_cov', None)
            self.register_parameter('num_batches_tracked', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_cov.zero_()
            # --- Initialize running_cov correctly as per the paper ---
            # The moving averages of Vri and beta are initialized to 0.
            self.running_cov[:, 2].zero_()
            # The moving averages of Vrr and Vii are initialized to 1/sqrt(2).
            self.running_cov[:, 0].fill_(1 / math.sqrt(2))
            self.running_cov[:, 1].fill_(1 / math.sqrt(2))
            self.num_batches_tracked.zero_()
        
        # Initialize bias (beta) to zero
        self.bias.data.zero_()
        
        # Initialize gamma_ri to 0
        self.weight.data[:, 2].zero_()
        # Initialize gamma_rr and gamma_ii to 1/sqrt(2)
        self.weight.data[:, 0].fill_(1 / math.sqrt(2))
        self.weight.data[:, 1].fill_(1 / math.sqrt(2))


    def forward(self, x):
        if not x.is_complex():
            raise TypeError("Input must be a complex tensor.")

        if self.training and self.track_running_stats:
            mean_complex = x.mean(dim=[0, 2, 3])
            mean_for_update = torch.stack([mean_complex.real, mean_complex.imag], dim=1).detach()
            self.num_batches_tracked.add_(1)
            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * mean_for_update

            centered_x = x - mean_complex.view(1, self.num_features, 1, 1)
            V_rr = (centered_x.real ** 2).mean(dim=[0, 2, 3])
            V_ii = (centered_x.imag ** 2).mean(dim=[0, 2, 3])
            V_ri = (centered_x.real * centered_x.imag).mean(dim=[0, 2, 3])
            cov_for_update = torch.stack([V_rr, V_ii, V_ri], dim=1).detach()
            self.running_cov.data = (1 - self.momentum) * self.running_cov.data + self.momentum * cov_for_update
            
            mean_to_use = mean_complex
            cov_to_use = torch.stack([V_rr, V_ii, V_ri], dim=1)
        else:
            mean_to_use = torch.complex(self.running_mean[:, 0], self.running_mean[:, 1])
            cov_to_use = self.running_cov
        
        mean_reshaped = mean_to_use.view(1, self.num_features, 1, 1)
        centered_x = x - mean_reshaped
        
        V_rr = cov_to_use[:, 0].view(1, self.num_features, 1, 1) + self.eps
        V_ii = cov_to_use[:, 1].view(1, self.num_features, 1, 1) + self.eps
        V_ri = cov_to_use[:, 2].view(1, self.num_features, 1, 1)
        
        s = V_rr * V_ii - V_ri ** 2
        t = s.clamp(min=self.eps).rsqrt()
        
        Rrr = V_ii * t
        Rii = V_rr * t
        Rri = -V_ri * t

        real_part = Rrr * centered_x.real + Rri * centered_x.imag
        imag_part = Rri * centered_x.real + Rii * centered_x.imag
        whitened_x = torch.complex(real_part, imag_part)

        gamma_rr = self.weight[:, 0].view(1, self.num_features, 1, 1)
        gamma_ii = self.weight[:, 1].view(1, self.num_features, 1, 1)
        gamma_ri = self.weight[:, 2].view(1, self.num_features, 1, 1)
        beta_r = self.bias[:, 0].view(1, self.num_features, 1, 1)
        beta_i = self.bias[:, 1].view(1, self.num_features, 1, 1)

        out_real = gamma_rr * whitened_x.real + gamma_ri * whitened_x.imag + beta_r
        out_imag = gamma_ri * whitened_x.real + gamma_ii * whitened_x.imag + beta_i

        return torch.complex(out_real, out_imag)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.num_features}, '
                f'eps={self.eps}, momentum={self.momentum}, '
                f'track_running_stats={self.track_running_stats})')

class ComplexResNet(nn.Module):
    # MODIFIED LINE: Add dropout_rate to the constructor
    def __init__(self, architecture_type: str, activation_function: str, input_channels: int, num_classes: int, dropout_rate: float = 0.2):
        super(ComplexResNet, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        self.dropout_rate = dropout_rate
        
        configs = {
            'WS': {'filters': 10, 'blocks_per_stage': [3, 3, 3]}, 
            'DN': {'filters': 8,  'blocks_per_stage': [4, 4, 4]},
            'IB': {'filters': 9,  'blocks_per_stage': [4, 4, 4]}, 
        }

        config = configs[architecture_type]
        self.initial_filters = config['filters']
        self.blocks_per_stage = config['blocks_per_stage']
        activation_map = {'crelu': CReLU, 'zrelu': ZReLU, 'modrelu': ModReLU, 'complex_cardioid': ComplexCardioid}
        self.activation_function = activation_function
        self.activation_fn_class = activation_map.get(activation_function.lower())
        if self.activation_fn_class is None:
            raise ValueError(f"Unknown activation function: {activation_function}")
        self.initial_complex_op = nn.Sequential(
            ComplexConv2d(input_channels, self.initial_filters, kernel_size=3, stride=1, padding=1, bias=False),
            ComplexBatchNorm2d(self.initial_filters),
            self.activation_fn_class()
        )
        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        current_channels = self.initial_filters
        for i, num_blocks in enumerate(self.blocks_per_stage):
            # MODIFIED LINE: Pass the dropout_rate to the ComplexResidualBlock
            stage = nn.Sequential(*[ComplexResidualBlock(current_channels, self.activation_fn_class, dropout_rate=self.dropout_rate) for _ in range(num_blocks)])
            self.stages.append(stage)
            
            if i < len(self.blocks_per_stage) - 1:
                self.downsample_layers.append(ComplexConv2d(current_channels, current_channels, kernel_size=1, stride=1, bias=False))
            current_channels *= 2
        self.final_channels = self.initial_filters * (2**(len(self.blocks_per_stage) - 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.final_channels*2, num_classes)
        self.apply(init_weights)


    def set_input(self, input_channels: int, num_classes:int):
        self.num_classes = num_classes
        self.initial_op = nn.Sequential(
            ComplexConv2d(input_channels, self.initial_filters, kernel_size=3, stride=1, padding=1, bias=False),
            ComplexBatchNorm2d(self.initial_filters),
            self.activation_fn_class()
        )

        self.fc = nn.Linear(self.final_channels*2, num_classes)

        
    def forward(self, x):
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
        x_real = self.avgpool(x.real)
        x_imag = self.avgpool(x.imag)
        x_real_flat = torch.flatten(x_real, 1)
        x_imag_flat = torch.flatten(x_imag, 1)
        # Calculate magnitude and phase directly for efficiency and stability
        magnitude = torch.hypot(x_real_flat, x_imag_flat)
        phase = torch.atan2(x_imag_flat, x_real_flat)
        # Concatenate and pass to the final layer
        x = torch.cat([magnitude, phase], dim=1)
        logits = self.fc(x)
        return logits
