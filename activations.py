"""
Author: Liam Laidlaw
Purpose: Additonal Activation Functions for cvnns.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F


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


class ModReLU(nn.Module):
    def __init__(self, bias: float=1, dtype=torch.complex64) -> None:
        """ModReLU module. 

        Performs ModReLU as a module. Can be used in a module list like traditional ReLU.

        Args:
            bias: Learnable bias value added to the activation.
            dtype: The expected datatype of inputs. Defaults to torch.complex64. 
        """
        super(ModReLU, self).__init__()
        self.bias = nn.Parameter(torch.tensor(bias, dtype=dtype)) # make bias learnable

    def forward(self, input):
        return modrelu(input, self.bias)


class ZReLU(nn.Module):
    def __init__(self):
        """zReLU module. 

        Performs zReLU activation as a module. Can be used in a module list like tractitional ReLU.
        """
        super(ZReLU, self).__init__()
    
    def forward(self, input):
        return zrelu(input)


class ComplexCardioid(nn.Module):
    def __init__(self):
        """ComplexCardioid module. 

        Performs complex cardioid activation as a module. Can be used in a module list like traditional ReLU.
        """
        super(ComplexCardioid, self).__init__()

    def forward(self, input):
        return complex_cardioid(input)


class CReLU(nn.Module):
    def __init__(self):
        """Complex ReLU module. 

        Performs complex ReLU activation as a module. Can be used in a module list like traditional ReLU.
        """
        super(CReLU, self).__init__()

    def forward(self, input):
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


