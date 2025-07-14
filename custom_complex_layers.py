"""
Custom Complex-Valued Layers for PyTorch

This module provides a custom implementation of ComplexBatchNorm2d that is
compatible with torch.nn.DataParallel for multi-GPU training.
"""
import torch
import torch.nn as nn
from torch.nn import Parameter
import math

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
            # --- FIX: Initialize running_cov correctly as per the paper ---
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
        t = torch.sqrt(s)
        inv_t = 1.0 / t
        
        Rrr = V_ii * inv_t
        Rii = V_rr * inv_t
        Rri = -V_ri * inv_t

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
