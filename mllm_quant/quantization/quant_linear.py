"""
Fake quantized linear layer for GPTQ.
Performs fake quantization during forward pass but stores weights in original precision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class QuantLinear(nn.Module):
    """
    Fake quantized linear layer.
    
    During forward pass, weights are quantized and dequantized (fake quantization).
    The actual weights are stored in original precision for saving.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        bias: Whether to include bias
        bits: Quantization bit width (default: 4)
        group_size: Group size for group-wise quantization (default: 128)
        sym: Whether to use symmetric quantization (default: True)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 4,
        group_size: int = 128,
        sym: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size if group_size > 0 else in_features
        self.sym = sym
        
        # Number of groups
        self.n_groups = (in_features + self.group_size - 1) // self.group_size
        
        # Weight in original precision
        self.register_buffer(
            'weight',
            torch.zeros(out_features, in_features)
        )
        
        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.register_buffer('bias', None)
        
        # Quantization parameters (per-group scales and zero points)
        self.register_buffer(
            'scales',
            torch.zeros(out_features, self.n_groups)
        )
        self.register_buffer(
            'zeros',
            torch.zeros(out_features, self.n_groups)
        )
        
        # Flag to indicate if quantization params are computed
        self.quantized = False
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 4,
        group_size: int = 128,
        sym: bool = True,
    ) -> "QuantLinear":
        """
        Create QuantLinear from an existing nn.Linear layer.
        
        Args:
            linear: The original Linear layer
            bits: Quantization bit width
            group_size: Group size for quantization
            sym: Whether to use symmetric quantization
            
        Returns:
            QuantLinear instance
        """
        quant_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            bits=bits,
            group_size=group_size,
            sym=sym,
        )
        
        # Copy weights
        quant_linear.weight.copy_(linear.weight.data)
        if linear.bias is not None:
            quant_linear.bias.copy_(linear.bias.data)
        
        return quant_linear
    
    def compute_quant_params(self) -> None:
        """
        Compute quantization parameters (scales and zeros) from current weights.
        Called after GPTQ optimization.
        """
        weight = self.weight.data
        
        # Quantization range
        if self.sym:
            qmax = 2 ** (self.bits - 1) - 1
            qmin = -2 ** (self.bits - 1)
        else:
            qmax = 2 ** self.bits - 1
            qmin = 0
        
        # Compute per-group scales and zeros
        for g in range(self.n_groups):
            start_idx = g * self.group_size
            end_idx = min(start_idx + self.group_size, self.in_features)
            
            group_weight = weight[:, start_idx:end_idx]
            
            w_min = group_weight.min(dim=1, keepdim=False).values
            w_max = group_weight.max(dim=1, keepdim=False).values
            
            if self.sym:
                # Symmetric quantization
                w_abs_max = torch.max(w_min.abs(), w_max.abs())
                scale = w_abs_max / qmax
                zero = torch.zeros_like(scale)
            else:
                # Asymmetric quantization
                scale = (w_max - w_min) / (qmax - qmin)
                zero = qmin - w_min / scale
            
            # Avoid division by zero
            scale = scale.clamp(min=1e-10)
            
            self.scales[:, g] = scale
            self.zeros[:, g] = zero
        
        self.quantized = True
    
    def fake_quantize_weight(self) -> torch.Tensor:
        """
        Perform fake quantization on weights.
        
        Returns:
            Fake quantized weight tensor
        """
        if not self.quantized:
            self.compute_quant_params()
        
        weight = self.weight.data
        
        # Quantization range
        if self.sym:
            qmax = 2 ** (self.bits - 1) - 1
            qmin = -2 ** (self.bits - 1)
        else:
            qmax = 2 ** self.bits - 1
            qmin = 0
        
        # Fake quantize per group
        fake_quant_weight = torch.zeros_like(weight)
        
        for g in range(self.n_groups):
            start_idx = g * self.group_size
            end_idx = min(start_idx + self.group_size, self.in_features)
            
            scale = self.scales[:, g].unsqueeze(1)
            zero = self.zeros[:, g].unsqueeze(1)
            
            group_weight = weight[:, start_idx:end_idx]
            
            # Quantize
            q_weight = torch.round(group_weight / scale + zero)
            q_weight = q_weight.clamp(qmin, qmax)
            
            # Dequantize
            fake_quant_weight[:, start_idx:end_idx] = (q_weight - zero) * scale
        
        return fake_quant_weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fake quantized weights.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        if self.quantized:
            weight = self.fake_quantize_weight()
        else:
            weight = self.weight
        
        return F.linear(x, weight, self.bias)
    
    def to_linear(self) -> nn.Linear:
        """
        Convert back to standard nn.Linear with fake quantized weights.
        
        Returns:
            nn.Linear with fake quantized weights
        """
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
        )
        
        if self.quantized:
            linear.weight.data.copy_(self.fake_quantize_weight())
        else:
            linear.weight.data.copy_(self.weight)
        
        if self.bias is not None:
            linear.bias.data.copy_(self.bias)
        
        return linear
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bits={self.bits}, "
            f"group_size={self.group_size}, sym={self.sym}"
        )
