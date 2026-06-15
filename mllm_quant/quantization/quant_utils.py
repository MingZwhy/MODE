"""
Quantization utilities for weight quantization.

Provides:
- WeightQuantizer: Weight quantization with symmetric/asymmetric modes
- find_qlayers: Utility to find layers to quantize
- Quantization helper functions
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


def get_minq_maxq(bits: int, sym: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get min and max quantization values for given bit-width.
    
    Args:
        bits: Quantization bit-width
        sym: Whether to use symmetric quantization
        
    Returns:
        Tuple of (minq, maxq) tensors
    """
    if sym:
        maxq = torch.tensor(2**(bits-1)-1)
        minq = -maxq - 1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = torch.tensor(0)
    return minq, maxq


def asym_quant(
    x: torch.Tensor, 
    scale: torch.Tensor, 
    zero: torch.Tensor, 
    maxq: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Asymmetric quantization.
    
    Args:
        x: Input tensor
        scale: Scale factor
        zero: Zero point
        maxq: Maximum quantization value
        
    Returns:
        Tuple of (quantized, scale, zero)
    """
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(
    q: torch.Tensor, 
    scale: torch.Tensor, 
    zero: torch.Tensor
) -> torch.Tensor:
    """
    Asymmetric dequantization.
    
    Args:
        q: Quantized tensor
        scale: Scale factor
        zero: Zero point
        
    Returns:
        Dequantized tensor
    """
    return scale * (q - zero)


def asym_quant_dequant(
    x: torch.Tensor, 
    scale: torch.Tensor, 
    zero: torch.Tensor, 
    maxq: torch.Tensor
) -> torch.Tensor:
    """
    Asymmetric quantization followed by dequantization (fake quantization).
    
    Args:
        x: Input tensor
        scale: Scale factor
        zero: Zero point
        maxq: Maximum quantization value
        
    Returns:
        Fake quantized tensor
    """
    return asym_dequant(*asym_quant(x, scale, zero, maxq))


def sym_quant(
    x: torch.Tensor, 
    scale: torch.Tensor, 
    maxq: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric quantization.
    
    Args:
        x: Input tensor
        scale: Scale factor
        maxq: Maximum quantization value
        
    Returns:
        Tuple of (quantized, scale)
    """
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq+1), maxq)
    return q, scale


def sym_dequant(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Symmetric dequantization.
    
    Args:
        q: Quantized tensor
        scale: Scale factor
        
    Returns:
        Dequantized tensor
    """
    return scale * q


def sym_quant_dequant(
    x: torch.Tensor, 
    scale: torch.Tensor, 
    maxq: torch.Tensor
) -> torch.Tensor:
    """
    Symmetric quantization followed by dequantization (fake quantization).
    
    Args:
        x: Input tensor
        scale: Scale factor
        maxq: Maximum quantization value
        
    Returns:
        Fake quantized tensor
    """
    return sym_dequant(*sym_quant(x, scale, maxq))


def binary_quant_dequant(x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    """Binary fake quantization used for 1-bit expert experiments."""
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, 1)
    return scale * (q - zero)


def two_compl(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Convert to two's complement representation."""
    return torch.where(x < 0, 2 ** bits + x, x)


def pack_i4(q: torch.Tensor) -> torch.Tensor:
    """
    Pack int4 tensor. Each uint8 stores two int4 values.
    
    Args:
        q: Signed int tensor with values in int4 range
        
    Returns:
        Packed uint8 tensor
    """
    assert torch.is_signed(q), 'The tensor to be packed should be signed int'
    minq, maxq = get_minq_maxq(4, True)
    assert torch.all(torch.logical_and(q >= minq, q <= maxq))

    q_i8 = two_compl(q.to(dtype=torch.int8), 4).to(torch.uint8)
    q_i4 = q_i8[:, 0::2] | (q_i8[:, 1::2] << 4)
    return q_i4


def unpack_i4(x: torch.Tensor) -> torch.Tensor:
    """
    Unpack int4 tensor stored in uint8 into int32 tensor.
    
    Args:
        x: Packed uint8 tensor
        
    Returns:
        Unpacked int32 tensor
    """
    assert x.dtype == torch.uint8, 'The tensor to be unpacked should be stored in uint8'

    out_shape = list(x.shape)
    out_shape[-1] *= 2  # Each uint8 packs two numbers

    # Low 4 bits
    x0 = (x & 0x0f).to(torch.int8)
    x0[x0 >= 8] -= 16
    x0 = x0.view(-1, x0.shape[-1])

    # High 4 bits
    x1 = ((x & 0xf0) >> 4).to(torch.int8)
    x1[x1 >= 8] -= 16
    x1 = x1.view(-1, x1.shape[-1])

    out = torch.empty(out_shape, device=x.device, dtype=torch.int32)
    out = out.view(-1, out.shape[-1])
    # Interleaving
    out[:, 0::2] = x0
    out[:, 1::2] = x1

    return out.view(out_shape)


class WeightQuantizer(nn.Module):
    """
    Weight quantizer supporting per-channel and group-wise quantization.
    
    Supports:
    - Symmetric and asymmetric quantization
    - Per-channel and group-wise quantization
    - MSE-based scale optimization (grid search)
    """

    def __init__(self, weight_quant_params: Optional[Dict] = None):
        super(WeightQuantizer, self).__init__()
        self.scale = None
        self.zero = None
        self.bits = 4
        self.perchannel = True
        self.group_size = -1
        self.sym = True
        self.mse = True
        self.norm = 2.4
        self.grid = 100
        self.maxshrink = 0.8
        self.minq = -8
        self.maxq = 7

        if weight_quant_params is not None and len(weight_quant_params) != 0:
            self.configure(weight_quant_params)

    def configure(self, weight_quant_params: Dict, device: str = 'cuda') -> None:
        """
        Configure quantizer with parameters.
        
        Args:
            weight_quant_params: Dictionary containing:
                - w_bits: Bit-width for quantization
                - perchannel: Whether to use per-channel quantization
                - w_groupsize: Group size (-1 for per-channel)
                - w_asym: Whether to use asymmetric quantization
                - w_clip: Whether to use MSE-based scale optimization
                - norm: Norm for error calculation
                - grid: Grid search resolution
                - maxshrink: Maximum shrink ratio
            device: Target device
        """
        self.bits = weight_quant_params.get('w_bits', 4)
        self.perchannel = weight_quant_params.get('perchannel', True)
        self.group_size = weight_quant_params.get('w_groupsize', -1)
        self.sym = not weight_quant_params.get('w_asym', False)
        self.mse = weight_quant_params.get('w_clip', True)
        self.norm = weight_quant_params.get('norm', 2.4)
        self.grid = weight_quant_params.get('grid', 100)
        self.maxshrink = weight_quant_params.get('maxshrink', 0.8)
        
        if self.sym:
            self.minq = -(2 ** (self.bits - 1))
            self.maxq = 2**(self.bits-1)-1
        else:
            self.minq = 0.0
            self.maxq = 2**self.bits - 1
        if self.bits == 1:
            self.minq = 0.0
            self.maxq = 1.0

    def find_params(self, x: torch.Tensor) -> None:
        """
        Find optimal quantization parameters for input tensor.

        Args:
            x: Input weight tensor
        """
        self.per_token_dynamic_calibration(x)

    def round_ste(self, x: torch.Tensor) -> torch.Tensor:
        """
        Straight-Through Estimator for rounding operation.
        
        Args:
            x: Input tensor
            
        Returns:
            Rounded tensor with gradient pass-through
        """
        return (x.round() - x).detach() + x
    
    def per_token_dynamic_calibration(self, x: torch.Tensor) -> None:
        """
        Compute scale and zero point using dynamic calibration.
        
        Args:
            x: Input weight tensor
        """
        dev = x.device
        if self.group_size != -1:
            # Reshape for per-group weight quantization
            x = x.reshape(-1, self.group_size)

        if self.bits == 1:
            # Binary quantization: q in {0, 1}, dequant = scale * (q - 0.5).
            # This yields two centroids at +/- mean(abs(x)) per row/group.
            scale = 2 * x.abs().mean(dim=-1, keepdim=True).to(torch.float32)
            self.scale = scale.clamp(min=1e-5, max=1e4)
            self.zero = torch.full_like(self.scale, 0.5)
            return

        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True).to(torch.float32)
        xmax = x.amax(reduce_shape, keepdim=True).to(torch.float32)

        if self.sym:
            abs_max = torch.max(xmax.abs(), xmin.abs())
            scale = abs_max / (2**(self.bits-1)-1)
            self.scale = scale.clamp(min=1e-5, max=1e4)
            self.zero = torch.zeros_like(self.scale)
        else:
            xrange = xmax - xmin
            scale = xrange / (2**self.bits-1)
            self.scale = scale.clamp(min=1e-5, max=1e4)
            zero_point = -(xmin) / (self.scale)
            self.zero = zero_point.clamp(min=-1e4, max=1e4).round()

        self.scale = self.scale.squeeze(1)
        self.zero = self.zero.squeeze(1)

        if self.mse:
            # Grid search for optimal scale
            xmin = xmin.squeeze(1)
            xmax = xmax.squeeze(1)
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:
                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]

        self.scale = self.scale.reshape(-1, 1)
        self.zero = self.zero.reshape(-1, 1)

    def fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform fake quantization (quantize + dequantize).

        Args:
            x: Input tensor

        Returns:
            Fake quantized tensor
        """
        assert not (self.perchannel and self.group_size != -1)
        
        if self.group_size != -1:
            assert len(x.shape) == 2
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)

        if self.bits == 1:
            x_dequant = binary_quant_dequant(x, self.scale, self.zero)
            if self.group_size != -1:
                x_dequant = x_dequant.reshape(dim1, dim2)
            return x_dequant.to(x.dtype)

        x_dtype = x.dtype
        x_int = self.round_ste(x / self.scale)
        x_int = x_int.add(self.zero)
        x_int = x_int.clamp(self.minq, self.maxq)
        x_dequant = x_int
        x_dequant = x_dequant.sub(self.zero)
        x_dequant = x_dequant.mul(self.scale)

        if self.group_size != -1:
            x_dequant = x_dequant.reshape(dim1, dim2)

        return x_dequant.to(x_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: find parameters and apply fake quantization.
        
        Args:
            x: Input weight tensor
            
        Returns:
            Fake quantized tensor
        """
        if self.bits >= 16:
            return x
        
        self.per_token_dynamic_calibration(x)
        return self.fake_quant(x)
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize and dequantize weights (used by GPTQ).
        
        Args:
            x: Input weight tensor
            
        Returns:
            Fake quantized tensor
        """
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.bits == 1:
                return binary_quant_dequant(x, self.scale, self.zero).to(x_dtype)
            if self.sym:
                return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)
        return x

    def free(self) -> None:
        """Free memory used by quantizer."""
        if self.scale is not None:
            del self.scale
            self.scale = None
        if self.zero is not None:
            del self.zero
            self.zero = None

    def ready(self) -> bool:
        """Check if quantizer has computed parameters."""
        return self.scale is not None and self.zero is not None
    
    def register_scales_and_zeros(self) -> None:
        """Register scale and zero as buffers for saving."""
        if self.scale is not None:
            self.register_buffer('scales', self.scale)
        if self.zero is not None:
            self.register_buffer('zeros', self.zero)


def find_qlayers(
    module: nn.Module, 
    layers: Optional[List[type]] = None, 
    name: str = ''
) -> Dict[str, nn.Module]:
    """
    Recursively find all layers of specified types in a module.
    
    Args:
        module: Module to search
        layers: List of layer types to find (default: [nn.Linear])
        name: Current module name prefix
        
    Returns:
        Dictionary mapping full layer names to layer modules
    """
    if layers is None:
        layers = [nn.Linear]
        
    if type(module) in layers:
        return {name: module}
    
    res = {}
    for name1, child in module.named_children():
        full_name = name + '.' + name1 if name != '' else name1
        res.update(find_qlayers(child, layers=layers, name=full_name))
    return res
