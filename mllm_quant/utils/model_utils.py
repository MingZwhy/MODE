"""
Utility functions for model manipulation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set, Tuple


def find_layers(
    module: nn.Module,
    layers: Optional[List[type]] = None,
    name: str = "",
) -> Dict[str, nn.Module]:
    """
    Find all layers of specified types in a module.
    
    Args:
        module: The module to search
        layers: List of layer types to find (default: [nn.Linear])
        name: Current module name prefix
        
    Returns:
        Dictionary mapping layer names to layer modules
    """
    if layers is None:
        layers = [nn.Linear]
    
    res = {}
    for child_name, child in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        if type(child) in layers:
            res[full_name] = child
        else:
            res.update(find_layers(child, layers, full_name))
    
    return res


def get_named_linears(
    module: nn.Module,
    name_prefix: str = "",
) -> Dict[str, nn.Linear]:
    """
    Get all named Linear layers in a module.
    
    Args:
        module: The module to search
        name_prefix: Prefix for layer names
        
    Returns:
        Dictionary mapping layer names to Linear layers
    """
    return find_layers(module, [nn.Linear], name_prefix)


def set_op_by_name(
    module: nn.Module,
    name: str,
    new_module: nn.Module,
) -> None:
    """
    Set a submodule by its full name path.
    
    Args:
        module: The parent module
        name: Full name path (e.g., "layer.0.mlp.fc1")
        new_module: The new module to set
    """
    parts = name.split(".")
    parent = module
    
    for part in parts[:-1]:
        parent = getattr(parent, part)
    
    setattr(parent, parts[-1], new_module)


def get_op_by_name(
    module: nn.Module,
    name: str,
) -> nn.Module:
    """
    Get a submodule by its full name path.
    
    Args:
        module: The parent module
        name: Full name path (e.g., "layer.0.mlp.fc1")
        
    Returns:
        The submodule
    """
    parts = name.split(".")
    result = module
    
    for part in parts:
        result = getattr(result, part)
    
    return result


def is_moe_router(name: str) -> bool:
    """
    Check if a layer name corresponds to a MoE router layer.
    Router layers should NOT be quantized.
    
    Args:
        name: Layer name
        
    Returns:
        True if the layer is a MoE router
    """
    # Common router naming patterns
    router_patterns = [
        "gate",        # Qwen3-VL-MoE uses "gate" for router
        "router",      # Common pattern
        "TopKRouter",  # Explicit router class name
        "moe.gate",    # MoE gate pattern
        ".gate.",      # Gate in path
    ]
    
    name_lower = name.lower()
    for pattern in router_patterns:
        if pattern.lower() in name_lower:
            return True
    
    return False


def get_layers_to_quantize(
    model: nn.Module,
    skip_patterns: Optional[List[str]] = None,
) -> Dict[str, nn.Linear]:
    """
    Get all Linear layers that should be quantized.
    Excludes MoE routers and layers matching skip patterns.
    
    Args:
        model: The model
        skip_patterns: Additional patterns to skip (substrings in layer names)
        
    Returns:
        Dictionary mapping layer names to Linear layers
    """
    if skip_patterns is None:
        skip_patterns = []
    
    all_linears = get_named_linears(model)
    
    layers_to_quantize = {}
    for name, layer in all_linears.items():
        # Skip router layers
        if is_moe_router(name):
            print(f"Skipping MoE router layer: {name}")
            continue
        
        # Skip layers matching skip patterns
        skip = False
        for pattern in skip_patterns:
            if pattern in name:
                print(f"Skipping layer matching pattern '{pattern}': {name}")
                skip = True
                break
        
        if not skip:
            layers_to_quantize[name] = layer
    
    return layers_to_quantize


def get_model_layers(model: nn.Module) -> List[nn.Module]:
    """
    Get the list of transformer decoder layers from the model.
    Handles different model architectures.
    
    Args:
        model: The model
        
    Returns:
        List of decoder layer modules
    """
    # Try different attribute paths based on model architecture
    layer_paths = [
        "model.language_model.layers",      # Qwen3-VL, Qwen3-VL-MoE
        "model.layers",                      # Common LLM pattern
        "language_model.model.layers",      # Alternative pattern
        "transformer.h",                     # GPT-style models
    ]
    
    for path in layer_paths:
        try:
            layers = model
            for attr in path.split("."):
                layers = getattr(layers, attr)
            return list(layers)
        except AttributeError:
            continue
    
    raise ValueError("Could not find transformer layers in model")


def get_embed_layer(model: nn.Module) -> nn.Module:
    """
    Get the embedding layer from the model.
    
    Args:
        model: The model
        
    Returns:
        The embedding layer
    """
    embed_paths = [
        "model.language_model.embed_tokens",
        "model.embed_tokens",
        "transformer.wte",
        "model.model.embed_tokens",
    ]
    
    for path in embed_paths:
        try:
            layer = model
            for attr in path.split("."):
                layer = getattr(layer, attr)
            return layer
        except AttributeError:
            continue
    
    raise ValueError("Could not find embedding layer in model")


def move_to_device(data: Dict, device: torch.device) -> Dict:
    """
    Move all tensors in a dictionary to a device.
    
    Args:
        data: Dictionary containing tensors
        device: Target device
        
    Returns:
        Dictionary with tensors moved to device
    """
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in data.items()
    }


def get_device(model: nn.Module) -> torch.device:
    """
    Get the device of a model.
    
    Args:
        model: The model
        
    Returns:
        The device
    """
    return next(model.parameters()).device
