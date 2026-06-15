"""
GPTQ and RTN quantization utilities.
Supports: Qwen3-VL, Qwen3-VL-MoE, InternVL, Kimi-VL models.

This module provides:
- GPTQ quantization with Hessian-based optimization
- RTN (Round-To-Nearest) quantization
- Mixed-precision quantization support
"""

import math
import time
import copy
import logging
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any

from .quant_utils import WeightQuantizer, find_qlayers

from mllm_quant.rotation.quarot_gptq_compat import quarot_kimi_expert_mid_pre_down, quarot_moe_mid_pre_down
from mllm_quant.expert_mixed_precision.bit_config import get_expert_bit


# ============================================================================
# Model type constants
# ============================================================================
QWEN3_VL_MODEL = "Qwen3VLForConditionalGeneration"
QWEN3_VL_MOE_MODEL = "Qwen3VLMoeForConditionalGeneration"
INTERNVL_MODEL = "InternVLForConditionalGeneration"
KIMIVL_MODEL = "KimiVLForConditionalGeneration"


def get_model_type(model) -> str:
    """Get model type string from model class name."""
    return model.__class__.__name__


def cleanup_memory(verbos: bool = True):
    """Clean up GPU and CPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    if verbos:
        logging.info("Memory cleaned up")


def strip_accelerate_dispatch_hooks(model: nn.Module) -> None:
    """
    Remove Accelerate ``AlignDeviceHook`` from models loaded with ``device_map='auto'``.

    GPTQ calibration does ``model.cpu()`` then moves e.g. ``language_model`` to ``cuda:0``.
    If dispatch hooks remain, forwards can still place tensors (e.g. ``position_ids``) on the
    old mapped GPU while ``rotary_emb`` buffers sit on ``cuda:0``, breaking RoPE matmul.
    Extra forwards hooks (e.g. QuaRot pre-hooks) make this more likely to surface.
    """
    try:
        from accelerate.hooks import remove_hook_from_module

        remove_hook_from_module(model, recurse=True)
    except Exception:
        pass


# ============================================================================
# MoE layer detection and handling
# ============================================================================
def is_moe_layer(layer) -> bool:
    """
    Check if a decoder layer uses MoE (SparseMoeBlock) or dense MLP.
    
    Supported MoE structures:
    - Qwen3-VL-MoE: Qwen3VLMoeTextSparseMoeBlock with 3D Parameter tensors
    - DeepseekV3MoE (Kimi-VL): DeepseekV3MoE with 3D Parameter tensors (DeepseekV3NaiveMoe)
    - InternVL: ModuleList of experts
    """
    if hasattr(layer, 'mlp'):
        mlp = layer.mlp
        # Check for SparseMoeBlock structure: has 'experts' and 'gate'
        if hasattr(mlp, 'experts') and hasattr(mlp, 'gate'):
            return True
    return False


def get_moe_expert_params(layer) -> Dict[str, nn.Parameter]:
    """
    Get MoE expert parameters from a layer.
    
    Supported MoE structures:
    - Qwen3VLMoeTextExperts / DeepseekV3NaiveMoe (3D Parameter):
      - gate_up_proj: (num_experts, 2 * intermediate_dim, hidden_dim)
      - down_proj: (num_experts, hidden_dim, intermediate_dim)
    
    Returns:
        Dictionary mapping parameter names to Parameter objects
    """
    params = {}
    if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
        experts = layer.mlp.experts
        if hasattr(experts, 'gate_up_proj'):
            params['mlp.experts.gate_up_proj'] = experts.gate_up_proj
        if hasattr(experts, 'down_proj'):
            params['mlp.experts.down_proj'] = experts.down_proj
    return params


def get_num_experts_from_layer(layer) -> int:
    """
    Get number of experts from a MoE layer.
    
    Supports:
    - 3D Parameter structure (Qwen3-VL-MoE, DeepseekV3MoE/Kimi-VL)
    - ModuleList structure (InternVL, etc.)
    """
    if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
        experts = layer.mlp.experts
        if hasattr(experts, 'num_experts'):
            return experts.num_experts
        if hasattr(experts, 'gate_up_proj'):
            # 3D Parameter structure (Qwen3-VL-MoE, DeepseekV3MoE/Kimi-VL)
            return experts.gate_up_proj.shape[0]
        if isinstance(experts, nn.ModuleList):
            # ModuleList structure (InternVL, etc.)
            return len(experts)
    return 0


def is_moe_3d_param_structure(experts) -> bool:
    """
    Check if experts use 3D Parameter structure.
    
    Supported: Qwen3-VL-MoE, DeepseekV3MoE (Kimi-VL)
    """
    return hasattr(experts, 'gate_up_proj') and hasattr(experts, 'down_proj')


def is_moe_modulelist_structure(experts) -> bool:
    """Check if experts use ModuleList structure (standard MoE style)."""
    return isinstance(experts, nn.ModuleList) and len(experts) > 0


def move_to_device(tensor_or_tuple, device):
    """Move tensor or tuple of tensors to specified device."""
    if tensor_or_tuple is None:
        return None
    if isinstance(tensor_or_tuple, torch.Tensor):
        return tensor_or_tuple.to(device)
    if isinstance(tensor_or_tuple, (tuple, list)):
        return type(tensor_or_tuple)(
            t.to(device) if isinstance(t, torch.Tensor) else t 
            for t in tensor_or_tuple
        )
    return tensor_or_tuple


def prepare_layer_kwargs(cache, sample_idx, device):
    """Prepare attention_mask, position_ids, position_embeddings for a layer forward pass."""
    attn_mask = move_to_device(cache['attention_masks'][sample_idx], device)
    pos_ids = move_to_device(cache['position_ids_list'][sample_idx], device)
    pos_emb = None
    if cache['position_embeddings_list']:
        pos_emb = move_to_device(cache['position_embeddings_list'][sample_idx], device)
    return attn_mask, pos_ids, pos_emb


def _set_expert_bit(
    params: Dict,
    expert_bit_map: Optional[Dict[int, Dict[int, int]]],
    layer_idx: int,
    expert_idx: int,
    default_bits: int,
) -> Dict:
    """Return a quantizer config with optional expert-level bit override."""
    out = copy.deepcopy(params)
    out["w_bits"] = get_expert_bit(expert_bit_map, layer_idx, expert_idx, default_bits)
    if out.get("w_groupsize", -1) != -1:
        out["perchannel"] = False
    return out


def _gptq_build_hinv(
    H: torch.Tensor,
    percdamp: float,
    *,
    cholesky_autodamp_retry: bool = False,
    cholesky_max_retries: int = 12,
    log_prefix: str = "",
) -> torch.Tensor:
    """
    Cholesky + inverse chain used by GPTQ ``fasterquant`` on Hessian ``H``.

    When ``cholesky_autodamp_retry`` is True, on non-PD Cholesky failures the diagonal
    damping is multiplied by 2 and retried (up to ``cholesky_max_retries`` extra attempts
    after the first), matching standard GPTQ jitter for ill-conditioned H.
    """
    n = H.shape[0]
    dev = H.device
    diag = torch.arange(n, device=dev)
    base_damp = percdamp * torch.mean(torch.diag(H))
    if not torch.isfinite(base_damp) or base_damp <= 0:
        base_damp = torch.tensor(1e-6, device=dev, dtype=H.dtype)

    def _one_chol_chain(damp_mult: float) -> torch.Tensor:
        H_try = H.clone()
        H_try[diag, diag] += base_damp * damp_mult
        L = torch.linalg.cholesky(H_try)
        Hi = torch.cholesky_inverse(L)
        return torch.linalg.cholesky(Hi, upper=True)

    damp_mult = 1.0
    for attempt in range(cholesky_max_retries + 1):
        try:
            Hinv = _one_chol_chain(damp_mult)
            if cholesky_autodamp_retry and attempt > 0 and log_prefix:
                bd = float(base_damp.item()) if hasattr(base_damp, "item") else float(base_damp)
                logging.warning(
                    f"{log_prefix}Cholesky OK after {attempt} damp retries "
                    f"(damp_mult={damp_mult:.4g} vs base_damp={bd:.4g})"
                )
            return Hinv
        except Exception as e:
            msg = str(e).lower()
            is_pd_fail = (
                "positive-definite" in msg
                or "positive definite" in msg
                or "not positive definite" in msg
            )
            if cholesky_autodamp_retry and attempt < cholesky_max_retries and is_pd_fail:
                damp_mult *= 2.0
                if log_prefix:
                    logging.warning(
                        f"{log_prefix}Cholesky failed ({e!s}); retry with damp_mult={damp_mult:.4g}"
                    )
                continue
            raise


# ============================================================================
# GPTQ Quantizer class for nn.Linear
# ============================================================================
class GPTQ:
    """
    GPTQ quantizer for a single linear layer.
    
    Implements the GPTQ algorithm that minimizes quantization error
    using second-order information (Hessian).
    """

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer = None

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        """
        Add a batch of input activations to compute Hessian.
        
        Args:
            inp: Input tensor
            out: Output tensor (not used, for interface compatibility)
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, 
        blocksize: int = 128, 
        percdamp: float = 0.01, 
        groupsize: int = -1, 
        actorder: bool = False, 
        static_groups: bool = False,
        cholesky_autodamp_retry: bool = False,
        cholesky_max_retries: int = 12,
        cholesky_log_prefix: str = "",
    ) -> None:
        """
        Perform GPTQ quantization on the layer weights.
        
        Args:
            blocksize: Block size for block-wise quantization
            percdamp: Damping percentage for Hessian
            groupsize: Group size for group-wise quantization (-1 for per-channel)
            actorder: Use descending activation order
            static_groups: Use static groups (precompute quantizer for each group)
            cholesky_autodamp_retry: On Cholesky failure, double effective damp and retry
            cholesky_max_retries: Max number of retries (each retry doubles damp multiplier)
            cholesky_log_prefix: Optional log prefix for retry messages
        """
        W = self.layer.weight.data.clone()
        W = W.float()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                group_end = min(i + groupsize, self.columns)
                if group_end - i != groupsize:
                    quantizer.group_size = -1
                quantizer.find_params(W[:, i:group_end])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        Hinv = _gptq_build_hinv(
            H,
            percdamp,
            cholesky_autodamp_retry=cholesky_autodamp_retry,
            cholesky_max_retries=cholesky_max_retries,
            log_prefix=cholesky_log_prefix,
        )

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            group_start = i1 + i
                            group_end = min(group_start + groupsize, self.columns)
                            original_group_size = self.quantizer.group_size
                            if group_end - group_start != groupsize:
                                self.quantizer.group_size = -1
                            self.quantizer.find_params(W[:, group_start:group_end])
                            self.quantizer.group_size = original_group_size
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            raise ValueError('NaN in weights')

    def free(self):
        """Free memory used by the quantizer."""
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        cleanup_memory(verbos=False)


# ============================================================================
# GPTQ Quantizer class for 2D weight tensor (MoE experts)
# ============================================================================
class GPTQWeight:
    """
    GPTQ quantizer for a 2D weight tensor (used for MoE expert weights).
    
    Unlike GPTQ class which works with nn.Linear, this class works directly
    with 2D weight tensors.
    
    For MoE experts, weight shape is (out_features, in_features).
    The Hessian is computed based on in_features dimension.
    """

    def __init__(self, weight: torch.Tensor, in_features: int = None):
        """
        Args:
            weight: 2D weight tensor of shape (out_features, in_features)
            in_features: Override input features dimension (useful when weight layout differs)
        """
        self.weight = weight
        self.dev = weight.device
        self.rows = weight.shape[0]
        # For standard Linear: weight is (out, in), so columns = in_features = shape[1]
        self.columns = in_features if in_features is not None else weight.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer = None

    def add_batch(self, inp: torch.Tensor) -> None:
        """
        Add a batch of input activations to compute Hessian.
        
        Args:
            inp: Input tensor of shape (batch, seq_len, in_features) or (tokens, in_features)
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        
        # Validate dimensions
        if inp.shape[-1] != self.columns:
            raise ValueError(f"Input dimension {inp.shape[-1]} does not match expected {self.columns}")
        
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, 
        blocksize: int = 128, 
        percdamp: float = 0.01, 
        groupsize: int = -1, 
        actorder: bool = False, 
        static_groups: bool = False,
        cholesky_autodamp_retry: bool = False,
        cholesky_max_retries: int = 12,
        cholesky_log_prefix: str = "",
    ) -> torch.Tensor:
        """
        Perform GPTQ quantization on the weight tensor.
        
        Returns:
            Quantized weight tensor
        """
        W = self.weight.clone().float()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        Hinv = _gptq_build_hinv(
            H,
            percdamp,
            cholesky_autodamp_retry=cholesky_autodamp_retry,
            cholesky_max_retries=cholesky_max_retries,
            log_prefix=cholesky_log_prefix,
        )

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        return Q.to(self.weight.dtype)

    def free(self):
        """Free memory used by the quantizer."""
        self.H = None
    torch.cuda.empty_cache()


# ============================================================================
# MoE token balancing helper functions
# ============================================================================
def balance_expert_tokens(
    expert_inputs: Dict[int, List[torch.Tensor]],
    all_layer_tokens: torch.Tensor,
    num_experts: int,
    top_k: int,
    keep_min: bool,
    percentage: float,
    dev: torch.device,
) -> Dict[int, torch.Tensor]:
    """
    Balance tokens across experts by padding experts with fewer tokens.
    
    Note: Padded tokens are only used for calibration (quantization), not for re-forward.
    Re-forward uses original routing logic.
    
    Args:
        expert_inputs: Dictionary mapping expert_idx to list of input tensors
        all_layer_tokens: All tokens from the layer (shape: [total_tokens, hidden_dim])
        num_experts: Number of experts
        top_k: Number of experts selected per token
        keep_min: Whether to enable minimum token balancing
        percentage: Minimum percentage of average tokens per expert
        dev: Device for tensor operations
    
    Returns:
        Dictionary mapping expert_idx to balanced input tensor
    """
    # Calculate total tokens and average tokens per expert
    total_tokens = all_layer_tokens.shape[0]
    avg_tokens_per_expert = (total_tokens * top_k) / num_experts
    min_tokens = int(avg_tokens_per_expert * percentage)

    balanced_inputs = {}
    padded_experts = set()  # Track which experts were padded
    
    for expert_idx in range(num_experts):
        if expert_inputs[expert_idx]:
            expert_tokens = torch.cat(expert_inputs[expert_idx], dim=0)
            num_expert_tokens = expert_tokens.shape[0]
        else:
            num_expert_tokens = 0
        
        if keep_min and num_expert_tokens < min_tokens:
            # Need to pad: randomly sample tokens from all_layer_tokens
            num_to_pad = min_tokens - num_expert_tokens
            if num_expert_tokens > 0:
                # Concatenate existing tokens with randomly sampled tokens
                random_indices = torch.randint(0, total_tokens, (num_to_pad,), device=dev)
                padded_tokens = all_layer_tokens[random_indices]
                balanced_inputs[expert_idx] = torch.cat([expert_tokens, padded_tokens], dim=0)
                logging.info(f"    Expert {expert_idx}: padded from {num_expert_tokens} to {min_tokens} tokens (for calibration only)")
            else:
                # No existing tokens, use all random samples
                random_indices = torch.randint(0, total_tokens, (min_tokens,), device=dev)
                balanced_inputs[expert_idx] = all_layer_tokens[random_indices]
                logging.info(f"    Expert {expert_idx}: padded from 0 to {min_tokens} tokens (for calibration only)")
            padded_experts.add(expert_idx)
        else:
            # No padding needed
            if num_expert_tokens > 0:
                balanced_inputs[expert_idx] = torch.cat(expert_inputs[expert_idx], dim=0)
            else:
                # Still no tokens, but keep_min is False or percentage is 0
                balanced_inputs[expert_idx] = None
    
    return balanced_inputs


# ============================================================================
# Sequential layer groups generation
# ============================================================================
def generate_sequential_for_layer(layer) -> List[List[str]]:
    """
    Generate sequential layer groups for quantization based on actual layer structure.
    
    Quantization is performed in sequence: qkv -> o_proj -> up/gate -> down
    This ensures proper activation collection for GPTQ.
    
    Attention layer structures:
    - Standard (Qwen3, Llama): q_proj, k_proj, v_proj (can be quantized together)
    - DeepseekV3 (Kimi-VL): q_proj, kv_a_proj_with_mqa, kv_b_proj (can be quantized together)
    
    For MoE layers (Qwen3VLMoeTextSparseMoeBlock):
    - experts use 3D Parameter tensors, handled separately
    - Sequential only includes attention layers
    
    For Dense MLP layers (Qwen3VLMoeTextMLP):
    - Standard nn.Linear layers: gate_proj, up_proj, down_proj
    
    Args:
        layer: The decoder layer to analyze
        
    Returns:
        List of layer name groups to be quantized sequentially
    """
    # Detect attention structure
    attn = layer.self_attn if hasattr(layer, 'self_attn') else None
    
    # Check for DeepseekV3Attention structure (Kimi-VL)
    if attn is not None and hasattr(attn, 'kv_a_proj_with_mqa') and hasattr(attn, 'kv_b_proj'):
        # DeepseekV3Attention: 
        # - q_proj and kv_a_proj_with_mqa share the same input (hidden_states), can be quantized together
        # - kv_b_proj's input is the output of kv_a_proj_with_mqa, needs re-forward after quantizing q_proj+kv_a_proj_with_mqa
        # - o_proj's input depends on all previous quantized layers, needs re-forward after quantizing kv_b_proj
        sequential = [
            ['self_attn.q_proj', 'self_attn.kv_a_proj_with_mqa'],
            ['self_attn.kv_b_proj'],
            ['self_attn.o_proj'],
        ]
    else:
        # Standard attention: q_proj, k_proj, v_proj can be quantized together
        sequential = [
            ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
            ['self_attn.o_proj'],
        ]
    
    # Check if this is a MoE layer or dense MLP layer
    if is_moe_layer(layer):
        # MoE layer: experts are 3D Parameters, handled separately
        # Only attention layers are quantized via sequential
        pass
    else:
        # Dense MLP layer: standard nn.Linear layers
        sequential.append(['mlp.up_proj', 'mlp.gate_proj'])
        sequential.append(['mlp.down_proj'])
    
    return sequential


def _fix_vlm_batch_dims(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure VLM input tensors have batch dimension."""
    if 'input_ids' in inputs and inputs['input_ids'].dim() == 1:
        inputs['input_ids'] = inputs['input_ids'].unsqueeze(0)
    if 'attention_mask' in inputs and inputs['attention_mask'].dim() == 1:
        inputs['attention_mask'] = inputs['attention_mask'].unsqueeze(0)
    for grid_key in ('image_grid_thw', 'video_grid_thw', 'image_grid_hws'):
        if grid_key in inputs and isinstance(inputs[grid_key], torch.Tensor) and inputs[grid_key].dim() == 1:
            inputs[grid_key] = inputs[grid_key].unsqueeze(0)
    return inputs


class _VisionCaptureStop(Exception):
    pass


def _to_cpu_detached(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, tuple):
        return tuple(_to_cpu_detached(x) for x in obj)
    if isinstance(obj, list):
        return [_to_cpu_detached(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_cpu_detached(v) for k, v in obj.items()}
    return obj


def _to_device(obj, dev: torch.device):
    if isinstance(obj, torch.Tensor):
        return obj.to(dev)
    if isinstance(obj, tuple):
        return tuple(_to_device(x, dev) for x in obj)
    if isinstance(obj, list):
        return [_to_device(x, dev) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_device(v, dev) for k, v in obj.items()}
    return obj


def _run_vlm_calibration_until_text(
    model: nn.Module,
    dataloader: List,
    layers,
    dev: torch.device,
    desc: Optional[str] = None,
) -> None:
    """
    Run full VLM calibration batches and stop at the first language layer.

    Vision modules execute normally before the first text decoder layer, so hooks on
    vision Linear layers can collect GPTQ activations without running the full LLM.
    """

    class StopForward(Exception):
        pass

    class StopCatcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            raise StopForward

    layers[0] = StopCatcher(layers[0].to(dev))
    iterator = tqdm(dataloader, desc=desc, unit="batch") if desc else dataloader
    try:
        for batch in iterator:
            try:
                if isinstance(batch, dict):
                    inputs = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    inputs = _fix_vlm_batch_dims(inputs)
                    model(**inputs)
                elif isinstance(batch, (tuple, list)):
                    input_ids = batch[0]
                    if isinstance(input_ids, dict):
                        inputs = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in input_ids.items()}
                        inputs = _fix_vlm_batch_dims(inputs)
                        model(**inputs)
                    else:
                        input_ids = input_ids.to(dev)
                        if input_ids.dim() == 1:
                            input_ids = input_ids.unsqueeze(0)
                        model(input_ids)
                else:
                    input_ids = batch.to(dev)
                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)
                    model(input_ids)
            except StopForward:
                pass
    finally:
        layers[0] = layers[0].module


def _capture_vision_block_inputs(
    model: nn.Module,
    dataloader: List,
    layers,
    block: nn.Module,
    dev: torch.device,
    desc: str,
) -> List[Tuple[torch.Tensor, Tuple[Any, ...], Dict[str, Any]]]:
    """Capture inputs to one vision block using full VLM forward up to that block."""

    samples = []

    def pre_hook(_, args, kwargs):
        if not args:
            return
        hidden_states = _to_cpu_detached(args[0])
        extra_args = _to_cpu_detached(args[1:])
        extra_kwargs = _to_cpu_detached(kwargs)
        samples.append((hidden_states, extra_args, extra_kwargs))
        raise _VisionCaptureStop

    handle = block.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        iterator = tqdm(dataloader, desc=desc, unit="batch")
        for batch in iterator:
            try:
                if isinstance(batch, dict):
                    inputs = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    inputs = _fix_vlm_batch_dims(inputs)
                    model(**inputs)
                elif isinstance(batch, (tuple, list)):
                    input_ids = batch[0]
                    if isinstance(input_ids, dict):
                        inputs = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in input_ids.items()}
                        inputs = _fix_vlm_batch_dims(inputs)
                        model(**inputs)
                    else:
                        input_ids = input_ids.to(dev)
                        if input_ids.dim() == 1:
                            input_ids = input_ids.unsqueeze(0)
                        model(input_ids)
                else:
                    input_ids = batch.to(dev)
                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)
                    model(input_ids)
            except _VisionCaptureStop:
                pass
    finally:
        handle.remove()

    if not samples:
        raise RuntimeError(
            f"No inputs captured for {desc}. Check that calibration batches contain valid images."
        )
    return samples


def _get_vision_quant_plan(model: nn.Module):
    """
    Return supported vision blocks and projector modules for Qwen3-VL-MoE / Kimi-VL.
    """
    if hasattr(model, 'model') and hasattr(model.model, 'visual'):
        vision = model.model.visual
        if hasattr(vision, 'blocks'):
            extras = []
            if hasattr(vision, 'merger'):
                extras.append(('model.visual.merger', vision.merger))
            if hasattr(vision, 'deepstack_merger_list'):
                for idx, merger in enumerate(vision.deepstack_merger_list):
                    extras.append((f'model.visual.deepstack_merger_list.{idx}', merger))
            return vision, vision.blocks, 'model.visual.blocks', extras

    if hasattr(model, 'vision_tower'):
        vision = model.vision_tower
        if hasattr(vision, 'encoder') and hasattr(vision.encoder, 'blocks'):
            extras = []
            if hasattr(model, 'multi_modal_projector'):
                extras.append(('multi_modal_projector', model.multi_modal_projector))
            return vision, vision.encoder.blocks, 'vision_tower.encoder.blocks', extras

    return None, None, None, []


def _vision_block_sequential(block: nn.Module) -> List[List[str]]:
    """Generate GPTQ order for supported vision transformer blocks."""
    if hasattr(block, 'attn'):
        return [
            ['attn.qkv'],
            ['attn.proj'],
            ['mlp.linear_fc1'],
            ['mlp.linear_fc2'],
        ]
    if hasattr(block, 'wqkv'):
        return [
            ['wqkv'],
            ['wo'],
            ['mlp.fc0'],
            ['mlp.fc1'],
        ]
    return [[name] for name, _ in block.named_modules() if name and isinstance(_, nn.Linear)]


def _vision_projector_sequential(module: nn.Module) -> List[List[str]]:
    """Generate GPTQ order for vision output merger/projector MLPs."""
    candidates = [
        ['linear_fc1'],
        ['linear_fc2'],
        ['linear_1'],
        ['linear_2'],
        ['fc0'],
        ['fc1'],
    ]
    full = find_qlayers(module, layers=[torch.nn.Linear])
    return [names for names in candidates if any(name in full for name in names)]


@torch.no_grad()
def quantize_vision_encoder_gptq(
    model: nn.Module,
    dataloader: List,
    layers,
    dev: torch.device,
    weight_quant_params: Dict,
    vision_bits: int,
    quantize_projector: bool = False,
) -> Dict[str, WeightQuantizer]:
    """
    GPTQ quantize supported VLM vision encoders before LLM layer-wise GPTQ.

    Supported now:
    - Qwen3-VL-MoE / Qwen3-VL: model.model.visual.blocks (+ optional merger / deepstack mergers)
    - Kimi-VL: model.vision_tower.encoder.blocks (+ optional multi_modal_projector)

    Args:
        quantize_projector: If True, also quantize the vision projector / merger modules
            after the transformer blocks. If False (default), only the transformer blocks
            are quantized; projector / merger stay in original precision.
    """
    if vision_bits >= 16:
        logging.info("Vision GPTQ: vision_bits>=16, skip vision quantization")
        return {}

    vision, blocks, block_prefix, extra_modules = _get_vision_quant_plan(model)
    if vision is None or blocks is None:
        logging.warning("Vision GPTQ requested, but no supported vision encoder was found")
        return {}

    quantizers = {}
    vision = vision.to(dev)
    for _, module in extra_modules:
        module.to(dev)

    vision_wqp = copy.deepcopy(weight_quant_params)
    vision_wqp['w_bits'] = vision_bits

    logging.info(
        f"-----Vision Encoder GPTQ Quantization ({vision_bits} bits, {len(blocks)} blocks)-----"
    )

    def make_gptq(linear: nn.Linear, qname: str) -> GPTQ:
        logging.info(f"  {qname} ({vision_bits} bits)")
        gptq_obj = GPTQ(linear)
        if hasattr(linear, "weight_quantizer"):
            gptq_obj.quantizer = linear.weight_quantizer
        else:
            gptq_obj.quantizer = WeightQuantizer()
        gptq_obj.quantizer.configure(vision_wqp)
        return gptq_obj

    def finish_gptq(gptq_obj: GPTQ, qname: str):
        if gptq_obj.nsamples == 0:
            gptq_obj.free()
            raise RuntimeError(f"No vision calibration activations collected for {qname}")
        logging.info(f"  GPTQ quantizing {qname} ({vision_bits} bits)")
        gptq_obj.fasterquant(
            percdamp=weight_quant_params['percdamp'],
            groupsize=weight_quant_params['w_groupsize'],
            actorder=weight_quant_params.get('act_order', False),
            static_groups=False,
        )
        quantizers[qname] = gptq_obj.quantizer
        gptq_obj.free()

    def qwen_attn_forward(block: nn.Module, hidden_states: torch.Tensor, extra_args, extra_kwargs):
        return block.attn(
            block.norm1(hidden_states),
            *_to_device(extra_args, hidden_states.device),
            **_to_device(extra_kwargs, hidden_states.device),
        )

    def kimi_attn_forward(block: nn.Module, hidden_states: torch.Tensor, extra_args, extra_kwargs):
        return block.attention_qkvpacked(
            block.norm0(hidden_states),
            *_to_device(extra_args, hidden_states.device),
            **_to_device(extra_kwargs, hidden_states.device),
        )

    def quantize_qwen_vision_block(
        block: nn.Module,
        block_samples: List[Tuple[torch.Tensor, Tuple[Any, ...], Dict[str, Any]]],
        prefix: str,
    ) -> List[Tuple[torch.Tensor, Tuple[Any, ...], Dict[str, Any]]]:
        qkv_name = f"{prefix}.attn.qkv"
        gptq_qkv = make_gptq(block.attn.qkv, qkv_name)
        logging.info(f"  [reforward] {qkv_name}: collect input via norm1 -> qkv (fp)")
        for hidden_states, _, _ in tqdm(block_samples, desc=f"Vision GPTQ {qkv_name}", unit="batch"):
            hs = hidden_states.to(dev)
            qkv_inp = block.norm1(hs)
            qkv_out = block.attn.qkv(qkv_inp)
            gptq_qkv.add_batch(qkv_inp, qkv_out)
            del hs, qkv_inp, qkv_out
        finish_gptq(gptq_qkv, qkv_name)

        proj_name = f"{prefix}.attn.proj"
        gptq_proj = make_gptq(block.attn.proj, proj_name)
        logging.info(
            f"  [reforward] {proj_name}: re-run norm1 -> qkv(quant) -> attn -> proj to recompute proj input"
        )
        handle = block.attn.proj.register_forward_hook(
            lambda _, inp, out: gptq_proj.add_batch(inp[0].data, out.data)
        )
        try:
            for hidden_states, extra_args, extra_kwargs in tqdm(
                block_samples, desc=f"Vision GPTQ {proj_name}", unit="batch"
            ):
                hs = hidden_states.to(dev)
                out = qwen_attn_forward(block, hs, extra_args, extra_kwargs)
                del hs, out
        finally:
            handle.remove()
        finish_gptq(gptq_proj, proj_name)

        fc1_name = f"{prefix}.mlp.linear_fc1"
        gptq_fc1 = make_gptq(block.mlp.linear_fc1, fc1_name)
        logging.info(
            f"  [reforward] {fc1_name}: re-run attn(quant qkv+proj) -> +hs -> norm2 -> linear_fc1 to recompute fc1 input"
        )
        mlp_inputs = []
        for hidden_states, extra_args, extra_kwargs in tqdm(
            block_samples, desc=f"Vision GPTQ {fc1_name}", unit="batch"
        ):
            hs = hidden_states.to(dev)
            attn_out = qwen_attn_forward(block, hs, extra_args, extra_kwargs)
            mlp_inp = block.norm2(hs + attn_out)
            fc1_out = block.mlp.linear_fc1(mlp_inp)
            gptq_fc1.add_batch(mlp_inp, fc1_out)
            mlp_inputs.append(mlp_inp.detach().cpu())
            del hs, attn_out, mlp_inp, fc1_out
        finish_gptq(gptq_fc1, fc1_name)

        fc2_name = f"{prefix}.mlp.linear_fc2"
        gptq_fc2 = make_gptq(block.mlp.linear_fc2, fc2_name)
        logging.info(
            f"  [reforward] {fc2_name}: reuse cached norm2 output -> linear_fc1(quant) -> act -> linear_fc2 to recompute fc2 input"
        )
        for mlp_inp_cpu in tqdm(mlp_inputs, desc=f"Vision GPTQ {fc2_name}", unit="batch"):
            mlp_inp = mlp_inp_cpu.to(dev)
            fc2_inp = block.mlp.act_fn(block.mlp.linear_fc1(mlp_inp))
            fc2_out = block.mlp.linear_fc2(fc2_inp)
            gptq_fc2.add_batch(fc2_inp, fc2_out)
            del mlp_inp, fc2_inp, fc2_out
        finish_gptq(gptq_fc2, fc2_name)
        del mlp_inputs

        logging.info(
            f"  [reforward] {prefix}: propagate full quantized block to produce input for next block"
        )
        next_samples = []
        for hidden_states, extra_args, extra_kwargs in tqdm(
            block_samples, desc=f"Vision propagate {prefix}", unit="batch"
        ):
            hs = hidden_states.to(dev)
            out = block(hs, *_to_device(extra_args, dev), **_to_device(extra_kwargs, dev))
            next_samples.append((_to_cpu_detached(out), extra_args, extra_kwargs))
            del hs, out
        return next_samples

    def quantize_kimi_vision_block(
        block: nn.Module,
        block_samples: List[Tuple[torch.Tensor, Tuple[Any, ...], Dict[str, Any]]],
        prefix: str,
    ) -> List[Tuple[torch.Tensor, Tuple[Any, ...], Dict[str, Any]]]:
        qkv_name = f"{prefix}.wqkv"
        gptq_qkv = make_gptq(block.wqkv, qkv_name)
        logging.info(f"  [reforward] {qkv_name}: collect input via norm0 -> wqkv (fp)")
        for hidden_states, _, _ in tqdm(block_samples, desc=f"Vision GPTQ {qkv_name}", unit="batch"):
            hs = hidden_states.to(dev)
            qkv_inp = block.norm0(hs)
            qkv_out = block.wqkv(qkv_inp)
            gptq_qkv.add_batch(qkv_inp, qkv_out)
            del hs, qkv_inp, qkv_out
        finish_gptq(gptq_qkv, qkv_name)

        wo_name = f"{prefix}.wo"
        gptq_wo = make_gptq(block.wo, wo_name)
        logging.info(
            f"  [reforward] {wo_name}: re-run norm0 -> wqkv(quant) -> attn -> wo to recompute wo input"
        )
        handle = block.wo.register_forward_hook(
            lambda _, inp, out: gptq_wo.add_batch(inp[0].data, out.data)
        )
        try:
            for hidden_states, extra_args, extra_kwargs in tqdm(
                block_samples, desc=f"Vision GPTQ {wo_name}", unit="batch"
            ):
                hs = hidden_states.to(dev)
                out = kimi_attn_forward(block, hs, extra_args, extra_kwargs)
                del hs, out
        finally:
            handle.remove()
        finish_gptq(gptq_wo, wo_name)

        fc0_name = f"{prefix}.mlp.fc0"
        gptq_fc0 = make_gptq(block.mlp.fc0, fc0_name)
        logging.info(
            f"  [reforward] {fc0_name}: re-run attn(quant wqkv+wo) -> +hs -> norm1 -> fc0 to recompute fc0 input"
        )
        mlp_inputs = []
        for hidden_states, extra_args, extra_kwargs in tqdm(
            block_samples, desc=f"Vision GPTQ {fc0_name}", unit="batch"
        ):
            hs = hidden_states.to(dev)
            attn_out = kimi_attn_forward(block, hs, extra_args, extra_kwargs)
            mlp_inp = block.norm1(hs + attn_out)
            fc0_out = block.mlp.fc0(mlp_inp)
            gptq_fc0.add_batch(mlp_inp, fc0_out)
            mlp_inputs.append(mlp_inp.detach().cpu())
            del hs, attn_out, mlp_inp, fc0_out
        finish_gptq(gptq_fc0, fc0_name)

        fc1_name = f"{prefix}.mlp.fc1"
        gptq_fc1 = make_gptq(block.mlp.fc1, fc1_name)
        logging.info(
            f"  [reforward] {fc1_name}: reuse cached norm1 output -> fc0(quant) -> act -> fc1 to recompute fc1 input"
        )
        for mlp_inp_cpu in tqdm(mlp_inputs, desc=f"Vision GPTQ {fc1_name}", unit="batch"):
            mlp_inp = mlp_inp_cpu.to(dev)
            fc1_inp = block.mlp.activation(block.mlp.fc0(mlp_inp))
            fc1_out = block.mlp.fc1(fc1_inp)
            gptq_fc1.add_batch(fc1_inp, fc1_out)
            del mlp_inp, fc1_inp, fc1_out
        finish_gptq(gptq_fc1, fc1_name)
        del mlp_inputs

        logging.info(
            f"  [reforward] {prefix}: propagate full quantized block to produce input for next block"
        )
        next_samples = []
        for hidden_states, extra_args, extra_kwargs in tqdm(
            block_samples, desc=f"Vision propagate {prefix}", unit="batch"
        ):
            hs = hidden_states.to(dev)
            out = block(hs, *_to_device(extra_args, dev), **_to_device(extra_kwargs, dev))
            next_samples.append((_to_cpu_detached(out), extra_args, extra_kwargs))
            del hs, out
        return next_samples

    def quantize_subset(
        module: nn.Module,
        subset: Dict[str, nn.Linear],
        prefix: str,
        group_desc: str,
        block_samples: Optional[List[Tuple[torch.Tensor, Tuple[Any, ...], Dict[str, Any]]]] = None,
    ):
        gptq = {}
        for name, linear in subset.items():
            logging.info(f"  {prefix}.{name} ({vision_bits} bits)")
            gptq[name] = GPTQ(linear)
            if hasattr(linear, "weight_quantizer"):
                gptq[name].quantizer = linear.weight_quantizer
            else:
                gptq[name].quantizer = WeightQuantizer()
            gptq[name].quantizer.configure(vision_wqp)

        if not gptq:
            return

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = [subset[name].register_forward_hook(add_batch(name)) for name in gptq]
        try:
            if block_samples is None:
                logging.info(
                    f"  [reforward] {prefix}.{','.join(gptq.keys())}: full VLM forward up to LLM "
                    f"(slow; vision blocks already quantized)"
                )
                _run_vlm_calibration_until_text(
                    model=model,
                    dataloader=dataloader,
                    layers=layers,
                    dev=dev,
                    desc=f"Vision GPTQ {group_desc}",
                )
            else:
                logging.info(
                    f"  [reforward] {prefix}.{','.join(gptq.keys())}: re-run {prefix} on captured inputs "
                    f"(fast; preceding linears in {prefix} already quantized)"
                )
                for hidden_states, extra_args, extra_kwargs in tqdm(
                    block_samples,
                    desc=f"Vision GPTQ {group_desc}",
                    unit="batch",
                ):
                    out = module(
                        hidden_states.to(dev),
                        *_to_device(extra_args, dev),
                        **_to_device(extra_kwargs, dev),
                    )
                    del out
        finally:
            for h in handles:
                h.remove()

        if all(gptq[name].nsamples == 0 for name in gptq):
            for name in gptq:
                gptq[name].free()
            raise RuntimeError(
                f"No vision calibration activations collected for {prefix}.{','.join(gptq.keys())}. "
                "Check that calibration batches contain valid images and that --calib_image_folder matches "
                "the image paths in --calib_data_path."
            )

        for name in gptq:
            if gptq[name].nsamples == 0:
                logging.warning(f"  Skip {prefix}.{name}: no vision calibration activations collected")
                gptq[name].free()
                continue
            logging.info(f"  GPTQ quantizing {prefix}.{name} ({vision_bits} bits)")
            gptq[name].fasterquant(
                percdamp=weight_quant_params['percdamp'],
                groupsize=weight_quant_params['w_groupsize'],
                actorder=weight_quant_params.get('act_order', False),
                static_groups=False,
            )
            quantizers[f'{prefix}.{name}'] = gptq[name].quantizer
            gptq[name].free()

    block_samples = None
    for block_idx, block in enumerate(blocks):
        block = block.to(dev)
        block_prefix_i = f'{block_prefix}.{block_idx}'
        logging.info(f"\nVision block {block_idx}:")
        if block_samples is None:
            block_samples = _capture_vision_block_inputs(
                model=model,
                dataloader=dataloader,
                layers=layers,
                block=block,
                dev=dev,
                desc=f"Vision capture block {block_idx}",
            )
        if hasattr(block, 'attn') and hasattr(block.attn, 'qkv'):
            block_samples = quantize_qwen_vision_block(block, block_samples, block_prefix_i)
        elif hasattr(block, 'wqkv') and hasattr(block, 'wo'):
            block_samples = quantize_kimi_vision_block(block, block_samples, block_prefix_i)
        else:
            full = find_qlayers(block, layers=[torch.nn.Linear])
            for names in _vision_block_sequential(block):
                subset = {n: full[n] for n in names if n in full}
                if subset:
                    quantize_subset(
                        block,
                        subset,
                        block_prefix_i,
                        f"block {block_idx} {','.join(subset.keys())}",
                        block_samples=block_samples,
                    )
        cleanup_memory(verbos=False)
    del block_samples

    if not quantize_projector:
        if extra_modules:
            logging.info(
                f"Vision projector quantization disabled (--quantize_vision_projector not set); "
                f"skipping {len(extra_modules)} projector module(s): "
                f"{[p for p, _ in extra_modules]}"
            )
        cleanup_memory(verbos=False)
        logging.info(
            f"-----Vision Encoder GPTQ Quantization Done ({len(quantizers)} layers, "
            f"projector skipped)-----"
        )
        return quantizers

    for prefix, module in extra_modules:
        module = module.to(dev)
        full = find_qlayers(module, layers=[torch.nn.Linear])
        logging.info(f"\nVision projector {prefix}:")

        # Capture projector input ONCE (full VLM up to projector; vision blocks already quantized).
        # Subsequent linears reuse this captured input and only replay the projector module itself.
        # Some extras (e.g. deepstack mergers called from inside LLM layers) won't fire before
        # the LLM stop point — fall back to the slow path in that case.
        proj_samples: Optional[List[Tuple[torch.Tensor, Tuple[Any, ...], Dict[str, Any]]]] = None
        try:
            proj_samples = _capture_vision_block_inputs(
                model=model,
                dataloader=dataloader,
                layers=layers,
                block=module,
                dev=dev,
                desc=f"Vision capture {prefix}",
            )
        except RuntimeError as e:
            logging.warning(
                f"  Could not capture input for {prefix} ({e}). "
                f"Falling back to full VLM forward per linear (slow)."
            )

        for names in _vision_projector_sequential(module):
            subset = {n: full[n] for n in names if n in full}
            if subset:
                quantize_subset(
                    module,
                    subset,
                    prefix,
                    f"{prefix} {','.join(subset.keys())}",
                    block_samples=proj_samples,
                )
        del proj_samples
        cleanup_memory(verbos=False)

    cleanup_memory(verbos=False)
    logging.info(f"-----Vision Encoder GPTQ Quantization Done ({len(quantizers)} layers)-----")
    return quantizers


# ============================================================================
# GPTQ Forward pass
# ============================================================================
@torch.no_grad()
def gptq_fwrd(
    model: nn.Module, 
    dataloader: List,
    dev: torch.device, 
    nsamples: int, 
    weight_quant_params: Dict,
    mix_w_bits: bool = False, 
    mix_w_bits_dict: Optional[Dict] = None,
    keep_min: bool = False,
    skip_rare_expert: bool = False,
    percentage: float = 0.1,
    quantize_vision: bool = False,
    quantize_vision_projector: bool = False,
    vision_bits: int = 4,
    expert_bit_map: Optional[Dict[int, Dict[int, int]]] = None,
) -> Dict[str, WeightQuantizer]:
    """
    Perform GPTQ quantization on a model.
    
    Args:
        model: The model to quantize
        dataloader: Calibration data loader
        dev: Device to use for quantization
        nsamples: Number of calibration samples
        weight_quant_params: Weight quantization parameters
        mix_w_bits: Enable mixed-precision quantization
        mix_w_bits_dict: Bit-width configuration for different layer types
            - 'attn': bits for attention layers (q, k, v, o)
            - 'up_and_gate': bits for up and gate projections
            - 'down': bits for down projections
        keep_min: Enable minimum token balancing for MoE experts
        percentage: Minimum percentage of average tokens per expert (used when keep_min=True)
    
    Returns:
        Dictionary of quantizers for each layer
    """
    logging.info('-----GPTQ Quantization-----')
    
    # Safely get and set use_cache (some configs may not have this attribute)
    use_cache = getattr(model.config, 'use_cache', None)
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    
    model_type = get_model_type(model)
    logging.info(f"Model type: {model_type}")
    
    # First, move entire model to CPU to ensure clean state
    model = model.cpu()
    torch.cuda.empty_cache()
    strip_accelerate_dispatch_hooks(model)

    # Force quantization device to cuda:0
    if isinstance(dev, torch.device) and dev.type == 'cuda':
        dev = torch.device('cuda:0')
    elif isinstance(dev, str) and dev.startswith('cuda'):
        dev = torch.device('cuda:0')
    
    # Get model layers based on architecture
    # Qwen3VLMoe: model.model.language_model.layers
    # Kimi-VL: model.language_model.model.layers
    # Standard LLM: model.model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'layers'):
        # Qwen3-VL-MoE, Qwen3-VL structure
        layers = model.model.language_model.layers
        embed_tokens = model.model.language_model.embed_tokens
        norm = model.model.language_model.norm
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers'):
        # Kimi-VL structure: model.language_model.model.layers
        layers = model.language_model.model.layers
        embed_tokens = model.language_model.model.embed_tokens
        norm = model.language_model.model.norm
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Standard LLM structure (e.g., Llama, Mistral)
        layers = model.model.layers
        embed_tokens = model.model.embed_tokens if hasattr(model.model, 'embed_tokens') else model.model.tok_embeddings
        norm = model.model.norm
    else:
        raise ValueError(f"Unsupported model architecture: {model_type}")
    
    # Ensure all layers are on CPU initially
    for layer in layers:
        layer = layer.cpu()
    
    dtype = next(iter(model.parameters())).dtype
    
    # Get hidden_size from config (may be in text_config for VLM models)
    if hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'hidden_size'):
        hidden_size = model.config.text_config.hidden_size
    elif hasattr(model.config, 'hidden_size'):
        hidden_size = model.config.hidden_size
    else:
        raise ValueError("Cannot find hidden_size in model config")
    
    # Use list to store variable-length inputs
    inps = []
    cache = {'i': 0, 'attention_mask': None, 'attention_masks': [], 'position_ids_list': [], 'position_embeddings_list': []}
    
    logging.info(f"Device: {dev}")

    # Catcher module to capture layer inputs
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            
        def forward(self, inp, **kwargs):
            # inp shape: (batch_size, seq_len, hidden_dim)
            # Store each sample (squeeze batch dimension if batch_size=1)
            if inp.dim() == 3 and inp.shape[0] == 1:
                inps.append(inp.squeeze(0).cpu())
            else:
                inps.append(inp.cpu())
            cache['i'] += 1
            # Store per-sample attention_mask and position_ids
            cache['attention_masks'].append(kwargs.get('attention_mask', None))
            cache['position_ids_list'].append(kwargs.get('position_ids', None))
            if 'position_embeddings' in kwargs:
                cache['position_embeddings_list'].append(kwargs['position_embeddings'])
            raise ValueError
    
    # Ensure model is on CPU before collecting calibration data
    model = model.cpu()
    
    # Move necessary parts to device temporarily for calibration data collection
    # For VLM models with language_model, we need to move the entire language_model
    # to device because rotary_emb and other components are part of it
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        # Move entire language_model to device (includes embed_tokens, norm, rotary_emb, etc.)
        model.model.language_model = model.model.language_model.to(dev)
        embed_tokens = model.model.language_model.embed_tokens
        norm = model.model.language_model.norm
    elif hasattr(model, 'model'):
        # Standard LLM: move embed_tokens and norm to device
        if hasattr(model.model, 'embed_tokens'):
            model.model.embed_tokens = model.model.embed_tokens.to(dev)
            embed_tokens = model.model.embed_tokens
        elif hasattr(model.model, 'tok_embeddings'):
            model.model.tok_embeddings = model.model.tok_embeddings.to(dev)
            embed_tokens = model.model.tok_embeddings
        model.model.norm = model.model.norm.to(dev)
        norm = model.model.norm
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        # Alternative VLM structure: move entire language_model.model to device
        model.language_model.model = model.language_model.model.to(dev)
        embed_tokens = model.language_model.model.embed_tokens
        norm = model.language_model.model.norm

    # Move visual encoder to device (needed for VLM image processing during calibration)
    _visual_moved = []
    for _parent, _attr in [
        (getattr(model, 'model', None), 'visual'),        # Qwen3-VL-MoE / Qwen3-VL
        (model, 'visual'),                                 # fallback visual
        (model, 'vision_tower'),                           # Kimi-VL
        (model, 'vision_model'),                           # InternVL
        (getattr(model, 'model', None), 'vision_model'),   # InternVL alt
    ]:
        if _parent is not None and hasattr(_parent, _attr):
            _vis = getattr(_parent, _attr)
            if isinstance(_vis, nn.Module):
                setattr(_parent, _attr, _vis.to(dev))
                _visual_moved.append((_parent, _attr))
                break  # only one visual encoder per model

    # Move visual projector / merger to device (some models have it as a separate module)
    for _parent, _attr in [
        (model, 'multi_modal_projector'),                  # Kimi-VL
        (model, 'mlp1'),                                   # InternVL
        (getattr(model, 'model', None), 'merger'),         # Qwen (if separate)
    ]:
        if _parent is not None and hasattr(_parent, _attr):
            _proj = getattr(_parent, _attr)
            if isinstance(_proj, nn.Module):
                setattr(_parent, _attr, _proj.to(dev))
                _visual_moved.append((_parent, _attr))

    quantizers = {}
    if quantize_vision:
        vision_quantizers = quantize_vision_encoder_gptq(
            model=model,
            dataloader=dataloader,
            layers=layers,
            dev=dev,
            weight_quant_params=weight_quant_params,
            vision_bits=vision_bits,
            quantize_projector=quantize_vision_projector,
        )
        quantizers.update(vision_quantizers)
    
    layers[0] = Catcher(layers[0].to(dev))
    
    def _fix_batch_dims(inputs):
        """Ensure VLM input tensors have batch dimension."""
        if 'input_ids' in inputs and inputs['input_ids'].dim() == 1:
            inputs['input_ids'] = inputs['input_ids'].unsqueeze(0)
        if 'attention_mask' in inputs and inputs['attention_mask'].dim() == 1:
            inputs['attention_mask'] = inputs['attention_mask'].unsqueeze(0)
        for grid_key in ('image_grid_thw', 'video_grid_thw', 'image_grid_hws'):
            if grid_key in inputs and isinstance(inputs[grid_key], torch.Tensor) and inputs[grid_key].dim() == 1:
                inputs[grid_key] = inputs[grid_key].unsqueeze(0)
        return inputs

    for batch in dataloader:
        try:
            # Handle different input formats
            if isinstance(batch, dict):
                # Full dictionary input (VLM models with images)
                inputs = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                inputs = _fix_batch_dims(inputs)
                model(**inputs)
            elif isinstance(batch, tuple) or isinstance(batch, list):
                input_ids = batch[0]
                if isinstance(input_ids, dict):
                    # Tuple/list containing dict
                    inputs = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in input_ids.items()}
                    inputs = _fix_batch_dims(inputs)
                    model(**inputs)
                else:
                    # Simple tensor input
                    input_ids = input_ids.to(dev)
                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)
                    model(input_ids)
            else:
                input_ids = batch.to(dev)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                model(input_ids)
        except ValueError:
            pass
    layers[0] = layers[0].module

    # Move all layers and model components back to CPU after calibration data collection
    layers[0] = layers[0].cpu()
    
    # Update references and move back to CPU
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        # Move entire language_model back to CPU
        model.model.language_model = model.model.language_model.cpu()
        embed_tokens = model.model.language_model.embed_tokens
        norm = model.model.language_model.norm
    elif hasattr(model, 'model'):
        # Standard LLM: move embed_tokens and norm back to CPU
        if hasattr(model.model, 'embed_tokens'):
            model.model.embed_tokens = model.model.embed_tokens.cpu()
            embed_tokens = model.model.embed_tokens
        elif hasattr(model.model, 'tok_embeddings'):
            model.model.tok_embeddings = model.model.tok_embeddings.cpu()
            embed_tokens = model.model.tok_embeddings
        model.model.norm = model.model.norm.cpu()
        norm = model.model.norm
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        # Alternative VLM structure: move entire language_model.model back to CPU
        model.language_model.model = model.language_model.model.cpu()
        embed_tokens = model.language_model.model.embed_tokens
        norm = model.language_model.model.norm

    # Move visual encoder back to CPU
    for _parent, _attr in _visual_moved:
        setattr(_parent, _attr, getattr(_parent, _attr).cpu())
    
    # Ensure entire model is on CPU
    model = model.cpu()
    # Ensure all layers are on CPU
    for i in range(len(layers)):
        layers[i] = layers[i].cpu()
    torch.cuda.empty_cache()

    # Initialize outputs list (same structure as inps)
    outs = [None] * len(inps)
    # Note: attention_mask and position_ids are now per-sample lists
    nsamples = len(inps)
    logging.info(f"Collected {nsamples} calibration samples")
    
    num_layers = len(layers)
    
    for layer_idx in range(num_layers):
        logging.info(f'\nLayer {layer_idx}:')
        # Move layer to cuda:0 for quantization
        layers[layer_idx] = layers[layer_idx].to(dev)
        layer = layers[layer_idx]
        
        # Check if this is a MoE layer
        layer_is_moe = is_moe_layer(layer)
        if layer_is_moe:
            logging.info(f"  (MoE layer)")
        
        # Find all nn.Linear layers
        full = find_qlayers(layer, layers=[torch.nn.Linear])
        
        # Print total token count for this layer
        total_layer_tokens = sum(inp.shape[0] if len(inp.shape) >= 2 else inp.shape[0] for inp in inps)
        logging.info(f"Layer {layer_idx}: total input tokens = {total_layer_tokens}")
        
        # Generate sequential groups based on layer structure
        sequential = generate_sequential_for_layer(layer)
        
        # Detect if this is Kimi-VL DeepseekV3Attention structure
        attn = layer.self_attn if hasattr(layer, 'self_attn') else None
        is_kimi_vl_attention = (
            attn is not None and 
            hasattr(attn, 'kv_a_proj_with_mqa') and 
            hasattr(attn, 'kv_b_proj')
        )
        
        # GPTQ quantization for nn.Linear layers (attention + dense MLP)
        sequential_idx = 0
        for names in sequential:
            # Filter to only existing layers in current layer
            subset = {n: full[n] for n in names if n in full}
                
            if not subset:
                sequential_idx += 1
                continue

            # Check if this is o_proj (after qkv) or down_proj (after up/gate)
            is_o_proj = 'self_attn.o_proj' in names
            is_down_proj = 'mlp.down_proj' in names
            
            # Check if this is kv_b_proj (Kimi-VL DeepseekV3Attention, needs re-forward after q_proj+kv_a_proj_with_mqa)
            is_kv_b_proj = 'self_attn.kv_b_proj' in names and len(names) == 1
            
            # Check if this is standard qkv group (q_proj+k_proj+v_proj for standard attention)
            is_standard_qkv_group = (
                'self_attn.q_proj' in names and 'self_attn.k_proj' in names and 'self_attn.v_proj' in names
            )
            
            # Check if this is Kimi-VL q_proj+kv_a_proj_with_mqa group (first group, no re-forward needed)
            is_kimi_vl_qkv_a_group = (
                'self_attn.q_proj' in names and 'self_attn.kv_a_proj_with_mqa' in names and 
                'self_attn.kv_b_proj' not in names
            )
            
            # Print re-forward message before collecting activations
            if is_kv_b_proj:
                # kv_b_proj needs re-forward after quantizing q_proj+kv_a_proj_with_mqa
                logging.info(f"    [Re-forward] Collecting kv_b_proj inputs using quantized q_proj+kv_a_proj_with_mqa")
            elif is_o_proj:
                if is_standard_qkv_group:
                    # This shouldn't happen, but just in case
                    pass
                elif is_kimi_vl_attention:
                    # Kimi-VL: o_proj after kv_b_proj (and q_proj+kv_a_proj_with_mqa)
                    logging.info(f"    [Re-forward] Collecting o_proj inputs using quantized q_proj+kv_a_proj_with_mqa+kv_b_proj")
                else:
                    # Standard attention: o_proj after qkv
                    logging.info(f"    [Re-forward] Collecting o_proj inputs using quantized qkv")
            elif is_down_proj:
                logging.info(f"    [Re-forward] Collecting down_proj inputs using quantized up_proj+gate_proj")

            gptq = {}
            for name in subset:
                weight_quant_params_copy = copy.deepcopy(weight_quant_params)
                
                # Apply mixed-precision bit allocation
                if mix_w_bits and mix_w_bits_dict:
                    if name in ['mlp.up_proj', 'mlp.gate_proj']:
                        weight_quant_params_copy['w_bits'] = mix_w_bits_dict.get('up_and_gate', weight_quant_params['w_bits'])
                    elif name in ['mlp.down_proj']:
                        weight_quant_params_copy['w_bits'] = mix_w_bits_dict.get('down', weight_quant_params['w_bits'])
                    elif name in ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj', 'self_attn.o_proj',
                                  'self_attn.kv_a_proj_with_mqa', 'self_attn.kv_b_proj']:
                        weight_quant_params_copy['w_bits'] = mix_w_bits_dict.get('attn', weight_quant_params['w_bits'])
                                    
                # Skip lm_head and output layers
                if 'lm_head' in name or 'output' in name:
                    logging.info(f"  Skip quantization for {name}")
                    continue
                
                # Skip router/gate layers (MoE routing)
                if name == 'mlp.gate' or 'router' in name.lower():
                    logging.info(f"  Skip MoE router: {name}")
                    continue
                    
                logging.info(f'  {name} ({weight_quant_params_copy["w_bits"]} bits)')
                
                gptq[name] = GPTQ(subset[name])
                
                # Use existing weight quantizer if available
                if hasattr(subset[name], "weight_quantizer"):
                    gptq[name].quantizer = subset[name].weight_quantizer
                else:
                    gptq[name].quantizer = WeightQuantizer()
                gptq[name].quantizer.configure(weight_quant_params_copy)
                
            if not gptq:
                sequential_idx += 1
                continue
                
            # Hook to collect activations
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            
            handles = []
            for name in gptq:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
                    
            # Forward pass to collect activations
            for j in range(nsamples):
                inp_j = inps[j].to(dev).unsqueeze(0)
                attn_mask_j, pos_ids_j, pos_emb_j = prepare_layer_kwargs(cache, j, dev)
                
                if pos_emb_j is not None:
                    out_j = layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j, position_embeddings=pos_emb_j)
                else:
                    out_j = layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j)
                
                # Handle different return types
                if isinstance(out_j, tuple):
                    outs[j] = out_j[0].squeeze(0).cpu()
                else:
                    outs[j] = out_j.squeeze(0).cpu()
                    
            for h in handles:
                h.remove()
                
            # Perform GPTQ quantization
            for name in gptq:
                if gptq[name].quantizer.bits >= 16:
                    logging.info(f"  w_bits>=16, skip {name} quantization")
                    continue
                    
                logging.info(f"  GPTQ quantizing {name} ({gptq[name].quantizer.bits} bits)")
                
                gptq[name].fasterquant(
                    percdamp=weight_quant_params['percdamp'], 
                    groupsize=weight_quant_params['w_groupsize'], 
                    actorder=weight_quant_params.get('act_order', False), 
                    static_groups=False,
                )
                quantizers[f'model.layers.{layer_idx}.{name}'] = gptq[name].quantizer
                gptq[name].free()
            
            sequential_idx += 1
        
        # For MoE layers: GPTQ quantize each expert separately
        if layer_is_moe:
            num_experts = get_num_experts_from_layer(layer)
            experts = layer.mlp.experts
            
            # Check MoE structure type
            if is_moe_3d_param_structure(experts):
                # 3D Parameter structure (Qwen3-VL-MoE, Qwen3-MoE)
                # Get weight dimensions
                # Actual transformers implementation uses: x @ gate_up_proj (not linear(x, W))
                # gate_up_proj: (num_experts, hidden_size, 2*intermediate)
                # down_proj: (num_experts, intermediate, hidden_size)
                # So in_features for gate_up = shape[1], in_features for down = shape[1]
                gate_up_proj = experts.gate_up_proj
                down_proj = experts.down_proj
            elif is_moe_modulelist_structure(experts):
                # ModuleList structure (standard MoE like InternVL with Qwen3-MoE)
                # Optimized: batch collect activations for all experts to reduce forward passes
                # GPTQ order: gate_proj + up_proj -> re-forward -> down_proj
                logging.info(f"  MoE ModuleList structure detected: {num_experts} experts")
                
                # Collect all expert layers
                expert_gate_layers = {}
                expert_up_layers = {}
                expert_down_layers = {}
                expert_gptq_gate = {}
                expert_gptq_up = {}
                expert_gptq_down = {}
                
                weight_quant_params_gate_up = copy.deepcopy(weight_quant_params)
                if mix_w_bits and mix_w_bits_dict:
                    weight_quant_params_gate_up['w_bits'] = mix_w_bits_dict.get('up_and_gate', weight_quant_params['w_bits'])
                
                weight_quant_params_down = copy.deepcopy(weight_quant_params)
                if mix_w_bits and mix_w_bits_dict:
                    weight_quant_params_down['w_bits'] = mix_w_bits_dict.get('down', weight_quant_params['w_bits'])
                
                bits_gate_up = weight_quant_params_gate_up['w_bits']
                bits_down = weight_quant_params_down['w_bits']
                
                # Step 1: Collect all gate_proj and up_proj layers, create GPTQ instances
                for expert_idx in range(num_experts):
                    expert = experts[expert_idx]
                    if expert is None:
                        continue

                    expert_gate_up_params = _set_expert_bit(
                        weight_quant_params_gate_up,
                        expert_bit_map,
                        layer_idx,
                        expert_idx,
                        bits_gate_up,
                    )
                    expert_down_params = _set_expert_bit(
                        weight_quant_params_down,
                        expert_bit_map,
                        layer_idx,
                        expert_idx,
                        bits_down,
                    )
                    expert_bits_gate_up = expert_gate_up_params["w_bits"]
                    expert_bits_down = expert_down_params["w_bits"]
                    
                    expert_layers = find_qlayers(expert, layers=[torch.nn.Linear])
                    gate_proj_layer = expert_layers.get('gate_proj')
                    up_proj_layer = expert_layers.get('up_proj')
                    down_proj_layer = expert_layers.get('down_proj')
                    
                    if gate_proj_layer is not None and expert_bits_gate_up < 16:
                        expert_gate_layers[expert_idx] = gate_proj_layer
                        expert_gptq_gate[expert_idx] = GPTQ(gate_proj_layer)
                        expert_gptq_gate[expert_idx].quantizer = WeightQuantizer()
                        expert_gptq_gate[expert_idx].quantizer.configure(expert_gate_up_params)
                    
                    if up_proj_layer is not None and expert_bits_gate_up < 16:
                        expert_up_layers[expert_idx] = up_proj_layer
                        expert_gptq_up[expert_idx] = GPTQ(up_proj_layer)
                        expert_gptq_up[expert_idx].quantizer = WeightQuantizer()
                        expert_gptq_up[expert_idx].quantizer.configure(expert_gate_up_params)
                    
                    if down_proj_layer is not None and expert_bits_down < 16:
                        expert_down_layers[expert_idx] = down_proj_layer
                        expert_gptq_down[expert_idx] = GPTQ(down_proj_layer)
                        expert_gptq_down[expert_idx].quantizer = WeightQuantizer()
                        expert_gptq_down[expert_idx].quantizer.configure(expert_down_params)
                
                # Step 2: Single forward pass to collect activations for all gate_proj and up_proj
                if expert_gate_layers or expert_up_layers:
                    handles = []
                    expert_token_counts = {}  # Track token counts per expert (only count once per expert, gate and up share same input)
                    expert_inputs_dict = {idx: [] for idx in range(num_experts)}  # Collect inputs for balancing
                    all_layer_tokens_list = []  # Collect all tokens for balancing
                    
                    # Get top_k from config or gate
                    top_k = getattr(layer.mlp.gate, 'top_k', None)
                    if top_k is None:
                        # Try to get from config
                        if hasattr(layer.mlp, 'config') and hasattr(layer.mlp.config, 'num_experts_per_tok'):
                            top_k = layer.mlp.config.num_experts_per_tok
                        elif hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'num_experts_per_tok'):
                            top_k = model.config.text_config.num_experts_per_tok
                        elif hasattr(model.config, 'num_experts_per_tok'):
                            top_k = model.config.num_experts_per_tok
                        else:
                            # Default to 1 if cannot determine
                            top_k = 1
                            logging.warning(f"  Cannot determine top_k, defaulting to 1")
                    
                    # Register hooks for all gate_proj and up_proj layers
                    for expert_idx, linear_layer in expert_gate_layers.items():
                        def add_batch_gate(idx):
                            def tmp(_, inp, out):
                                inp_data = inp[0].data
                                # Count tokens: inp_data shape is (batch_size, seq_len, hidden_dim) or (seq_len, hidden_dim)
                                if len(inp_data.shape) == 3:
                                    num_tokens = inp_data.shape[0] * inp_data.shape[1]
                                    # Flatten for collection
                                    inp_data_flat = inp_data.reshape(-1, inp_data.shape[-1])
                                else:
                                    num_tokens = inp_data.shape[0]
                                    inp_data_flat = inp_data
                                
                                expert_token_counts[idx] = expert_token_counts.get(idx, 0) + num_tokens
                                expert_inputs_dict[idx].append(inp_data_flat.detach().clone())
                                expert_gptq_gate[idx].add_batch(inp_data, out.data)
                            return tmp
                        handles.append(linear_layer.register_forward_hook(add_batch_gate(expert_idx)))
                    
                    for expert_idx, linear_layer in expert_up_layers.items():
                        def add_batch_up(idx):
                            def tmp(_, inp, out):
                                expert_gptq_up[idx].add_batch(inp[0].data, out.data)
                            return tmp
                        handles.append(linear_layer.register_forward_hook(add_batch_up(expert_idx)))
                    
                    # Collect all tokens from expert_inputs_dict (already collected in add_batch_gate)
                    # We'll concatenate them after forward pass
                    
                    # Single forward pass for all experts
                    logging.info(f"  Collecting activations for gate_proj and up_proj ({len(expert_gate_layers) + len(expert_up_layers)} layers)")
                    for j in range(nsamples):
                        inp_j = inps[j].to(dev).unsqueeze(0)
                        attn_mask_j, pos_ids_j, pos_emb_j = prepare_layer_kwargs(cache, j, dev)
                        
                        if pos_emb_j is not None:
                            layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j, position_embeddings=pos_emb_j)
                        else:
                            layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j)
                    
                    for h in handles:
                        h.remove()
                    
                    # Get hidden_dim from first expert input or gate layer
                    hidden_dim = None
                    if expert_inputs_dict and any(expert_inputs_dict.values()):
                        # Get from first non-empty expert input
                        for expert_idx in range(num_experts):
                            if expert_inputs_dict[expert_idx]:
                                hidden_dim = expert_inputs_dict[expert_idx][0].shape[-1]
                                break
                    elif hasattr(layer.mlp, 'gate') and hasattr(layer.mlp.gate, 'weight'):
                        hidden_dim = layer.mlp.gate.weight.shape[-1]
                    elif expert_gate_layers:
                        first_gate = list(expert_gate_layers.values())[0]
                        if hasattr(first_gate, 'in_features'):
                            hidden_dim = first_gate.in_features
                    
                    if hidden_dim is None:
                        hidden_dim = 4096  # Default fallback
                        logging.warning(f"  Cannot determine hidden_dim, defaulting to {hidden_dim}")
                    
                    # Collect token pool from expert_inputs_dict (post-attention representation)
                    # These are the actual inputs to expert gate_proj/up_proj, in the correct representation space
                    # Note: tokens routed to multiple experts will appear multiple times, but this is fine for sampling
                    expert_token_pool_list = []
                    for expert_idx in range(num_experts):
                        if expert_inputs_dict[expert_idx]:
                            expert_token_pool_list.extend(expert_inputs_dict[expert_idx])
                    
                    if expert_token_pool_list:
                        all_layer_tokens = torch.cat(expert_token_pool_list, dim=0)  # (total_routed_tokens, hidden_dim)
                    else:
                        all_layer_tokens = torch.empty(0, hidden_dim, device=dev, dtype=torch.float32)
                    
                    # Calculate total unique tokens from layer inputs (for avg calculation)
                    # all_layer_tokens may have duplicates due to top_k routing, so use inps for counting
                    total_unique_tokens = sum(inps[j].reshape(-1, inps[j].shape[-1]).shape[0] if len(inps[j].shape) == 3 else inps[j].shape[0] for j in range(nsamples))
                    logging.info(f"  Token pool for balancing: using post-attention expert inputs (pool_size={all_layer_tokens.shape[0]}, unique_tokens={total_unique_tokens})")
                    
                    # Balance expert tokens if keep_min is enabled, or skip rare experts if skip_rare_expert is enabled
                    padded_experts = set()  # Track which experts were padded
                    skipped_experts = set()  # Track which experts are skipped (skip_rare_expert mode)
                    padded_gate_inputs = {}  # Store padded gate inputs for down_proj calibration
                    
                    # Calculate average tokens and minimum tokens (used by both keep_min and skip_rare_expert)
                    if (keep_min or skip_rare_expert) and all_layer_tokens.shape[0] > 0:
                        # Use unique token count for avg calculation (all_layer_tokens may have top_k duplicates)
                        total_tokens = total_unique_tokens
                        pool_size = all_layer_tokens.shape[0]  # Size of the sampling pool (may have duplicates)
                        avg_tokens_per_expert = (total_tokens * top_k) / num_experts
                        min_tokens = int(avg_tokens_per_expert * percentage)
                        
                        if keep_min:
                            logging.info(f"  Balancing expert tokens (top_k={top_k}, percentage={percentage})")
                            # For each expert with gate_proj, check if padding is needed
                            for expert_idx in expert_gate_layers.keys():
                                num_expert_tokens = expert_token_counts.get(expert_idx, 0)
                                if num_expert_tokens < min_tokens:
                                    # Need to pad: randomly sample tokens from all_layer_tokens (post-attention token pool)
                                    num_to_pad = min_tokens - num_expert_tokens
                                    random_indices = torch.randint(0, pool_size, (num_to_pad,), device=dev)
                                    padded_tokens = all_layer_tokens[random_indices]
                                    
                                    # Forward padded tokens through gate_proj and up_proj, add to both GPTQs
                                    gate_layer = expert_gate_layers[expert_idx]
                                    up_layer = expert_up_layers.get(expert_idx)
                                    with torch.no_grad():
                                        padded_gate_output = gate_layer(padded_tokens)
                                        expert_gptq_gate[expert_idx].add_batch(padded_tokens.unsqueeze(0) if len(padded_tokens.shape) == 2 else padded_tokens, padded_gate_output)
                                        
                                        # Also add to up_proj GPTQ (gate and up share the same input)
                                        if up_layer is not None and expert_idx in expert_gptq_up:
                                            padded_up_output = up_layer(padded_tokens)
                                            expert_gptq_up[expert_idx].add_batch(padded_tokens.unsqueeze(0) if len(padded_tokens.shape) == 2 else padded_tokens, padded_up_output)
                                    
                                    # Store padded gate inputs for down_proj calibration
                                    # Concatenate original inputs with padded tokens
                                    original_inputs = torch.cat(expert_inputs_dict[expert_idx], dim=0) if expert_inputs_dict[expert_idx] else torch.empty(0, padded_tokens.shape[-1], device=dev, dtype=padded_tokens.dtype)
                                    padded_gate_inputs[expert_idx] = torch.cat([original_inputs, padded_tokens], dim=0)
                                    
                                    logging.info(f"    Expert {expert_idx}: padded from {num_expert_tokens} to {min_tokens} tokens (for gate_proj AND up_proj calibration)")
                                    padded_experts.add(expert_idx)
                                else:
                                    # Store original inputs for down_proj calibration
                                    if expert_inputs_dict[expert_idx]:
                                        padded_gate_inputs[expert_idx] = torch.cat(expert_inputs_dict[expert_idx], dim=0)
                        
                        elif skip_rare_expert:
                            logging.info(f"  Skipping rare experts (top_k={top_k}, percentage={percentage}, min_tokens={min_tokens})")
                            # For each expert, check if it should be skipped
                            for expert_idx in expert_gate_layers.keys():
                                num_expert_tokens = expert_token_counts.get(expert_idx, 0)
                                if num_expert_tokens < min_tokens:
                                    # Skip this expert: set bits to 16 (no quantization)
                                    skipped_experts.add(expert_idx)
                                    logging.info(f"    Expert {expert_idx}: skipping quantization ({num_expert_tokens} tokens < {min_tokens} min_tokens)")
                                else:
                                    # Store original inputs for down_proj calibration
                                    if expert_inputs_dict[expert_idx]:
                                        padded_gate_inputs[expert_idx] = torch.cat(expert_inputs_dict[expert_idx], dim=0)
                    
                    # If not using keep_min or skip_rare_expert, store all original inputs
                    if not keep_min and not skip_rare_expert:
                        for expert_idx in expert_gate_layers.keys():
                            if expert_inputs_dict.get(expert_idx):
                                padded_gate_inputs[expert_idx] = torch.cat(expert_inputs_dict[expert_idx], dim=0)
                    
                    # Print token counts for each expert (only print if not already printed during padding/skipping)
                    all_expert_indices = sorted(set(expert_gate_layers.keys()) | set(expert_up_layers.keys()))
                    for expert_idx in all_expert_indices:
                        token_count = expert_token_counts.get(expert_idx, 0)
                        # Only print if not padded and not skipped (padded/skipped experts already printed above)
                        if not (keep_min and expert_idx in padded_experts) and not (skip_rare_expert and expert_idx in skipped_experts):
                            logging.info(f"    Expert {expert_idx}: {token_count} tokens")
                    
                    # Step 3: Quantize all gate_proj and up_proj (skip experts in skipped_experts)
                    for expert_idx in sorted(expert_gate_layers.keys()):
                        if expert_idx in skipped_experts:
                            # Skip quantization: set bits to 16 (no quantization)
                            logging.info(f"  Skipping mlp.experts.{expert_idx}.gate_proj (rare expert, bits=16)")
                            # Create a dummy quantizer with bits=16
                            dummy_quantizer = WeightQuantizer()
                            dummy_quantizer.configure({'w_bits': 16})
                            quantizers[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj'] = dummy_quantizer
                            expert_gptq_gate[expert_idx].free()
                        else:
                            gate_nsamples = expert_gptq_gate[expert_idx].nsamples if hasattr(expert_gptq_gate[expert_idx], 'nsamples') else 0
                            gate_actual_tokens = padded_gate_inputs[expert_idx].shape[0] if expert_idx in padded_gate_inputs and padded_gate_inputs[expert_idx] is not None else expert_token_counts.get(expert_idx, 0)
                            logging.info(f"  GPTQ quantizing mlp.experts.{expert_idx}.gate_proj ({bits_gate_up} bits), batches={gate_nsamples}, tokens={gate_actual_tokens}")
                            expert_gptq_gate[expert_idx].fasterquant(
                                percdamp=weight_quant_params['percdamp'],
                                groupsize=weight_quant_params['w_groupsize'],
                                actorder=weight_quant_params.get('act_order', False),
                                static_groups=False,
                            )
                            quantizers[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj'] = expert_gptq_gate[expert_idx].quantizer
                            expert_gptq_gate[expert_idx].free()
                    
                    for expert_idx in sorted(expert_up_layers.keys()):
                        if expert_idx in skipped_experts:
                            # Skip quantization: set bits to 16 (no quantization)
                            logging.info(f"  Skipping mlp.experts.{expert_idx}.up_proj (rare expert, bits=16)")
                            # Create a dummy quantizer with bits=16
                            dummy_quantizer = WeightQuantizer()
                            dummy_quantizer.configure({'w_bits': 16})
                            quantizers[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj'] = dummy_quantizer
                            expert_gptq_up[expert_idx].free()
                        else:
                            up_nsamples = expert_gptq_up[expert_idx].nsamples if hasattr(expert_gptq_up[expert_idx], 'nsamples') else 0
                            up_actual_tokens = padded_gate_inputs[expert_idx].shape[0] if expert_idx in padded_gate_inputs and padded_gate_inputs[expert_idx] is not None else expert_token_counts.get(expert_idx, 0)
                            logging.info(f"  GPTQ quantizing mlp.experts.{expert_idx}.up_proj ({bits_gate_up} bits), batches={up_nsamples}, tokens={up_actual_tokens}")
                            expert_gptq_up[expert_idx].fasterquant(
                                percdamp=weight_quant_params['percdamp'],
                                groupsize=weight_quant_params['w_groupsize'],
                                actorder=weight_quant_params.get('act_order', False),
                                static_groups=False,
                            )
                            quantizers[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj'] = expert_gptq_up[expert_idx].quantizer
                            expert_gptq_up[expert_idx].free()
                
                # Step 4: Re-forward to collect down_proj inputs using quantized gate_proj and up_proj
                if expert_down_layers:
                    logging.info(f"    [Re-forward] Collecting down_proj inputs using quantized gate_proj+up_proj")
                    
                    # For all experts, manually forward inputs through gate->up->down
                    # This ensures we use the original routing (from expert_inputs_dict) rather than re-routing with quantized gate_proj
                    for expert_idx in sorted(expert_down_layers.keys()):
                        gate_layer = expert_gate_layers.get(expert_idx)
                        up_layer = expert_up_layers.get(expert_idx)
                        down_layer = expert_down_layers[expert_idx]
                        
                        if gate_layer and up_layer:
                            # Skip experts that are skipped in skip_rare_expert mode
                            if expert_idx in skipped_experts:
                                continue  # Skip this expert
                            
                            # Get inputs: use padded inputs if available, otherwise use original inputs
                            if expert_idx in padded_gate_inputs and padded_gate_inputs[expert_idx].shape[0] > 0:
                                gate_inputs = padded_gate_inputs[expert_idx]  # Use padded inputs for padded experts
                                logging.debug(f"    Expert {expert_idx}: using padded inputs, shape={gate_inputs.shape}")
                            elif expert_inputs_dict.get(expert_idx) and expert_inputs_dict[expert_idx]:
                                gate_inputs = torch.cat(expert_inputs_dict[expert_idx], dim=0)  # Use original inputs for non-padded experts
                                logging.debug(f"    Expert {expert_idx}: using original inputs, shape={gate_inputs.shape}")
                            else:
                                logging.warning(f"    Expert {expert_idx}: no inputs found, skipping")
                                continue  # Skip if no inputs
                            
                            with torch.no_grad():
                                # Forward through quantized gate_proj (weights are already quantized)
                                gate_output = gate_layer(gate_inputs)
                                # Apply activation (SiLU for gate)
                                gate_act = torch.nn.functional.silu(gate_output)
                                
                                # Forward through quantized up_proj (weights are already quantized)
                                up_output = up_layer(gate_inputs)
                                
                                # Element-wise multiply gate_act and up_output
                                gated_output = gate_act * up_output
                                ex_mod = experts[expert_idx] if expert_idx < len(experts) else None
                                gated_output = quarot_kimi_expert_mid_pre_down(gated_output, ex_mod)

                                # Forward through down_proj to get output (for add_batch)
                                down_output = down_layer(gated_output)
                                
                                # Collect down_proj inputs and outputs
                                # add_batch expects 2D tensor (num_tokens, features) or 3D (batch, num_tokens, features)
                                # gated_output is 2D: (num_tokens, intermediate_dim)
                                # down_output is 2D: (num_tokens, hidden_dim)
                                # For add_batch: if 2D, it will unsqueeze to [1, num_tokens, features], then tmp=1 (batch_size)
                                # So nsamples counts batches, not tokens. But Hessian will contain all tokens.
                                num_tokens_for_down = gated_output.shape[0]
                                logging.debug(f"    Expert {expert_idx}: forwarding {num_tokens_for_down} tokens for down_proj calibration")
                                expert_gptq_down[expert_idx].add_batch(gated_output, down_output)
                    
                    # Step 5: Quantize all down_proj (skip experts in skipped_experts)
                    for expert_idx in sorted(expert_down_layers.keys()):
                        if expert_idx in skipped_experts:
                            # Skip quantization: set bits to 16 (no quantization)
                            logging.info(f"  Skipping mlp.experts.{expert_idx}.down_proj (rare expert, bits=16)")
                            # Create a dummy quantizer with bits=16
                            dummy_quantizer = WeightQuantizer()
                            dummy_quantizer.configure({'w_bits': 16})
                            quantizers[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj'] = dummy_quantizer
                            expert_gptq_down[expert_idx].free()
                            continue
                        
                        down_layer = expert_down_layers[expert_idx]
                        weight_shape = down_layer.weight.shape
                        nsamples_collected = expert_gptq_down[expert_idx].nsamples if hasattr(expert_gptq_down[expert_idx], 'nsamples') else 0
                        # Calculate actual token count from Hessian matrix shape (H is [features, features], built from [features, num_tokens])
                        # The number of tokens is reflected in the Hessian accumulation, not directly in nsamples
                        # For debugging: nsamples is batch count, actual tokens processed = nsamples * tokens_per_batch
                        # But since we call add_batch once with all tokens, nsamples=1, but H contains all tokens
                        # Get token count from the input we used
                        if expert_idx in padded_gate_inputs and padded_gate_inputs[expert_idx].shape[0] > 0:
                            actual_token_count = padded_gate_inputs[expert_idx].shape[0]
                        elif expert_inputs_dict.get(expert_idx) and expert_inputs_dict[expert_idx]:
                            actual_token_count = sum(t.shape[0] for t in expert_inputs_dict[expert_idx])
                        else:
                            actual_token_count = 0
                        logging.info(f"  GPTQ quantizing mlp.experts.{expert_idx}.down_proj ({bits_down} bits), weight shape={weight_shape}, batches={nsamples_collected}, tokens={actual_token_count}")
                        expert_gptq_down[expert_idx].fasterquant(
                            percdamp=weight_quant_params['percdamp'],
                            groupsize=weight_quant_params['w_groupsize'],
                            actorder=weight_quant_params.get('act_order', False),
                            static_groups=False,
                        )
                        quantizers[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj'] = expert_gptq_down[expert_idx].quantizer
                        expert_gptq_down[expert_idx].free()
                    
                    # Step 6: Re-forward with original inputs (not padded) to ensure correct output
                    logging.info(f"    [Re-forward] Re-forwarding with original inputs (not padded) after quantizing all experts")
                    for j in range(nsamples):
                        inp_j = inps[j].to(dev).unsqueeze(0)
                        attn_mask_j, pos_ids_j, pos_emb_j = prepare_layer_kwargs(cache, j, dev)
                        
                        if pos_emb_j is not None:
                            layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j, position_embeddings=pos_emb_j)
                        else:
                            layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j)
                
                # Clean up ModuleList MoE intermediate tensors
                del expert_gate_layers, expert_up_layers, expert_down_layers
                del expert_gptq_gate, expert_gptq_up, expert_gptq_down
                del expert_inputs_dict, expert_token_counts
                try:
                    del all_layer_tokens, padded_gate_inputs
                    del expert_token_pool_list, all_layer_tokens_list
                except NameError:
                    pass
                gc.collect()
                torch.cuda.empty_cache()
                
                # Collect quantized output for next layer (MUST do before continue)
                logging.info(f"  [ModuleList MoE] Collecting quantized outputs for layer {layer_idx} -> layer {layer_idx + 1}")
                for j in range(nsamples):
                    inp_j = inps[j].to(dev).unsqueeze(0)
                    attn_mask_j, pos_ids_j, pos_emb_j = prepare_layer_kwargs(cache, j, dev)
                    if pos_emb_j is not None:
                        out_j = layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j, position_embeddings=pos_emb_j)
                    else:
                        out_j = layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j)
                    if isinstance(out_j, tuple):
                        outs[j] = out_j[0].squeeze(0).cpu()
                    else:
                        outs[j] = out_j.squeeze(0).cpu()
                    del inp_j, out_j, attn_mask_j, pos_ids_j, pos_emb_j
                
                # Move layer back to CPU and clean GPU memory
                layers[layer_idx] = layers[layer_idx].cpu()
                del layer
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                inps, outs = outs, inps
                logging.info(f"  [ModuleList MoE] Layer {layer_idx} done: moved to CPU, swapped inps/outs for next layer")
                continue
            else:
                logging.warning(f"  Unknown MoE structure, skipping expert quantization")
                # Still need to collect output and clean up
                logging.info(f"  [Unknown MoE] Collecting quantized outputs for layer {layer_idx} -> layer {layer_idx + 1}")
                for j in range(nsamples):
                    inp_j = inps[j].to(dev).unsqueeze(0)
                    attn_mask_j, pos_ids_j, pos_emb_j = prepare_layer_kwargs(cache, j, dev)
                    if pos_emb_j is not None:
                        out_j = layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j, position_embeddings=pos_emb_j)
                    else:
                        out_j = layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j)
                    if isinstance(out_j, tuple):
                        outs[j] = out_j[0].squeeze(0).cpu()
                    else:
                        outs[j] = out_j.squeeze(0).cpu()
                    del inp_j, out_j, attn_mask_j, pos_ids_j, pos_emb_j
                layers[layer_idx] = layers[layer_idx].cpu()
                del layer
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                inps, outs = outs, inps
                logging.info(f"  [Unknown MoE] Layer {layer_idx} done: moved to CPU, swapped inps/outs for next layer")
                continue
            
            # Continue with 3D Parameter structure quantization
            gate_up_proj = experts.gate_up_proj
            down_proj = experts.down_proj
            
            hidden_dim = gate_up_proj.shape[1]  # in_features for gate_up (hidden_size)
            intermediate_dim = down_proj.shape[1]  # in_features for down (intermediate)
            
            logging.info(f"  gate_up_proj shape: {gate_up_proj.shape}")
            logging.info(f"  down_proj shape: {down_proj.shape}")
            logging.info(f"  hidden_dim (gate_up in_features): {hidden_dim}")
            logging.info(f"  intermediate_dim (down in_features): {intermediate_dim}")
            
            # Collect per-expert inputs by hooking into the experts module
            expert_inputs = {i: [] for i in range(num_experts)}  # gate_up inputs
            expert_down_inputs = {i: [] for i in range(num_experts)}  # down inputs
            all_layer_tokens_list = []  # Collect all tokens for balancing
            
            # Get top_k from config or gate
            top_k = getattr(layer.mlp.gate, 'top_k', None)
            if top_k is None:
                # Try to get from config
                if hasattr(layer.mlp, 'config') and hasattr(layer.mlp.config, 'num_experts_per_tok'):
                    top_k = layer.mlp.config.num_experts_per_tok
                elif hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'num_experts_per_tok'):
                    top_k = model.config.text_config.num_experts_per_tok
                elif hasattr(model.config, 'num_experts_per_tok'):
                    top_k = model.config.num_experts_per_tok
                else:
                    # Default to 1 if cannot determine
                    top_k = 1
                    logging.warning(f"  Cannot determine top_k, defaulting to 1")
            
            # Store original forward to restore later
            original_experts_forward = experts.forward
            
            def hooked_experts_forward(hidden_states, routing_weights, router_indices):
                """Modified forward that collects per-expert inputs (matches actual transformers impl)."""
                batch_size = hidden_states.shape[0]
                hidden_states = hidden_states.reshape(-1, experts.hidden_size)
                
                # Collect all layer tokens for balancing
                all_layer_tokens_list.append(hidden_states.detach().clone())
                
                # Use training-style loop to collect per-expert inputs
                next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
                with torch.no_grad():
                    expert_mask = torch.nn.functional.one_hot(router_indices.long(), num_classes=experts.num_experts)
                    expert_mask = expert_mask.permute(2, 1, 0)
                    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
                
                for expert_idx_tensor in expert_hit[:]:
                    expert_idx = expert_idx_tensor[0].item()
                    with torch.no_grad():
                        _, token_idx = torch.where(expert_mask[expert_idx])
                    
                    current_state = hidden_states[token_idx]
                    
                    # Collect gate_up input for this expert
                    if len(current_state) > 0:
                        expert_inputs[expert_idx].append(current_state.detach().clone())
                    
                    # Compute: gate_up = current_state @ gate_up_proj[expert_idx]
                    gate_up = current_state @ experts.gate_up_proj[expert_idx]
                    gate, up = gate_up.chunk(2, dim=-1)
                    gated_output = up * experts.act_fn(gate)
                    gated_output = quarot_moe_mid_pre_down(gated_output, experts)
                    
                    # Collect down input for this expert
                    if len(gated_output) > 0:
                        expert_down_inputs[expert_idx].append(gated_output.detach().clone())
                    
                    # Compute: out = gated_output @ down_proj[expert_idx]
                    out = gated_output @ experts.down_proj[expert_idx]
                    weighted_output = out * routing_weights[token_idx, expert_idx, None]
                    next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
                
                next_states = next_states.view(batch_size, -1, experts.hidden_size)
                return next_states
            
            # Replace forward temporarily
            experts.forward = hooked_experts_forward
            
            # Forward pass to collect per-expert inputs
            for j in range(nsamples):
                inp_j = inps[j].to(dev).unsqueeze(0)
                attn_mask_j, pos_ids_j, pos_emb_j = prepare_layer_kwargs(cache, j, dev)
                
                if pos_emb_j is not None:
                    layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j, position_embeddings=pos_emb_j)
                else:
                    layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j)
            
            # Restore original forward
            experts.forward = original_experts_forward
            
            # Concatenate all layer tokens
            if all_layer_tokens_list:
                all_layer_tokens = torch.cat(all_layer_tokens_list, dim=0)  # (total_tokens, hidden_dim)
            else:
                all_layer_tokens = torch.empty(0, hidden_dim, device=dev, dtype=gate_up_proj.dtype)
            
            # Balance expert tokens if keep_min is enabled, or skip rare experts if skip_rare_expert is enabled
            padded_experts_3d = set()  # Track which experts were padded (for logging)
            skipped_experts_3d = set()  # Track which experts are skipped (skip_rare_expert mode)
            original_expert_inputs = {}  # Store original inputs (not padded) for final re-forward
            
            # Calculate average tokens and minimum tokens (used by both keep_min and skip_rare_expert)
            if (keep_min or skip_rare_expert) and all_layer_tokens.shape[0] > 0:
                total_tokens = all_layer_tokens.shape[0]
                avg_tokens_per_expert = (total_tokens * top_k) / num_experts
                min_tokens = int(avg_tokens_per_expert * percentage)
                
                # Count original tokens before balancing/skipping and save original inputs
                original_token_counts = {}
                for expert_idx in range(num_experts):
                    if expert_inputs[expert_idx]:
                        original_token_counts[expert_idx] = sum(t.shape[0] for t in expert_inputs[expert_idx])
                        # Save original inputs (not padded) for final re-forward
                        original_expert_inputs[expert_idx] = torch.cat(expert_inputs[expert_idx], dim=0)
                    else:
                        original_token_counts[expert_idx] = 0
                        original_expert_inputs[expert_idx] = None
                
                if keep_min:
                    logging.info(f"  Balancing expert tokens (top_k={top_k}, percentage={percentage})")
                    expert_inputs = balance_expert_tokens(
                        expert_inputs=expert_inputs,
                        all_layer_tokens=all_layer_tokens,
                        num_experts=num_experts,
                        top_k=top_k,
                        keep_min=keep_min,
                        percentage=percentage,
                        dev=dev,
                    )
                    
                    # Track which experts were padded by checking if token count increased
                    for expert_idx in range(num_experts):
                        if expert_inputs.get(expert_idx) is not None:
                            new_count = expert_inputs[expert_idx].shape[0]
                            if new_count > original_token_counts.get(expert_idx, 0):
                                padded_experts_3d.add(expert_idx)
                
                elif skip_rare_expert:
                    logging.info(f"  Skipping rare experts (top_k={top_k}, percentage={percentage}, min_tokens={min_tokens})")
                    # Track which experts should be skipped
                    for expert_idx in range(num_experts):
                        if original_token_counts[expert_idx] < min_tokens:
                            skipped_experts_3d.add(expert_idx)
                            logging.info(f"    Expert {expert_idx}: skipping quantization ({original_token_counts[expert_idx]} tokens < {min_tokens} min_tokens)")
                    # Concatenate expert_inputs (not yet done; only original_expert_inputs was concatenated above)
                    for expert_idx in range(num_experts):
                        if isinstance(expert_inputs.get(expert_idx), list):
                            if expert_inputs[expert_idx]:
                                expert_inputs[expert_idx] = torch.cat(expert_inputs[expert_idx], dim=0)
                            else:
                                expert_inputs[expert_idx] = None
            else:
                # No padding/skipping, concatenate and save original inputs as-is
                for expert_idx in range(num_experts):
                    if expert_inputs[expert_idx]:
                        expert_inputs[expert_idx] = torch.cat(expert_inputs[expert_idx], dim=0)
                        original_expert_inputs[expert_idx] = expert_inputs[expert_idx]
                    else:
                        expert_inputs[expert_idx] = None
                        original_expert_inputs[expert_idx] = None
            
            # === Quantize gate_up_proj for each expert ===
            # Weight shape: (hidden_size, 2*intermediate) - used as x @ W
            # GPTQ processes columns (in_features), so we transpose to (out, in), quantize, transpose back
            weight_quant_params_copy = copy.deepcopy(weight_quant_params)
            if mix_w_bits and mix_w_bits_dict:
                weight_quant_params_copy['w_bits'] = mix_w_bits_dict.get('up_and_gate', weight_quant_params['w_bits'])
            if weight_quant_params_copy.get('w_groupsize', -1) != -1:
                weight_quant_params_copy['perchannel'] = False
            
            bits = weight_quant_params_copy['w_bits']
            if bits < 16 or expert_bit_map:
                logging.info(f"  GPTQ quantizing mlp.experts.gate_up_proj (base={bits} bits) [{num_experts} experts]")
                
                for expert_idx in range(num_experts):
                    expert_wqp = _set_expert_bit(
                        weight_quant_params_copy,
                        expert_bit_map,
                        layer_idx,
                        expert_idx,
                        bits,
                    )
                    expert_bits = expert_wqp["w_bits"]
                    # Skip rare experts if skip_rare_expert is enabled
                    if expert_idx in skipped_experts_3d:
                        logging.info(f"    Expert {expert_idx}: skipping quantization (rare expert, bits=16)")
                        # Create a dummy quantizer with bits=16
                        dummy_quantizer = WeightQuantizer()
                        dummy_quantizer.configure({'w_bits': 16})
                        quantizers[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_up_proj'] = dummy_quantizer
                        continue
                    if expert_bits >= 16:
                        logging.info(f"    Expert {expert_idx}: w_bits>=16, skip gate_up_proj quantization")
                        continue
                    
                    # W shape: (hidden_dim, 2*intermediate) - for x @ W computation
                    # Transpose to (2*intermediate, hidden_dim) for GPTQ processing
                    W_expert = gate_up_proj.data[expert_idx].t().contiguous()  # (2*intermediate, hidden_dim)
                    
                    # Get collected inputs for this expert (may have been balanced)
                    if expert_inputs.get(expert_idx) is not None:
                        all_expert_inputs = expert_inputs[expert_idx]  # Already concatenated and balanced
                        num_tokens = all_expert_inputs.shape[0]
                        # Only log if not padded (padded experts already logged in balance_expert_tokens)
                        if expert_idx not in padded_experts_3d:
                            logging.info(f"    Expert {expert_idx}: {num_tokens} tokens")
                    else:
                        logging.warning(f"    Expert {expert_idx}: no inputs collected, using random")
                        all_expert_inputs = torch.randn(128, hidden_dim, device=dev, dtype=W_expert.dtype)
                    
                    gptq_expert = GPTQWeight(W_expert, in_features=hidden_dim)
                    gptq_expert.quantizer = WeightQuantizer()
                    gptq_expert.quantizer.configure(expert_wqp)
                    
                    gptq_expert.add_batch(all_expert_inputs)
                    
                    Q = gptq_expert.fasterquant(
                        percdamp=weight_quant_params['percdamp'],
                        groupsize=weight_quant_params['w_groupsize'],
                        actorder=weight_quant_params.get('act_order', False),
                    )
                    
                    # Transpose back to original shape (hidden_dim, 2*intermediate)
                    gate_up_proj.data[expert_idx] = Q.t().contiguous()
                    quantizers[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_up_proj'] = gptq_expert.quantizer
                    gptq_expert.free()
                    
                    if (expert_idx + 1) % 10 == 0 or expert_idx == num_experts - 1:
                        logging.info(f"    Expert {expert_idx + 1}/{num_experts} done")
            
            # === Re-forward to collect down_proj inputs using quantized gate_up_proj ===
            # After quantizing gate_up_proj, we need to re-forward to collect down_proj inputs
            # using the quantized gate_up_proj weights
            # For padded experts, we need to manually forward padded inputs through gate_up->down
            expert_down_inputs_quantized = {i: [] for i in range(num_experts)}  # down inputs with quantized gate_up
            
            # First, manually forward padded inputs for padded experts (skip skipped experts)
            for expert_idx in padded_experts_3d:
                if expert_idx in skipped_experts_3d:
                    continue  # Skip rare experts
                if expert_inputs.get(expert_idx) is not None and expert_inputs[expert_idx].shape[0] > 0:
                    padded_gate_inputs = expert_inputs[expert_idx]
                    with torch.no_grad():
                        # Forward through quantized gate_up_proj
                        gate_up = padded_gate_inputs @ experts.gate_up_proj[expert_idx]
                        gate, up = gate_up.chunk(2, dim=-1)
                        gated_output = up * experts.act_fn(gate)
                        gated_output = quarot_moe_mid_pre_down(gated_output, experts)
                        
                        # Collect down input for this expert
                        expert_down_inputs_quantized[expert_idx].append(gated_output.detach().clone())
            
            def hooked_experts_forward_quantized(hidden_states, routing_weights, router_indices):
                """Modified forward that collects down_proj inputs using quantized gate_up_proj."""
                batch_size = hidden_states.shape[0]
                hidden_states = hidden_states.reshape(-1, experts.hidden_size)
                
                next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
                with torch.no_grad():
                    expert_mask = torch.nn.functional.one_hot(router_indices.long(), num_classes=experts.num_experts)
                    expert_mask = expert_mask.permute(2, 1, 0)
                    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
                
                for expert_idx_tensor in expert_hit[:]:
                    expert_idx = expert_idx_tensor[0].item()
                    # Skip padded experts (already processed manually) and skipped experts
                    if expert_idx in padded_experts_3d or expert_idx in skipped_experts_3d:
                        continue
                    
                    with torch.no_grad():
                        _, token_idx = torch.where(expert_mask[expert_idx])
                    
                    current_state = hidden_states[token_idx]
                    
                    # Compute using quantized gate_up_proj: gate_up = current_state @ gate_up_proj[expert_idx]
                    gate_up = current_state @ experts.gate_up_proj[expert_idx]
                    gate, up = gate_up.chunk(2, dim=-1)
                    gated_output = up * experts.act_fn(gate)
                    gated_output = quarot_moe_mid_pre_down(gated_output, experts)
                    
                    # Collect down input for this expert (using quantized gate_up_proj)
                    if len(gated_output) > 0:
                        expert_down_inputs_quantized[expert_idx].append(gated_output.detach().clone())
                    
                    # Compute: out = gated_output @ down_proj[expert_idx]
                    out = gated_output @ experts.down_proj[expert_idx]
                    weighted_output = out * routing_weights[token_idx, expert_idx, None]
                    next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
                
                next_states = next_states.view(batch_size, -1, experts.hidden_size)
                return next_states
            
            # === Quantize down_proj for each expert ===
            # Weight shape: (intermediate, hidden_size) - used as x @ W
            weight_quant_params_copy = copy.deepcopy(weight_quant_params)
            if mix_w_bits and mix_w_bits_dict:
                weight_quant_params_copy['w_bits'] = mix_w_bits_dict.get('down', weight_quant_params['w_bits'])
            if weight_quant_params_copy.get('w_groupsize', -1) != -1:
                weight_quant_params_copy['perchannel'] = False
            
            bits = weight_quant_params_copy['w_bits']
            if bits < 16 or expert_bit_map:
                # Print re-forward message before collecting activations
                logging.info(f"  [Re-forward] Collecting down_proj inputs using quantized gate_up_proj")
                logging.info(f"  GPTQ quantizing mlp.experts.down_proj (base={bits} bits) [{num_experts} experts]")
                
                # Replace forward temporarily to collect down_proj inputs with quantized gate_up_proj
                # (only for non-padded experts, padded experts already processed)
                experts.forward = hooked_experts_forward_quantized
                
                # Re-forward pass to collect down_proj inputs using quantized gate_up_proj
                for j in range(nsamples):
                    inp_j = inps[j].to(dev).unsqueeze(0)
                    attn_mask_j, pos_ids_j, pos_emb_j = prepare_layer_kwargs(cache, j, dev)
                    
                    if pos_emb_j is not None:
                        layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j, position_embeddings=pos_emb_j)
                    else:
                        layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j)
                
                # Restore original forward
                experts.forward = original_experts_forward
                
                for expert_idx in range(num_experts):
                    expert_wqp = _set_expert_bit(
                        weight_quant_params_copy,
                        expert_bit_map,
                        layer_idx,
                        expert_idx,
                        bits,
                    )
                    expert_bits = expert_wqp["w_bits"]
                    # Skip rare experts if skip_rare_expert is enabled
                    if expert_idx in skipped_experts_3d:
                        logging.info(f"    Expert {expert_idx}: skipping down_proj quantization (rare expert, bits=16)")
                        # Create a dummy quantizer with bits=16
                        dummy_quantizer = WeightQuantizer()
                        dummy_quantizer.configure({'w_bits': 16})
                        quantizers[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj'] = dummy_quantizer
                        continue
                    if expert_bits >= 16:
                        logging.info(f"    Expert {expert_idx}: w_bits>=16, skip down_proj quantization")
                        continue
                    
                    # W shape: (intermediate, hidden_dim) - for x @ W computation
                    # Transpose to (hidden_dim, intermediate) for GPTQ processing
                    W_expert = down_proj.data[expert_idx].t().contiguous()  # (hidden_dim, intermediate)
                    
                    # Get collected down inputs using quantized gate_up_proj
                    if expert_down_inputs_quantized[expert_idx]:
                        all_down_inputs = torch.cat(expert_down_inputs_quantized[expert_idx], dim=0)  # (total_tokens, intermediate)
                    else:
                        logging.warning(f"    Expert {expert_idx}: no inputs collected, using random")
                        all_down_inputs = torch.randn(128, intermediate_dim, device=dev, dtype=W_expert.dtype)
                    
                    # Print shapes for verification
                    logging.info(f"    Expert {expert_idx}: down_proj weight shape={W_expert.shape}, input tokens shape={all_down_inputs.shape} (tokens={all_down_inputs.shape[0]})")
                    
                    gptq_expert = GPTQWeight(W_expert, in_features=intermediate_dim)
                    gptq_expert.quantizer = WeightQuantizer()
                    gptq_expert.quantizer.configure(expert_wqp)
                    
                    gptq_expert.add_batch(all_down_inputs)
                    
                    Q = gptq_expert.fasterquant(
                        percdamp=weight_quant_params['percdamp'],
                        groupsize=weight_quant_params['w_groupsize'],
                        actorder=weight_quant_params.get('act_order', False),
                    )
                    
                    # Transpose back to original shape (intermediate, hidden_dim)
                    down_proj.data[expert_idx] = Q.t().contiguous()
                    quantizers[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj'] = gptq_expert.quantizer
                    gptq_expert.free()
                    
                    if (expert_idx + 1) % 10 == 0 or expert_idx == num_experts - 1:
                        logging.info(f"    Expert {expert_idx + 1}/{num_experts} done")
                
                # Step 6: Re-forward with original inputs (not padded) to ensure correct output
                logging.info(f"    [Re-forward] Re-forwarding with original inputs (not padded) after quantizing all experts")
                for j in range(nsamples):
                    inp_j = inps[j].to(dev).unsqueeze(0)
                    attn_mask_j, pos_ids_j, pos_emb_j = prepare_layer_kwargs(cache, j, dev)
                    
                    if pos_emb_j is not None:
                        layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j, position_embeddings=pos_emb_j)
                    else:
                        layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j)
            
            # Clean up MoE intermediate tensors and free GPU memory
            del expert_inputs, expert_down_inputs, expert_down_inputs_quantized, original_expert_inputs
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
                
        # Get quantized output for next layer
        logging.info(f"  [3D MoE / Dense] Collecting quantized outputs for layer {layer_idx} -> layer {layer_idx + 1}")
        for j in range(nsamples):
            inp_j = inps[j].to(dev).unsqueeze(0)
            attn_mask_j, pos_ids_j, pos_emb_j = prepare_layer_kwargs(cache, j, dev)
            
            if pos_emb_j is not None:
                out_j = layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j, position_embeddings=pos_emb_j)
            else:
                out_j = layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j)
            
            # Handle different return types - move to CPU immediately to free GPU memory
            if isinstance(out_j, tuple):
                outs[j] = out_j[0].squeeze(0).cpu()
            else:
                outs[j] = out_j.squeeze(0).cpu()
            del inp_j, out_j, attn_mask_j, pos_ids_j, pos_emb_j

        # Move layer back to CPU and aggressively clean GPU memory after each layer
        layers[layer_idx] = layers[layer_idx].cpu()
        del layer
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        logging.info(f"  [3D MoE / Dense] Layer {layer_idx} done: moved to CPU, swapped inps/outs for next layer")

    # Restore use_cache if it was originally set
    if hasattr(model.config, 'use_cache') and use_cache is not None:
        model.config.use_cache = use_cache
    cleanup_memory(verbos=True)
    logging.info('-----GPTQ Quantization Done-----\n')
    return quantizers


# ============================================================================
# RTN (Round-To-Nearest) Forward pass
# ============================================================================
@torch.no_grad()
def rtn_fwrd(
    model: nn.Module, 
    dev: torch.device, 
    weight_quant_params: Dict, 
    mix_w_bits: bool = False, 
    mix_w_bits_dict: Optional[Dict] = None,
    expert_protect_map: Optional[Dict[int, set]] = None,
    protect_bits: int = 4,
    quantize_shared_experts: bool = False,
    quantize_vision: bool = False,
    vision_bits: int = 4,
    expert_bit_map: Optional[Dict[int, Dict[int, int]]] = None,
) -> Dict[str, WeightQuantizer]:
    """
    Perform RTN (Round-To-Nearest) quantization on a model.
    
    RTN is a simpler quantization method that doesn't require calibration data.
    It directly rounds weights to the nearest quantized value.
    
    Args:
        model: The model to quantize
        dev: Device to use for quantization
        weight_quant_params: Weight quantization parameters
        mix_w_bits: Enable mixed-precision quantization
        mix_w_bits_dict: Bit-width configuration for different layer types
            - 'attn': bits for attention layers (q, k, v, o)
            - 'up_and_gate': bits for up and gate projections
            - 'down': bits for down projections
        expert_protect_map: Per-layer protected expert indices.
            Dict mapping layer_idx -> set of expert indices to protect at protect_bits.
        protect_bits: Bit-width for protected experts (default: 4)
    
    Returns:
        Dictionary of quantizers for each layer
    """
    logging.info('-----RTN Quantization-----')
    
    model_type = get_model_type(model)
    logging.info(f"Model type: {model_type}")
    
    # Get model layers based on architecture
    # Qwen3VLMoe: model.model.language_model.layers
    # Kimi-VL: model.language_model.model.layers
    # Standard LLM: model.model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'layers'):
        # Qwen3-VL-MoE, Qwen3-VL structure
        layers = model.model.language_model.layers
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers'):
        # Kimi-VL structure: model.language_model.model.layers
        layers = model.language_model.model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Standard LLM structure (e.g., Llama, Mistral)
        layers = model.model.layers
    else:
        raise ValueError(f"Unsupported model architecture: {model_type}")
    
    torch.cuda.empty_cache()

    quantizers = {}

    for layer_idx in tqdm(range(len(layers)), desc="(RTN Quant.) Layers"):
        layer = layers[layer_idx].to(dev)

        # Check if this is a MoE layer
        layer_is_moe = is_moe_layer(layer)
        # ModuleList MoE: routed experts are quantized in the dedicated block below with
        # correct mix_w_bits (up_and_gate / down). The generic find_qlayers pass also sees
        # mlp.experts.* Linears but mixed-precision keys only match mlp.gate_proj etc., so
        # without skipping we would RTN-quantize every expert twice (and first pass uses wrong bits).
        skip_generic_expert_linears = False
        if layer_is_moe and hasattr(layer.mlp, 'experts'):
            skip_generic_expert_linears = is_moe_modulelist_structure(layer.mlp.experts)
            
        # Find all nn.Linear layers
        subset = find_qlayers(layer, layers=[torch.nn.Linear])
        
        # Quantize nn.Linear layers
        for name in subset:
            weight_quant_params_copy = copy.deepcopy(weight_quant_params)
            
            # Skip lm_head and output layers
            if 'lm_head' in name:
                logging.info(f"  Skip lm_head quantization")
                continue
            if 'output' in name:
                logging.info(f"  Skip output layer quantization")
                continue
            
            # Skip router/gate layers (MoE routing)
            if name == 'mlp.gate' or 'router' in name.lower():
                logging.info(f"  Skip MoE router: {name}")
                continue

            # Routed experts (ModuleList MoE): only the dedicated loop below applies correct bits
            if skip_generic_expert_linears and name.startswith('mlp.experts.'):
                continue

            # shared_experts: controlled by quantize_shared_experts flag
            if 'shared_experts' in name:
                if not quantize_shared_experts:
                    logging.info(f"  Skip shared_experts (flag off): {name}")
                    continue
                if mix_w_bits and mix_w_bits_dict:
                    if name in ['mlp.shared_experts.gate_proj', 'mlp.shared_experts.up_proj']:
                        weight_quant_params_copy['w_bits'] = mix_w_bits_dict.get('up_and_gate', weight_quant_params['w_bits'])
                    elif name == 'mlp.shared_experts.down_proj':
                        weight_quant_params_copy['w_bits'] = mix_w_bits_dict.get('down', weight_quant_params['w_bits'])
            # Apply mixed-precision bit allocation for standard layers
            elif mix_w_bits and mix_w_bits_dict:
                if name in ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj', 'self_attn.o_proj',
                            'self_attn.kv_a_proj_with_mqa', 'self_attn.kv_b_proj']:
                    weight_quant_params_copy['w_bits'] = mix_w_bits_dict.get('attn', weight_quant_params['w_bits'])
                elif name in ['mlp.up_proj', 'mlp.gate_proj']:
                    weight_quant_params_copy['w_bits'] = mix_w_bits_dict.get('up_and_gate', weight_quant_params['w_bits'])
                elif name in ['mlp.down_proj']:
                    weight_quant_params_copy['w_bits'] = mix_w_bits_dict.get('down', weight_quant_params['w_bits'])
            
            if weight_quant_params_copy.get('w_groupsize', -1) != -1:
                weight_quant_params_copy['perchannel'] = False

            bits = weight_quant_params_copy['w_bits']
            if bits >= 16:
                logging.info(f"  w_bits>=16, skip {name} quantization")
                continue
            
            logging.info(f"  RTN quantizing {name} ({bits} bits)")

            # Create or use existing quantizer
            if hasattr(subset[name], "weight_quantizer"):
                quantizer = subset[name].weight_quantizer
            else:
                quantizer = WeightQuantizer()
            quantizer.configure(weight_quant_params_copy)
            
            # Quantize weights directly
            W = subset[name].weight.data
            subset[name].weight.data = quantizer.forward(W)

            quantizers[f'model.layers.{layer_idx}.{name}'] = quantizer
        
        # For MoE layers: RTN quantize each expert separately
        if layer_is_moe:
            num_experts = get_num_experts_from_layer(layer)
            experts = layer.mlp.experts
            
            # Check MoE structure type
            if is_moe_modulelist_structure(experts):
                # ModuleList structure (standard MoE like InternVL with Qwen3-MoE)
                # Each expert is a separate module with gate_proj, up_proj, down_proj
                logging.info(f"  MoE ModuleList structure detected: {num_experts} experts")
                for expert_idx in range(num_experts):
                    expert = experts[expert_idx]
                    if expert is None:
                        continue
                    
                    # Find Linear layers in this expert
                    expert_layers = find_qlayers(expert, layers=[torch.nn.Linear])
                    
                    for name, linear_layer in expert_layers.items():
                        weight_quant_params_copy = copy.deepcopy(weight_quant_params)
                        
                        # Apply mixed-precision bit allocation
                        if mix_w_bits and mix_w_bits_dict:
                            if name in ['gate_proj', 'up_proj']:
                                weight_quant_params_copy['w_bits'] = mix_w_bits_dict.get('up_and_gate', weight_quant_params['w_bits'])
                            elif name == 'down_proj':
                                weight_quant_params_copy['w_bits'] = mix_w_bits_dict.get('down', weight_quant_params['w_bits'])
                        weight_quant_params_copy['w_bits'] = get_expert_bit(
                            expert_bit_map,
                            layer_idx,
                            expert_idx,
                            weight_quant_params_copy['w_bits'],
                        )
                        
                        if weight_quant_params_copy.get('w_groupsize', -1) != -1:
                            weight_quant_params_copy['perchannel'] = False

                        bits = weight_quant_params_copy['w_bits']
                        if bits >= 16:
                            continue
                        
                        logging.info(f"  RTN quantizing mlp.experts.{expert_idx}.{name} ({bits} bits)")
                        
                        quantizer = WeightQuantizer()
                        quantizer.configure(weight_quant_params_copy)
                        W = linear_layer.weight.data
                        linear_layer.weight.data = quantizer.forward(W)
                        
                        quantizers[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.{name}'] = quantizer
                
                # Skip 3D Parameter quantization for ModuleList structure
                layers[layer_idx] = layer.cpu()
                torch.cuda.empty_cache()
                del layer
                continue
            
            # Determine protected experts for this layer
            protected_experts = set()
            if expert_protect_map is not None and layer_idx in expert_protect_map:
                protected_experts = expert_protect_map[layer_idx]

            # === Quantize gate_up_proj for each expert ===
            # Weight shape: (hidden_size, 2*intermediate) - used as x @ W
            # 3D Parameter structure (Qwen3-VL-MoE, Qwen3-MoE)
            if hasattr(experts, 'gate_up_proj'):
                gate_up_proj = experts.gate_up_proj
                weight_quant_params_base = copy.deepcopy(weight_quant_params)
                
                if mix_w_bits and mix_w_bits_dict:
                    weight_quant_params_base['w_bits'] = mix_w_bits_dict.get('up_and_gate', weight_quant_params['w_bits'])
                
                if weight_quant_params_base.get('w_groupsize', -1) != -1:
                    weight_quant_params_base['perchannel'] = False
                
                base_bits = weight_quant_params_base['w_bits']
                if base_bits < 16 or protected_experts:
                    n_protected = len(protected_experts) if protected_experts else 0
                    logging.info(f"  RTN quantizing mlp.experts.gate_up_proj (base={base_bits}b, protected={n_protected} at {protect_bits}b) [{num_experts} experts]")
                    
                    bits_count = {}
                    for expert_idx in range(num_experts):
                        wqp = copy.deepcopy(weight_quant_params_base)
                        mapped_bit = get_expert_bit(expert_bit_map, layer_idx, expert_idx, wqp['w_bits'])
                        if mapped_bit != wqp['w_bits']:
                            wqp['w_bits'] = mapped_bit
                        elif expert_idx in protected_experts:
                            wqp['w_bits'] = protect_bits
                        
                        bits_e = wqp['w_bits']
                        bits_count[bits_e] = bits_count.get(bits_e, 0) + 1
                        if bits_e >= 16:
                            continue
                        
                        W_expert = gate_up_proj.data[expert_idx].t().contiguous()
                        
                        quantizer = WeightQuantizer()
                        quantizer.configure(wqp)
                        Q = quantizer.forward(W_expert)
                        gate_up_proj.data[expert_idx] = Q.t().contiguous()
                        
                        quantizers[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_up_proj'] = quantizer
                        
                        if (expert_idx + 1) % 10 == 0 or expert_idx == num_experts - 1:
                            logging.info(f"    Expert {expert_idx + 1}/{num_experts} done")
                    logging.info(f"    gate_up bit distribution: {dict(sorted(bits_count.items()))}")
            
            # === Quantize down_proj for each expert ===
            # Weight shape: (intermediate, hidden_size) - used as x @ W
            if hasattr(experts, 'down_proj'):
                down_proj = experts.down_proj
                weight_quant_params_base = copy.deepcopy(weight_quant_params)
                
                if mix_w_bits and mix_w_bits_dict:
                    weight_quant_params_base['w_bits'] = mix_w_bits_dict.get('down', weight_quant_params['w_bits'])
                
                if weight_quant_params_base.get('w_groupsize', -1) != -1:
                    weight_quant_params_base['perchannel'] = False
                
                base_bits = weight_quant_params_base['w_bits']
                if base_bits < 16 or protected_experts:
                    n_protected_d = len(protected_experts) if protected_experts else 0
                    logging.info(f"  RTN quantizing mlp.experts.down_proj (base={base_bits}b, protected={n_protected_d} at {protect_bits}b) [{num_experts} experts]")
                    
                    bits_count = {}
                    for expert_idx in range(num_experts):
                        wqp = copy.deepcopy(weight_quant_params_base)
                        mapped_bit = get_expert_bit(expert_bit_map, layer_idx, expert_idx, wqp['w_bits'])
                        if mapped_bit != wqp['w_bits']:
                            wqp['w_bits'] = mapped_bit
                        elif expert_idx in protected_experts:
                            wqp['w_bits'] = protect_bits
                        
                        bits_e = wqp['w_bits']
                        bits_count[bits_e] = bits_count.get(bits_e, 0) + 1
                        if bits_e >= 16:
                            continue
                        
                        W_expert = down_proj.data[expert_idx].t().contiguous()
                        
                        quantizer = WeightQuantizer()
                        quantizer.configure(wqp)
                        Q = quantizer.forward(W_expert)
                        down_proj.data[expert_idx] = Q.t().contiguous()
                        
                        quantizers[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj'] = quantizer
                        
                        if (expert_idx + 1) % 10 == 0 or expert_idx == num_experts - 1:
                            logging.info(f"    Expert {expert_idx + 1}/{num_experts} done")
                    logging.info(f"    down bit distribution: {dict(sorted(bits_count.items()))}")
            
        layers[layer_idx] = layer.cpu()
        torch.cuda.empty_cache()
        del layer
            
    cleanup_memory(verbos=True)
    logging.info('-----RTN Quantization Done-----\n')
    return quantizers


# ============================================================================
# Unified quantization interface
# ============================================================================
def quantize_model(
    model: nn.Module,
    method: str = "gptq",
    dataloader: Optional[List] = None,
    dev: torch.device = torch.device("cuda"),
    nsamples: int = 128,
    weight_quant_params: Optional[Dict] = None,
    mix_w_bits: bool = False,
    mix_w_bits_dict: Optional[Dict] = None,
    keep_min: bool = False,
    skip_rare_expert: bool = False,
    percentage: float = 0.1,
    expert_protect_map: Optional[Dict[int, set]] = None,
    protect_bits: int = 4,
    quantize_shared_experts: bool = False,
    quantize_vision: bool = False,
    quantize_vision_projector: bool = False,
    vision_bits: int = 4,
    expert_bit_map: Optional[Dict[int, Dict[int, int]]] = None,
) -> Dict[str, WeightQuantizer]:
    """
    Unified interface for model quantization.
    
    Args:
        model: The model to quantize
        method: Quantization method ("gptq" or "rtn")
        dataloader: Calibration data loader (required for GPTQ)
        dev: Device to use for quantization
        nsamples: Number of calibration samples (for GPTQ)
        weight_quant_params: Weight quantization parameters
        mix_w_bits: Enable mixed-precision quantization
        mix_w_bits_dict: Bit-width configuration for different layer types
        keep_min: Enable minimum token balancing for MoE experts
        percentage: Minimum percentage of average tokens per expert (used when keep_min=True)
    
    Returns:
        Dictionary of quantizers for each layer
    """
    # Default weight quantization parameters
    if weight_quant_params is None:
        weight_quant_params = {
            'w_bits': 4,
            'w_groupsize': 128,
            'w_asym': False,
            'w_clip': True,
            'perchannel': True,
            'percdamp': 0.01,
            'act_order': False,
            'norm': 2.4,
            'grid': 100,
            'maxshrink': 0.8,
        }
    
    if method.lower() == "gptq":
        if dataloader is None:
            raise ValueError("GPTQ requires calibration dataloader")
        return gptq_fwrd(
            model=model,
            dataloader=dataloader,
            dev=dev,
            nsamples=nsamples,
            weight_quant_params=weight_quant_params,
            mix_w_bits=mix_w_bits,
            mix_w_bits_dict=mix_w_bits_dict,
            keep_min=keep_min,
            skip_rare_expert=skip_rare_expert,
            percentage=percentage,
            quantize_vision=quantize_vision,
            quantize_vision_projector=quantize_vision_projector,
            vision_bits=vision_bits,
            expert_bit_map=expert_bit_map,
        )
    elif method.lower() == "rtn":
        return rtn_fwrd(
            model=model,
            dev=dev,
            weight_quant_params=weight_quant_params,
            mix_w_bits=mix_w_bits,
            mix_w_bits_dict=mix_w_bits_dict,
            expert_protect_map=expert_protect_map,
            protect_bits=protect_bits,
            quantize_shared_experts=quantize_shared_experts,
            expert_bit_map=expert_bit_map,
        )
    else:
        raise ValueError(f"Unknown quantization method: {method}. Supported: 'gptq', 'rtn'")
