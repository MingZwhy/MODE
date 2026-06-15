"""
Fast GPTQ quantization with multi-GPU parallel expert quantization.

Drop-in replacement for gptq_utils.gptq_fwrd with parallelized MoE expert quantization.

Key optimization:
- Distributes expert-level GPTQ quantization across ALL available GPUs using ThreadPoolExecutor.
- Maintains algorithm correctness:
  * Layer order preserved (layer 0 before layer 1, etc.)
  * Within each layer: attention -> gate_up/gate+up -> down
  * Expert quantization is embarrassingly parallel within each stage.
- For each stage (e.g., gate_up quantization), experts are distributed round-robin across GPUs.
  Each GPU processes its assigned experts sequentially; all GPUs work in parallel.

Usage:
    from mllm_quant.quantization.gpt_utils_fast import quantize_model_fast
    quantizers = quantize_model_fast(model, method="gptq", dataloader=..., ...)
"""

import math
import time
import copy
import logging
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any
from tqdm.auto import tqdm

from .gptq_utils import (
    GPTQ,
    GPTQWeight,
    get_model_type,
    is_moe_layer,
    get_num_experts_from_layer,
    is_moe_3d_param_structure,
    is_moe_modulelist_structure,
    generate_sequential_for_layer,
    move_to_device,
    prepare_layer_kwargs,
    balance_expert_tokens,
    cleanup_memory,
    rtn_fwrd,
    strip_accelerate_dispatch_hooks,
    quantize_vision_encoder_gptq,
    quarot_kimi_expert_mid_pre_down,
    QWEN3_VL_MODEL,
    QWEN3_VL_MOE_MODEL,
    INTERNVL_MODEL,
    KIMIVL_MODEL,
)
from .quant_utils import WeightQuantizer, find_qlayers
from mllm_quant.expert_mixed_precision.bit_config import get_expert_bit

from mllm_quant.rotation.quarot_gptq_compat import quarot_moe_mid_pre_down


def _expert_wqp(base: Dict, expert_bit_map: Optional[Dict[int, Dict[int, int]]], layer_idx: int, expert_idx: int) -> Dict:
    out = copy.deepcopy(base)
    out["w_bits"] = get_expert_bit(expert_bit_map, layer_idx, expert_idx, out.get("w_bits", 4))
    if out.get("w_groupsize", -1) != -1:
        out["perchannel"] = False
    return out


# ============================================================================
# Multi-GPU parallel expert quantization helpers
# ============================================================================

def _quantize_single_expert_worker(task: dict, gpu_id: int):
    """
    Worker function: quantize a single expert weight on specified GPU.

    Supports two modes:
    1. Activation-based: pass 'activations' and 'in_features' to compute Hessian on target GPU.
    2. Pre-computed Hessian: pass 'H' and optionally 'nsamples' (for ModuleList GPTQ instances).

    Args:
        task: dict containing weight, activations/H, quantizer_config, etc.
        gpu_id: target CUDA device id.

    Returns:
        (task_id, quantized_weight_cpu, quantizer)
    """
    device = torch.device(f'cuda:{gpu_id}')
    task_id = task['task_id']
    weight = task['weight'].to(device)
    quantizer_config = task['quantizer_config']
    percdamp = task['percdamp']
    groupsize = task['groupsize']
    actorder = task['actorder']

    if task.get('H') is not None:
        # Pre-computed Hessian mode (ModuleList structure)
        gptq = GPTQWeight(weight)
        gptq.H = task['H'].to(device)
        gptq.nsamples = task.get('nsamples', 1)
    else:
        # Activation-based mode (3D parameter structure)
        in_features = task['in_features']
        activations = task.get('activations')
        if activations is not None and activations.shape[0] > 0:
            activations = activations.to(device)
        else:
            activations = torch.randn(128, in_features, device=device, dtype=weight.dtype)

        gptq = GPTQWeight(weight, in_features=in_features)
        gptq.add_batch(activations)
        del activations

    gptq.quantizer = WeightQuantizer()
    gptq.quantizer.configure(quantizer_config)

    Q = gptq.fasterquant(
        percdamp=percdamp,
        groupsize=groupsize,
        actorder=actorder,
    )

    Q_cpu = Q.cpu()
    quantizer = gptq.quantizer
    gptq.free()
    del weight, Q
    torch.cuda.empty_cache()

    return task_id, Q_cpu, quantizer


def parallel_quantize_experts(
    tasks: List[dict],
    num_gpus: int = None,
) -> Dict[Any, Tuple[torch.Tensor, WeightQuantizer]]:
    """
    Distribute expert quantization tasks across multiple GPUs using ThreadPoolExecutor.

    Each thread handles one GPU and processes its assigned experts sequentially.
    All GPU threads run concurrently, achieving true multi-GPU parallelism
    (CUDA ops release the GIL).

    Args:
        tasks: list of task dicts. Each must contain:
            - task_id: unique hashable identifier
            - weight: 2D tensor (out_features, in_features)
            - quantizer_config: dict for WeightQuantizer.configure()
            - percdamp, groupsize, actorder: GPTQ parameters
            And either:
            - activations: (num_tokens, in_features) tensor + in_features: int
            Or:
            - H: pre-computed Hessian tensor + nsamples: int
        num_gpus: number of GPUs to use (defaults to torch.cuda.device_count()).

    Returns:
        Dict mapping task_id -> (quantized_weight_cpu, quantizer)
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    num_gpus = max(1, num_gpus)

    if not tasks:
        return {}

    # Round-robin assignment of tasks to GPUs
    gpu_task_lists: List[List[Tuple[dict, int]]] = [[] for _ in range(num_gpus)]
    for i, task in enumerate(tasks):
        gpu_id = i % num_gpus
        gpu_task_lists[gpu_id].append((task, gpu_id))

    results: Dict[Any, Tuple[torch.Tensor, WeightQuantizer]] = {}

    def _process_gpu_tasks(gpu_tasks):
        """Process a batch of tasks assigned to one GPU (runs in a thread)."""
        local_results = {}
        for task, gid in gpu_tasks:
            task_id, Q_cpu, quantizer = _quantize_single_expert_worker(task, gid)
            local_results[task_id] = (Q_cpu, quantizer)
        return local_results

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for gpu_id in range(num_gpus):
            if gpu_task_lists[gpu_id]:
                futures.append(executor.submit(_process_gpu_tasks, gpu_task_lists[gpu_id]))

        for future in as_completed(futures):
            try:
                local_results = future.result()
                results.update(local_results)
            except Exception as e:
                logging.error(f"Expert quantization worker failed: {e}")
                raise

    elapsed = time.time() - t0
    logging.info(f"    Parallel quantization: {len(tasks)} experts on {num_gpus} GPUs in {elapsed:.1f}s")
    return results


# ============================================================================
# Fast GPTQ Forward pass (multi-GPU parallel expert quantization)
# ============================================================================

@torch.no_grad()
def gptq_fwrd_fast(
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
    quantize_shared_experts: bool = False,
    quantize_vision: bool = False,
    quantize_vision_projector: bool = False,
    vision_bits: int = 4,
    expert_bit_map: Optional[Dict[int, Dict[int, int]]] = None,
) -> Dict[str, WeightQuantizer]:
    """
    GPTQ quantization with multi-GPU parallel expert quantization.

    Same algorithm and correctness guarantees as gptq_fwrd, but parallelizes
    expert-level quantization across all available CUDA GPUs for significant
    speedup on fine-grained MoE models.

    Args:
        (same as gptq_fwrd in gptq_utils.py)

    Returns:
        Dictionary of quantizers for each layer.
    """
    logging.info('-----GPTQ Quantization (Fast / Multi-GPU)-----')

    num_gpus = torch.cuda.device_count()
    logging.info(f"Available GPUs for parallel expert quantization: {num_gpus}")

    # Safely get and set use_cache
    use_cache = getattr(model.config, 'use_cache', None)
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False

    model_type = get_model_type(model)
    logging.info(f"Model type: {model_type}")

    # Move entire model to CPU for clean state
    model = model.cpu()
    torch.cuda.empty_cache()
    strip_accelerate_dispatch_hooks(model)

    # Force quantization device to cuda:0
    if isinstance(dev, torch.device) and dev.type == 'cuda':
        dev = torch.device('cuda:0')
    elif isinstance(dev, str) and dev.startswith('cuda'):
        dev = torch.device('cuda:0')

    # ================================================================
    # Get model layers based on architecture
    # ================================================================
    if hasattr(model, 'model') and hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'layers'):
        layers = model.model.language_model.layers
        embed_tokens = model.model.language_model.embed_tokens
        norm = model.model.language_model.norm
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers'):
        layers = model.language_model.model.layers
        embed_tokens = model.language_model.model.embed_tokens
        norm = model.language_model.model.norm
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        embed_tokens = model.model.embed_tokens if hasattr(model.model, 'embed_tokens') else model.model.tok_embeddings
        norm = model.model.norm
    else:
        raise ValueError(f"Unsupported model architecture: {model_type}")

    for layer in layers:
        layer = layer.cpu()

    dtype = next(iter(model.parameters())).dtype

    if hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'hidden_size'):
        hidden_size = model.config.text_config.hidden_size
    elif hasattr(model.config, 'hidden_size'):
        hidden_size = model.config.hidden_size
    else:
        raise ValueError("Cannot find hidden_size in model config")

    # ================================================================
    # Calibration data collection (Catcher)
    # ================================================================
    inps = []
    cache = {
        'i': 0, 'attention_mask': None,
        'attention_masks': [], 'position_ids_list': [], 'position_embeddings_list': [],
    }

    logging.info(f"Device: {dev}")

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            if inp.dim() == 3 and inp.shape[0] == 1:
                inps.append(inp.squeeze(0).cpu())
            else:
                inps.append(inp.cpu())
            cache['i'] += 1
            cache['attention_masks'].append(kwargs.get('attention_mask', None))
            cache['position_ids_list'].append(kwargs.get('position_ids', None))
            if 'position_embeddings' in kwargs:
                cache['position_embeddings_list'].append(kwargs['position_embeddings'])
            raise ValueError

    model = model.cpu()

    # Move necessary components to device for calibration data collection
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        model.model.language_model = model.model.language_model.to(dev)
        embed_tokens = model.model.language_model.embed_tokens
        norm = model.model.language_model.norm
    elif hasattr(model, 'model'):
        if hasattr(model.model, 'embed_tokens'):
            model.model.embed_tokens = model.model.embed_tokens.to(dev)
            embed_tokens = model.model.embed_tokens
        elif hasattr(model.model, 'tok_embeddings'):
            model.model.tok_embeddings = model.model.tok_embeddings.to(dev)
            embed_tokens = model.model.tok_embeddings
        model.model.norm = model.model.norm.to(dev)
        norm = model.model.norm
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
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

    print("begin first forward pass")

    # 使用 tqdm 显示简单美观的进度条
    dataloader_iter = tqdm(dataloader, desc="Calib first forward", unit="batch")

    for batch in dataloader_iter:
        try:
            if isinstance(batch, dict):
                inputs = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                inputs = _fix_batch_dims(inputs)
                model(**inputs)
            elif isinstance(batch, (tuple, list)):
                input_ids = batch[0]
                if isinstance(input_ids, dict):
                    inputs = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in input_ids.items()}
                    inputs = _fix_batch_dims(inputs)
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
        except ValueError:
            # 单个 batch 出错时跳过，继续下一个
            pass
    layers[0] = layers[0].module

    print("end first forward pass")

    # Move everything back to CPU
    layers[0] = layers[0].cpu()
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        model.model.language_model = model.model.language_model.cpu()
        embed_tokens = model.model.language_model.embed_tokens
        norm = model.model.language_model.norm
    elif hasattr(model, 'model'):
        if hasattr(model.model, 'embed_tokens'):
            model.model.embed_tokens = model.model.embed_tokens.cpu()
            embed_tokens = model.model.embed_tokens
        elif hasattr(model.model, 'tok_embeddings'):
            model.model.tok_embeddings = model.model.tok_embeddings.cpu()
            embed_tokens = model.model.tok_embeddings
        model.model.norm = model.model.norm.cpu()
        norm = model.model.norm
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        model.language_model.model = model.language_model.model.cpu()
        embed_tokens = model.language_model.model.embed_tokens
        norm = model.language_model.model.norm

    # Move visual encoder back to CPU
    for _parent, _attr in _visual_moved:
        setattr(_parent, _attr, getattr(_parent, _attr).cpu())

    model = model.cpu()
    for i in range(len(layers)):
        layers[i] = layers[i].cpu()
    torch.cuda.empty_cache()

    outs = [None] * len(inps)
    nsamples = len(inps)
    logging.info(f"Collected {nsamples} calibration samples")

    # ================================================================
    # Layer-by-layer quantization
    # ================================================================
    num_layers = len(layers)

    for layer_idx in range(num_layers):
        layer_t0 = time.time()
        logging.info(f'\nLayer {layer_idx}:')
        layers[layer_idx] = layers[layer_idx].to(dev)
        layer = layers[layer_idx]

        layer_is_moe = is_moe_layer(layer)
        if layer_is_moe:
            logging.info(f"  (MoE layer)")

        full = find_qlayers(layer, layers=[torch.nn.Linear])

        total_layer_tokens = sum(inp.shape[0] if len(inp.shape) >= 2 else inp.shape[0] for inp in inps)
        logging.info(f"Layer {layer_idx}: total input tokens = {total_layer_tokens}")

        sequential = generate_sequential_for_layer(layer)

        attn = layer.self_attn if hasattr(layer, 'self_attn') else None
        is_kimi_vl_attention = (
            attn is not None and
            hasattr(attn, 'kv_a_proj_with_mqa') and
            hasattr(attn, 'kv_b_proj')
        )

        # ============================================================
        # GPTQ quantization for nn.Linear layers (attention + dense MLP)
        # (Same as original gptq_fwrd - sequential within each group)
        # ============================================================
        sequential_idx = 0
        for names in sequential:
            subset = {n: full[n] for n in names if n in full}
            if not subset:
                sequential_idx += 1
                continue

            is_o_proj = 'self_attn.o_proj' in names
            is_down_proj = 'mlp.down_proj' in names
            is_kv_b_proj = 'self_attn.kv_b_proj' in names and len(names) == 1
            is_standard_qkv_group = (
                'self_attn.q_proj' in names and 'self_attn.k_proj' in names and 'self_attn.v_proj' in names
            )
            is_kimi_vl_qkv_a_group = (
                'self_attn.q_proj' in names and 'self_attn.kv_a_proj_with_mqa' in names and
                'self_attn.kv_b_proj' not in names
            )

            if is_kv_b_proj:
                logging.info(f"    [Re-forward] Collecting kv_b_proj inputs using quantized q_proj+kv_a_proj_with_mqa")
            elif is_o_proj:
                if is_kimi_vl_attention:
                    logging.info(f"    [Re-forward] Collecting o_proj inputs using quantized q_proj+kv_a_proj_with_mqa+kv_b_proj")
                else:
                    logging.info(f"    [Re-forward] Collecting o_proj inputs using quantized qkv")
            elif is_down_proj:
                logging.info(f"    [Re-forward] Collecting down_proj inputs using quantized up_proj+gate_proj")

            gptq = {}
            for name in subset:
                weight_quant_params_copy = copy.deepcopy(weight_quant_params)
                if mix_w_bits and mix_w_bits_dict:
                    if name in ['mlp.up_proj', 'mlp.gate_proj']:
                        weight_quant_params_copy['w_bits'] = mix_w_bits_dict.get('up_and_gate', weight_quant_params['w_bits'])
                    elif name in ['mlp.down_proj']:
                        weight_quant_params_copy['w_bits'] = mix_w_bits_dict.get('down', weight_quant_params['w_bits'])
                    elif name in ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj', 'self_attn.o_proj',
                                  'self_attn.kv_a_proj_with_mqa', 'self_attn.kv_b_proj']:
                        weight_quant_params_copy['w_bits'] = mix_w_bits_dict.get('attn', weight_quant_params['w_bits'])

                if 'lm_head' in name or 'output' in name:
                    logging.info(f"  Skip quantization for {name}")
                    continue
                if name == 'mlp.gate' or 'router' in name.lower():
                    logging.info(f"  Skip MoE router: {name}")
                    continue

                logging.info(f'  {name} ({weight_quant_params_copy["w_bits"]} bits)')
                gptq[name] = GPTQ(subset[name])
                if hasattr(subset[name], "weight_quantizer"):
                    gptq[name].quantizer = subset[name].weight_quantizer
                else:
                    gptq[name].quantizer = WeightQuantizer()
                gptq[name].quantizer.configure(weight_quant_params_copy)

            if not gptq:
                sequential_idx += 1
                continue

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in gptq:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

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

            for h in handles:
                h.remove()

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

        # ============================================================
        # MoE expert quantization (PARALLELIZED across GPUs)
        # ============================================================
        if layer_is_moe:
            num_experts = get_num_experts_from_layer(layer)
            experts = layer.mlp.experts

            # ----------------------------------------------------------
            # ModuleList MoE structure (e.g. InternVL)
            # ----------------------------------------------------------
            if is_moe_modulelist_structure(experts):
                logging.info(f"  MoE ModuleList structure detected: {num_experts} experts")

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

                # Step 1: Create GPTQ instances for all experts
                for expert_idx in range(num_experts):
                    expert = experts[expert_idx]
                    if expert is None:
                        continue
                    expert_wqp_gate_up = _expert_wqp(weight_quant_params_gate_up, expert_bit_map, layer_idx, expert_idx)
                    expert_wqp_down = _expert_wqp(weight_quant_params_down, expert_bit_map, layer_idx, expert_idx)
                    expert_bits_gate_up = expert_wqp_gate_up["w_bits"]
                    expert_bits_down = expert_wqp_down["w_bits"]
                    expert_layers = find_qlayers(expert, layers=[torch.nn.Linear])
                    gate_proj_layer = expert_layers.get('gate_proj')
                    up_proj_layer = expert_layers.get('up_proj')
                    down_proj_layer = expert_layers.get('down_proj')

                    if gate_proj_layer is not None and expert_bits_gate_up < 16:
                        expert_gate_layers[expert_idx] = gate_proj_layer
                        expert_gptq_gate[expert_idx] = GPTQ(gate_proj_layer)
                        expert_gptq_gate[expert_idx].quantizer = WeightQuantizer()
                        expert_gptq_gate[expert_idx].quantizer.configure(expert_wqp_gate_up)

                    if up_proj_layer is not None and expert_bits_gate_up < 16:
                        expert_up_layers[expert_idx] = up_proj_layer
                        expert_gptq_up[expert_idx] = GPTQ(up_proj_layer)
                        expert_gptq_up[expert_idx].quantizer = WeightQuantizer()
                        expert_gptq_up[expert_idx].quantizer.configure(expert_wqp_gate_up)

                    if down_proj_layer is not None and expert_bits_down < 16:
                        expert_down_layers[expert_idx] = down_proj_layer
                        expert_gptq_down[expert_idx] = GPTQ(down_proj_layer)
                        expert_gptq_down[expert_idx].quantizer = WeightQuantizer()
                        expert_gptq_down[expert_idx].quantizer.configure(expert_wqp_down)

                # Step 2: Single forward pass to collect activations for gate_proj and up_proj
                if expert_gate_layers or expert_up_layers:
                    handles = []
                    expert_token_counts = {}
                    expert_inputs_dict = {idx: [] for idx in range(num_experts)}
                    all_layer_tokens_list = []

                    top_k = getattr(layer.mlp.gate, 'top_k', None)
                    if top_k is None:
                        if hasattr(layer.mlp, 'config') and hasattr(layer.mlp.config, 'num_experts_per_tok'):
                            top_k = layer.mlp.config.num_experts_per_tok
                        elif hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'num_experts_per_tok'):
                            top_k = model.config.text_config.num_experts_per_tok
                        elif hasattr(model.config, 'num_experts_per_tok'):
                            top_k = model.config.num_experts_per_tok
                        else:
                            top_k = 1
                            logging.warning(f"  Cannot determine top_k, defaulting to 1")

                    for expert_idx, linear_layer in expert_gate_layers.items():
                        def add_batch_gate(idx):
                            def tmp(_, inp, out):
                                inp_data = inp[0].data
                                if len(inp_data.shape) == 3:
                                    num_tokens = inp_data.shape[0] * inp_data.shape[1]
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

                    # Build token pool for balancing
                    hidden_dim = None
                    if expert_inputs_dict and any(expert_inputs_dict.values()):
                        for eidx in range(num_experts):
                            if expert_inputs_dict[eidx]:
                                hidden_dim = expert_inputs_dict[eidx][0].shape[-1]
                                break
                    if hidden_dim is None:
                        if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp.gate, 'weight'):
                            hidden_dim = layer.mlp.gate.weight.shape[-1]
                        elif expert_gate_layers:
                            first_gate = list(expert_gate_layers.values())[0]
                            hidden_dim = getattr(first_gate, 'in_features', 4096)
                        else:
                            hidden_dim = 4096

                    expert_token_pool_list = []
                    for eidx in range(num_experts):
                        if expert_inputs_dict[eidx]:
                            expert_token_pool_list.extend(expert_inputs_dict[eidx])

                    if expert_token_pool_list:
                        all_layer_tokens = torch.cat(expert_token_pool_list, dim=0)
                    else:
                        all_layer_tokens = torch.empty(0, hidden_dim, device=dev, dtype=torch.float32)

                    total_unique_tokens = sum(
                        inps[j].reshape(-1, inps[j].shape[-1]).shape[0] if len(inps[j].shape) == 3 else inps[j].shape[0]
                        for j in range(nsamples)
                    )
                    logging.info(f"  Token pool: pool_size={all_layer_tokens.shape[0]}, unique_tokens={total_unique_tokens}")

                    # Balance / skip rare experts
                    padded_experts = set()
                    skipped_experts = set()
                    padded_gate_inputs = {}

                    if (keep_min or skip_rare_expert) and all_layer_tokens.shape[0] > 0:
                        total_tokens = total_unique_tokens
                        pool_size = all_layer_tokens.shape[0]
                        avg_tokens_per_expert = (total_tokens * top_k) / num_experts
                        min_tokens = int(avg_tokens_per_expert * percentage)

                        if keep_min:
                            logging.info(f"  Balancing expert tokens (top_k={top_k}, percentage={percentage})")
                            for expert_idx in expert_gate_layers.keys():
                                num_expert_tokens = expert_token_counts.get(expert_idx, 0)
                                if num_expert_tokens < min_tokens:
                                    num_to_pad = min_tokens - num_expert_tokens
                                    random_indices = torch.randint(0, pool_size, (num_to_pad,), device=dev)
                                    padded_tokens = all_layer_tokens[random_indices]
                                    gate_layer_l = expert_gate_layers[expert_idx]
                                    up_layer_l = expert_up_layers.get(expert_idx)
                                    with torch.no_grad():
                                        padded_gate_output = gate_layer_l(padded_tokens)
                                        expert_gptq_gate[expert_idx].add_batch(
                                            padded_tokens.unsqueeze(0) if len(padded_tokens.shape) == 2 else padded_tokens,
                                            padded_gate_output)
                                        if up_layer_l is not None and expert_idx in expert_gptq_up:
                                            padded_up_output = up_layer_l(padded_tokens)
                                            expert_gptq_up[expert_idx].add_batch(
                                                padded_tokens.unsqueeze(0) if len(padded_tokens.shape) == 2 else padded_tokens,
                                                padded_up_output)
                                    original_inputs = torch.cat(expert_inputs_dict[expert_idx], dim=0) if expert_inputs_dict[expert_idx] else torch.empty(0, padded_tokens.shape[-1], device=dev, dtype=padded_tokens.dtype)
                                    padded_gate_inputs[expert_idx] = torch.cat([original_inputs, padded_tokens], dim=0)
                                    logging.info(f"    Expert {expert_idx}: padded from {num_expert_tokens} to {min_tokens} tokens")
                                    padded_experts.add(expert_idx)
                                else:
                                    if expert_inputs_dict[expert_idx]:
                                        padded_gate_inputs[expert_idx] = torch.cat(expert_inputs_dict[expert_idx], dim=0)
                        elif skip_rare_expert:
                            logging.info(f"  Skipping rare experts (top_k={top_k}, percentage={percentage}, min_tokens={min_tokens})")
                            for expert_idx in expert_gate_layers.keys():
                                num_expert_tokens = expert_token_counts.get(expert_idx, 0)
                                if num_expert_tokens < min_tokens:
                                    skipped_experts.add(expert_idx)
                                    logging.info(f"    Expert {expert_idx}: skipping ({num_expert_tokens} tokens < {min_tokens})")
                                else:
                                    if expert_inputs_dict[expert_idx]:
                                        padded_gate_inputs[expert_idx] = torch.cat(expert_inputs_dict[expert_idx], dim=0)

                    if not keep_min and not skip_rare_expert:
                        for expert_idx in expert_gate_layers.keys():
                            if expert_inputs_dict.get(expert_idx):
                                padded_gate_inputs[expert_idx] = torch.cat(expert_inputs_dict[expert_idx], dim=0)

                    # Print token counts
                    all_expert_indices = sorted(set(expert_gate_layers.keys()) | set(expert_up_layers.keys()))
                    for expert_idx in all_expert_indices:
                        token_count = expert_token_counts.get(expert_idx, 0)
                        if not (keep_min and expert_idx in padded_experts) and not (skip_rare_expert and expert_idx in skipped_experts):
                            logging.info(f"    Expert {expert_idx}: {token_count} tokens")

                    # ================================================
                    # Step 3: PARALLEL quantize gate_proj and up_proj
                    # ================================================
                    logging.info(f"  [Parallel] Quantizing gate_proj + up_proj across {num_gpus} GPUs")
                    tasks = []

                    for expert_idx in sorted(expert_gate_layers.keys()):
                        if expert_idx in skipped_experts:
                            dummy_quantizer = WeightQuantizer()
                            dummy_quantizer.configure({'w_bits': 16})
                            quantizers[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj'] = dummy_quantizer
                            expert_gptq_gate[expert_idx].free()
                            continue
                        tasks.append({
                            'task_id': ('gate', expert_idx),
                            'weight': expert_gptq_gate[expert_idx].layer.weight.data.clone(),
                            'H': expert_gptq_gate[expert_idx].H.clone(),
                            'nsamples': expert_gptq_gate[expert_idx].nsamples,
                            'quantizer_config': _expert_wqp(weight_quant_params_gate_up, expert_bit_map, layer_idx, expert_idx),
                            'percdamp': weight_quant_params['percdamp'],
                            'groupsize': weight_quant_params['w_groupsize'],
                            'actorder': weight_quant_params.get('act_order', False),
                        })

                    for expert_idx in sorted(expert_up_layers.keys()):
                        if expert_idx in skipped_experts:
                            dummy_quantizer = WeightQuantizer()
                            dummy_quantizer.configure({'w_bits': 16})
                            quantizers[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj'] = dummy_quantizer
                            expert_gptq_up[expert_idx].free()
                            continue
                        tasks.append({
                            'task_id': ('up', expert_idx),
                            'weight': expert_gptq_up[expert_idx].layer.weight.data.clone(),
                            'H': expert_gptq_up[expert_idx].H.clone(),
                            'nsamples': expert_gptq_up[expert_idx].nsamples,
                            'quantizer_config': _expert_wqp(weight_quant_params_gate_up, expert_bit_map, layer_idx, expert_idx),
                            'percdamp': weight_quant_params['percdamp'],
                            'groupsize': weight_quant_params['w_groupsize'],
                            'actorder': weight_quant_params.get('act_order', False),
                        })

                    # Free GPTQ instances on cuda:0 before parallel quantization (save memory)
                    for expert_idx in expert_gptq_gate:
                        if expert_idx not in skipped_experts:
                            expert_gptq_gate[expert_idx].free()
                    for expert_idx in expert_gptq_up:
                        if expert_idx not in skipped_experts:
                            expert_gptq_up[expert_idx].free()
                    torch.cuda.empty_cache()

                    results = parallel_quantize_experts(tasks, num_gpus)

                    # Write back quantized weights
                    for (proj_type, expert_idx), (Q_cpu, quantizer) in results.items():
                        target_device = dev  # layer is on dev (cuda:0)
                        if proj_type == 'gate':
                            expert_gate_layers[expert_idx].weight.data = Q_cpu.to(target_device).reshape(
                                expert_gate_layers[expert_idx].weight.shape)
                            quantizers[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj'] = quantizer
                        elif proj_type == 'up':
                            expert_up_layers[expert_idx].weight.data = Q_cpu.to(target_device).reshape(
                                expert_up_layers[expert_idx].weight.shape)
                            quantizers[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj'] = quantizer
                    del results, tasks
                    torch.cuda.empty_cache()

                    # ================================================
                    # Step 4: Re-forward for down_proj inputs
                    # ================================================
                    if expert_down_layers:
                        logging.info(f"    [Re-forward] Collecting down_proj inputs using quantized gate_proj+up_proj")
                        for expert_idx in sorted(expert_down_layers.keys()):
                            gate_layer_l = expert_gate_layers.get(expert_idx)
                            up_layer_l = expert_up_layers.get(expert_idx)
                            down_layer_l = expert_down_layers[expert_idx]

                            if gate_layer_l and up_layer_l:
                                if expert_idx in skipped_experts:
                                    continue
                                if expert_idx in padded_gate_inputs and padded_gate_inputs[expert_idx].shape[0] > 0:
                                    gate_inputs = padded_gate_inputs[expert_idx]
                                elif expert_inputs_dict.get(expert_idx) and expert_inputs_dict[expert_idx]:
                                    gate_inputs = torch.cat(expert_inputs_dict[expert_idx], dim=0)
                                else:
                                    logging.warning(f"    Expert {expert_idx}: no inputs found, skipping")
                                    continue

                                with torch.no_grad():
                                    gate_output = gate_layer_l(gate_inputs)
                                    gate_act = torch.nn.functional.silu(gate_output)
                                    up_output = up_layer_l(gate_inputs)
                                    gated_output = gate_act * up_output
                                    ex_mod = (
                                        layer.mlp.experts[expert_idx]
                                        if expert_idx < len(layer.mlp.experts)
                                        else None
                                    )
                                    gated_output = quarot_kimi_expert_mid_pre_down(gated_output, ex_mod)
                                    down_output = down_layer_l(gated_output)
                                    expert_gptq_down[expert_idx].add_batch(gated_output, down_output)

                        # ================================================
                        # Step 5: PARALLEL quantize down_proj
                        # ================================================
                        logging.info(f"  [Parallel] Quantizing down_proj across {num_gpus} GPUs")
                        tasks = []
                        for expert_idx in sorted(expert_down_layers.keys()):
                            if expert_idx in skipped_experts:
                                dummy_quantizer = WeightQuantizer()
                                dummy_quantizer.configure({'w_bits': 16})
                                quantizers[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj'] = dummy_quantizer
                                expert_gptq_down[expert_idx].free()
                                continue
                            tasks.append({
                                'task_id': expert_idx,
                                'weight': expert_gptq_down[expert_idx].layer.weight.data.clone(),
                                'H': expert_gptq_down[expert_idx].H.clone(),
                                'nsamples': expert_gptq_down[expert_idx].nsamples,
                                'quantizer_config': _expert_wqp(weight_quant_params_down, expert_bit_map, layer_idx, expert_idx),
                                'percdamp': weight_quant_params['percdamp'],
                                'groupsize': weight_quant_params['w_groupsize'],
                                'actorder': weight_quant_params.get('act_order', False),
                            })

                        for expert_idx in expert_gptq_down:
                            if expert_idx not in skipped_experts:
                                expert_gptq_down[expert_idx].free()
                        torch.cuda.empty_cache()

                        results = parallel_quantize_experts(tasks, num_gpus)

                        for expert_idx, (Q_cpu, quantizer) in results.items():
                            expert_down_layers[expert_idx].weight.data = Q_cpu.to(dev).reshape(
                                expert_down_layers[expert_idx].weight.shape)
                            quantizers[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj'] = quantizer
                        del results, tasks

                    # Step 6: Final re-forward
                    logging.info(f"    [Re-forward] Re-forwarding after quantizing all experts")
                    for j in range(nsamples):
                        inp_j = inps[j].to(dev).unsqueeze(0)
                        attn_mask_j, pos_ids_j, pos_emb_j = prepare_layer_kwargs(cache, j, dev)
                        if pos_emb_j is not None:
                            layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j, position_embeddings=pos_emb_j)
                        else:
                            layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j)

                # Cleanup ModuleList MoE routed experts
                del expert_gate_layers, expert_up_layers, expert_down_layers
                del expert_gptq_gate, expert_gptq_up, expert_gptq_down
                del expert_inputs_dict, expert_token_counts
                try:
                    del all_layer_tokens, padded_gate_inputs, expert_token_pool_list, all_layer_tokens_list
                except NameError:
                    pass
                gc.collect()
                torch.cuda.empty_cache()

                # ============================================================
                # Shared experts GPTQ quantization (ModuleList path)
                # ============================================================
                se = getattr(layer.mlp, 'shared_experts', None)
                if quantize_shared_experts and se is not None:
                    se_layers = find_qlayers(se, layers=[torch.nn.Linear])
                    se_gate = se_layers.get('gate_proj')
                    se_up = se_layers.get('up_proj')
                    se_down = se_layers.get('down_proj')

                    se_wqp_gate_up = copy.deepcopy(weight_quant_params)
                    se_wqp_down = copy.deepcopy(weight_quant_params)
                    if mix_w_bits and mix_w_bits_dict:
                        se_wqp_gate_up['w_bits'] = mix_w_bits_dict.get('up_and_gate', weight_quant_params['w_bits'])
                        se_wqp_down['w_bits'] = mix_w_bits_dict.get('down', weight_quant_params['w_bits'])
                    if se_wqp_down.get('w_groupsize', -1) != -1:
                        se_wqp_down['perchannel'] = False
                    se_bits_gu = se_wqp_gate_up['w_bits']
                    se_bits_d = se_wqp_down['w_bits']

                    if se_gate and se_up and se_bits_gu < 16:
                        logging.info(f"  [shared_experts] Quantizing gate_proj+up_proj ({se_bits_gu}b) + down_proj ({se_bits_d}b)")

                        se_gptq_gate = GPTQ(se_gate)
                        se_gptq_gate.quantizer = WeightQuantizer()
                        se_gptq_gate.quantizer.configure(se_wqp_gate_up)
                        se_gptq_up = GPTQ(se_up)
                        se_gptq_up.quantizer = WeightQuantizer()
                        se_gptq_up.quantizer.configure(se_wqp_gate_up)

                        handles_se = []
                        def _se_hook_gate(_, inp, out):
                            se_gptq_gate.add_batch(inp[0].data, out.data)
                        def _se_hook_up(_, inp, out):
                            se_gptq_up.add_batch(inp[0].data, out.data)
                        handles_se.append(se_gate.register_forward_hook(_se_hook_gate))
                        handles_se.append(se_up.register_forward_hook(_se_hook_up))

                        for j in range(nsamples):
                            inp_j = inps[j].to(dev).unsqueeze(0)
                            attn_mask_j, pos_ids_j, pos_emb_j = prepare_layer_kwargs(cache, j, dev)
                            if pos_emb_j is not None:
                                layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j, position_embeddings=pos_emb_j)
                            else:
                                layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j)
                        for h in handles_se:
                            h.remove()

                        _se_chol_log = f"    [shared_experts L{layer_idx}] "
                        logging.info(f"    GPTQ quantizing shared_experts.gate_proj")
                        se_gptq_gate.fasterquant(
                            percdamp=weight_quant_params['percdamp'],
                            groupsize=weight_quant_params['w_groupsize'],
                            actorder=weight_quant_params.get('act_order', False),
                            cholesky_autodamp_retry=True,
                            cholesky_max_retries=12,
                            cholesky_log_prefix=_se_chol_log,
                        )
                        quantizers[f'model.layers.{layer_idx}.mlp.shared_experts.gate_proj'] = se_gptq_gate.quantizer
                        se_gptq_gate.free()

                        logging.info(f"    GPTQ quantizing shared_experts.up_proj")
                        se_gptq_up.fasterquant(
                            percdamp=weight_quant_params['percdamp'],
                            groupsize=weight_quant_params['w_groupsize'],
                            actorder=weight_quant_params.get('act_order', False),
                            cholesky_autodamp_retry=True,
                            cholesky_max_retries=12,
                            cholesky_log_prefix=_se_chol_log,
                        )
                        quantizers[f'model.layers.{layer_idx}.mlp.shared_experts.up_proj'] = se_gptq_up.quantizer
                        se_gptq_up.free()

                        # Re-forward for down_proj Hessian
                        if se_down and se_bits_d < 16:
                            se_gptq_down = GPTQ(se_down)
                            se_gptq_down.quantizer = WeightQuantizer()
                            se_gptq_down.quantizer.configure(se_wqp_down)

                            se_inputs_for_down = []
                            def _se_capture_gate_input(_, inp, out):
                                se_inputs_for_down.append(inp[0].data.detach().clone())
                            h_cap = se_gate.register_forward_hook(_se_capture_gate_input)
                            for j in range(nsamples):
                                inp_j = inps[j].to(dev).unsqueeze(0)
                                attn_mask_j, pos_ids_j, pos_emb_j = prepare_layer_kwargs(cache, j, dev)
                                if pos_emb_j is not None:
                                    layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j, position_embeddings=pos_emb_j)
                                else:
                                    layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j)
                            h_cap.remove()

                            logging.info(f"    [Re-forward] Collecting shared_experts.down_proj inputs")
                            # Each sample is (1, seq, hidden); seq differs across calibration batches — flatten to tokens then cat.
                            se_flat = []
                            for t in se_inputs_for_down:
                                t = t.detach()
                                if t.dim() == 3:
                                    t = t.reshape(-1, t.shape[-1])
                                elif t.dim() == 2:
                                    pass
                                else:
                                    t = t.reshape(-1, t.shape[-1])
                                se_flat.append(t)
                            all_se_inp = torch.cat(se_flat, dim=0)
                            del se_inputs_for_down, se_flat
                            with torch.no_grad():
                                gate_out = se_gate(all_se_inp)
                                gate_act = torch.nn.functional.silu(gate_out)
                                up_out = se_up(all_se_inp)
                                gated = gate_act * up_out
                                del gate_out, gate_act, up_out, all_se_inp
                                gated = quarot_kimi_expert_mid_pre_down(gated, se)
                                down_out = se_down(gated)
                                se_gptq_down.add_batch(gated, down_out)
                                del gated, down_out

                            logging.info(f"    GPTQ quantizing shared_experts.down_proj")
                            se_gptq_down.fasterquant(
                                percdamp=weight_quant_params['percdamp'],
                                groupsize=weight_quant_params['w_groupsize'],
                                actorder=weight_quant_params.get('act_order', False),
                                cholesky_autodamp_retry=True,
                                cholesky_max_retries=12,
                                cholesky_log_prefix=_se_chol_log,
                            )
                            quantizers[f'model.layers.{layer_idx}.mlp.shared_experts.down_proj'] = se_gptq_down.quantizer
                            se_gptq_down.free()

                        logging.info(f"  [shared_experts] Done")
                        gc.collect()
                        torch.cuda.empty_cache()

                # Collect quantized output for next layer
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

                layers[layer_idx] = layers[layer_idx].cpu()
                del layer
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                inps, outs = outs, inps
                elapsed = time.time() - layer_t0
                logging.info(f"  [ModuleList MoE] Layer {layer_idx} done in {elapsed:.1f}s")
                continue

            elif is_moe_3d_param_structure(experts):
                # ----------------------------------------------------------
                # 3D Parameter MoE structure (Qwen3-VL-MoE, DeepseekV3/Kimi-VL)
                # ----------------------------------------------------------
                gate_up_proj = experts.gate_up_proj
                down_proj = experts.down_proj

                hidden_dim = gate_up_proj.shape[1]
                intermediate_dim = down_proj.shape[1]

                logging.info(f"  gate_up_proj shape: {gate_up_proj.shape}")
                logging.info(f"  down_proj shape: {down_proj.shape}")
                logging.info(f"  hidden_dim: {hidden_dim}, intermediate_dim: {intermediate_dim}")

                # Collect per-expert inputs via hooked forward
                expert_inputs = {i: [] for i in range(num_experts)}
                expert_down_inputs = {i: [] for i in range(num_experts)}
                all_layer_tokens_list = []

                top_k = getattr(layer.mlp.gate, 'top_k', None)
                if top_k is None:
                    if hasattr(layer.mlp, 'config') and hasattr(layer.mlp.config, 'num_experts_per_tok'):
                        top_k = layer.mlp.config.num_experts_per_tok
                    elif hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'num_experts_per_tok'):
                        top_k = model.config.text_config.num_experts_per_tok
                    elif hasattr(model.config, 'num_experts_per_tok'):
                        top_k = model.config.num_experts_per_tok
                    else:
                        top_k = 1
                        logging.warning(f"  Cannot determine top_k, defaulting to 1")

                original_experts_forward = experts.forward

                def hooked_experts_forward(hidden_states, routing_weights, router_indices):
                    batch_size = hidden_states.shape[0]
                    hidden_states = hidden_states.reshape(-1, experts.hidden_size)
                    all_layer_tokens_list.append(hidden_states.detach().clone())

                    next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
                    with torch.no_grad():
                        expert_mask = torch.nn.functional.one_hot(router_indices.long(), num_classes=experts.num_experts)
                        expert_mask = expert_mask.permute(2, 1, 0)
                        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

                    for expert_idx_tensor in expert_hit[:]:
                        expert_idx_val = expert_idx_tensor[0].item()
                        with torch.no_grad():
                            _, token_idx = torch.where(expert_mask[expert_idx_val])
                        current_state = hidden_states[token_idx]
                        if len(current_state) > 0:
                            expert_inputs[expert_idx_val].append(current_state.detach().clone())
                        gate_up = current_state @ experts.gate_up_proj[expert_idx_val]
                        gate, up = gate_up.chunk(2, dim=-1)
                        gated_output = up * experts.act_fn(gate)
                        gated_output = quarot_moe_mid_pre_down(gated_output, experts)
                        if len(gated_output) > 0:
                            expert_down_inputs[expert_idx_val].append(gated_output.detach().clone())
                        out = gated_output @ experts.down_proj[expert_idx_val]
                        weighted_output = out * routing_weights[token_idx, expert_idx_val, None]
                        next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))

                    next_states = next_states.view(batch_size, -1, experts.hidden_size)
                    return next_states

                experts.forward = hooked_experts_forward

                for j in range(nsamples):
                    inp_j = inps[j].to(dev).unsqueeze(0)
                    attn_mask_j, pos_ids_j, pos_emb_j = prepare_layer_kwargs(cache, j, dev)
                    if pos_emb_j is not None:
                        layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j, position_embeddings=pos_emb_j)
                    else:
                        layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j)

                experts.forward = original_experts_forward

                # Concatenate all layer tokens
                if all_layer_tokens_list:
                    all_layer_tokens = torch.cat(all_layer_tokens_list, dim=0)
                else:
                    all_layer_tokens = torch.empty(0, hidden_dim, device=dev, dtype=gate_up_proj.dtype)

                # Balance / skip rare experts
                padded_experts_3d = set()
                skipped_experts_3d = set()
                original_expert_inputs = {}

                if (keep_min or skip_rare_expert) and all_layer_tokens.shape[0] > 0:
                    total_tokens = all_layer_tokens.shape[0]
                    avg_tokens_per_expert = (total_tokens * top_k) / num_experts
                    min_tokens = int(avg_tokens_per_expert * percentage)

                    original_token_counts = {}
                    for expert_idx in range(num_experts):
                        if expert_inputs[expert_idx]:
                            original_token_counts[expert_idx] = sum(t.shape[0] for t in expert_inputs[expert_idx])
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
                        for expert_idx in range(num_experts):
                            if expert_inputs.get(expert_idx) is not None:
                                new_count = expert_inputs[expert_idx].shape[0]
                                if new_count > original_token_counts.get(expert_idx, 0):
                                    padded_experts_3d.add(expert_idx)
                    elif skip_rare_expert:
                        logging.info(f"  Skipping rare experts (top_k={top_k}, percentage={percentage}, min_tokens={min_tokens})")
                        for expert_idx in range(num_experts):
                            if original_token_counts[expert_idx] < min_tokens:
                                skipped_experts_3d.add(expert_idx)
                                logging.info(f"    Expert {expert_idx}: skipping ({original_token_counts[expert_idx]} tokens < {min_tokens})")
                        for expert_idx in range(num_experts):
                            if isinstance(expert_inputs.get(expert_idx), list):
                                if expert_inputs[expert_idx]:
                                    expert_inputs[expert_idx] = torch.cat(expert_inputs[expert_idx], dim=0)
                                else:
                                    expert_inputs[expert_idx] = None
                else:
                    for expert_idx in range(num_experts):
                        if expert_inputs[expert_idx]:
                            expert_inputs[expert_idx] = torch.cat(expert_inputs[expert_idx], dim=0)
                            original_expert_inputs[expert_idx] = expert_inputs[expert_idx]
                        else:
                            expert_inputs[expert_idx] = None
                            original_expert_inputs[expert_idx] = None

                # ================================================
                # PARALLEL quantize gate_up_proj
                # ================================================
                weight_quant_params_copy = copy.deepcopy(weight_quant_params)
                if mix_w_bits and mix_w_bits_dict:
                    weight_quant_params_copy['w_bits'] = mix_w_bits_dict.get('up_and_gate', weight_quant_params['w_bits'])
                if weight_quant_params_copy.get('w_groupsize', -1) != -1:
                    weight_quant_params_copy['perchannel'] = False

                bits = weight_quant_params_copy['w_bits']
                if bits < 16 or expert_bit_map:
                    logging.info(f"  [Parallel] GPTQ quantizing mlp.experts.gate_up_proj (base={bits} bits) [{num_experts} experts] across {num_gpus} GPUs")

                    tasks = []
                    for expert_idx in range(num_experts):
                        if expert_idx in skipped_experts_3d:
                            logging.info(f"    Expert {expert_idx}: skipping (rare expert, bits=16)")
                            dummy_quantizer = WeightQuantizer()
                            dummy_quantizer.configure({'w_bits': 16})
                            quantizers[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_up_proj'] = dummy_quantizer
                            continue
                        expert_wqp = _expert_wqp(weight_quant_params_copy, expert_bit_map, layer_idx, expert_idx)
                        if expert_wqp["w_bits"] >= 16:
                            logging.info(f"    Expert {expert_idx}: w_bits>=16, skip gate_up_proj quantization")
                            continue

                        W_expert = gate_up_proj.data[expert_idx].t().contiguous()
                        all_expert_inp = expert_inputs.get(expert_idx)

                        if all_expert_inp is not None:
                            num_tokens = all_expert_inp.shape[0]
                            if expert_idx not in padded_experts_3d:
                                logging.info(f"    Expert {expert_idx}: {num_tokens} tokens")
                        else:
                            logging.warning(f"    Expert {expert_idx}: no inputs collected, using random")
                            all_expert_inp = torch.randn(128, hidden_dim, device=dev, dtype=W_expert.dtype)

                        tasks.append({
                            'task_id': expert_idx,
                            'weight': W_expert,
                            'activations': all_expert_inp,
                            'quantizer_config': expert_wqp,
                            'in_features': hidden_dim,
                            'percdamp': weight_quant_params['percdamp'],
                            'groupsize': weight_quant_params['w_groupsize'],
                            'actorder': weight_quant_params.get('act_order', False),
                        })

                    results = parallel_quantize_experts(tasks, num_gpus)

                    for expert_idx, (Q_cpu, quantizer) in results.items():
                        gate_up_proj.data[expert_idx] = Q_cpu.t().contiguous().to(gate_up_proj.device)
                        quantizers[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_up_proj'] = quantizer
                    del results, tasks
                    torch.cuda.empty_cache()

                # ================================================
                # Re-forward to collect down_proj inputs using quantized gate_up
                # ================================================
                expert_down_inputs_quantized = {i: [] for i in range(num_experts)}

                # Manually forward padded expert inputs
                for expert_idx in padded_experts_3d:
                    if expert_idx in skipped_experts_3d:
                        continue
                    if expert_inputs.get(expert_idx) is not None and expert_inputs[expert_idx].shape[0] > 0:
                        padded_inp = expert_inputs[expert_idx]
                        with torch.no_grad():
                            gate_up = padded_inp @ experts.gate_up_proj[expert_idx]
                            gate, up = gate_up.chunk(2, dim=-1)
                            gated_output = up * experts.act_fn(gate)
                            gated_output = quarot_moe_mid_pre_down(gated_output, experts)
                            expert_down_inputs_quantized[expert_idx].append(gated_output.detach().clone())

                def hooked_experts_forward_quantized(hidden_states, routing_weights, router_indices):
                    batch_size = hidden_states.shape[0]
                    hidden_states = hidden_states.reshape(-1, experts.hidden_size)
                    next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
                    with torch.no_grad():
                        expert_mask = torch.nn.functional.one_hot(router_indices.long(), num_classes=experts.num_experts)
                        expert_mask = expert_mask.permute(2, 1, 0)
                        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

                    for expert_idx_tensor in expert_hit[:]:
                        expert_idx_val = expert_idx_tensor[0].item()
                        if expert_idx_val in padded_experts_3d or expert_idx_val in skipped_experts_3d:
                            continue
                        with torch.no_grad():
                            _, token_idx = torch.where(expert_mask[expert_idx_val])
                        current_state = hidden_states[token_idx]
                        gate_up = current_state @ experts.gate_up_proj[expert_idx_val]
                        gate, up = gate_up.chunk(2, dim=-1)
                        gated_output = up * experts.act_fn(gate)
                        gated_output = quarot_moe_mid_pre_down(gated_output, experts)
                        if len(gated_output) > 0:
                            expert_down_inputs_quantized[expert_idx_val].append(gated_output.detach().clone())
                        out = gated_output @ experts.down_proj[expert_idx_val]
                        weighted_output = out * routing_weights[token_idx, expert_idx_val, None]
                        next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))

                    next_states = next_states.view(batch_size, -1, experts.hidden_size)
                    return next_states

                # ================================================
                # PARALLEL quantize down_proj
                # ================================================
                weight_quant_params_copy = copy.deepcopy(weight_quant_params)
                if mix_w_bits and mix_w_bits_dict:
                    weight_quant_params_copy['w_bits'] = mix_w_bits_dict.get('down', weight_quant_params['w_bits'])
                if weight_quant_params_copy.get('w_groupsize', -1) != -1:
                    weight_quant_params_copy['perchannel'] = False

                bits = weight_quant_params_copy['w_bits']
                if bits < 16 or expert_bit_map:
                    logging.info(f"  [Re-forward] Collecting down_proj inputs using quantized gate_up_proj")
                    logging.info(f"  [Parallel] GPTQ quantizing mlp.experts.down_proj ({bits} bits) [{num_experts} experts] across {num_gpus} GPUs")

                    experts.forward = hooked_experts_forward_quantized
                    for j in range(nsamples):
                        inp_j = inps[j].to(dev).unsqueeze(0)
                        attn_mask_j, pos_ids_j, pos_emb_j = prepare_layer_kwargs(cache, j, dev)
                        if pos_emb_j is not None:
                            layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j, position_embeddings=pos_emb_j)
                        else:
                            layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j)
                    experts.forward = original_experts_forward

                    tasks = []
                    for expert_idx in range(num_experts):
                        if expert_idx in skipped_experts_3d:
                            logging.info(f"    Expert {expert_idx}: skipping down_proj (rare expert, bits=16)")
                            dummy_quantizer = WeightQuantizer()
                            dummy_quantizer.configure({'w_bits': 16})
                            quantizers[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj'] = dummy_quantizer
                            continue
                        expert_wqp = _expert_wqp(weight_quant_params_copy, expert_bit_map, layer_idx, expert_idx)
                        if expert_wqp["w_bits"] >= 16:
                            logging.info(f"    Expert {expert_idx}: w_bits>=16, skip down_proj quantization")
                            continue

                        W_expert = down_proj.data[expert_idx].t().contiguous()
                        if expert_down_inputs_quantized[expert_idx]:
                            all_down_inputs = torch.cat(expert_down_inputs_quantized[expert_idx], dim=0)
                        else:
                            logging.warning(f"    Expert {expert_idx}: no down inputs, using random")
                            all_down_inputs = torch.randn(128, intermediate_dim, device=dev, dtype=W_expert.dtype)

                        logging.info(f"    Expert {expert_idx}: down_proj weight={W_expert.shape}, tokens={all_down_inputs.shape[0]}")

                        tasks.append({
                            'task_id': expert_idx,
                            'weight': W_expert,
                            'activations': all_down_inputs,
                            'quantizer_config': expert_wqp,
                            'in_features': intermediate_dim,
                            'percdamp': weight_quant_params['percdamp'],
                            'groupsize': weight_quant_params['w_groupsize'],
                            'actorder': weight_quant_params.get('act_order', False),
                        })

                    results = parallel_quantize_experts(tasks, num_gpus)

                    for expert_idx, (Q_cpu, quantizer) in results.items():
                        down_proj.data[expert_idx] = Q_cpu.t().contiguous().to(down_proj.device)
                        quantizers[f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj'] = quantizer
                    del results, tasks

                    # Final re-forward after all experts quantized
                    logging.info(f"    [Re-forward] Re-forwarding after quantizing all experts")
                    for j in range(nsamples):
                        inp_j = inps[j].to(dev).unsqueeze(0)
                        attn_mask_j, pos_ids_j, pos_emb_j = prepare_layer_kwargs(cache, j, dev)
                        if pos_emb_j is not None:
                            layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j, position_embeddings=pos_emb_j)
                        else:
                            layer(inp_j, attention_mask=attn_mask_j, position_ids=pos_ids_j)

                # Clean up 3D MoE intermediate tensors
                del expert_inputs, expert_down_inputs, expert_down_inputs_quantized, original_expert_inputs
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()

            else:
                logging.warning(f"  Unknown MoE structure, skipping expert quantization")
                # Collect output and clean up
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
                elapsed = time.time() - layer_t0
                logging.info(f"  [Unknown MoE] Layer {layer_idx} done in {elapsed:.1f}s")
                continue

        # ============================================================
        # Collect quantized output for next layer
        # ============================================================
        logging.info(f"  Collecting quantized outputs for layer {layer_idx} -> layer {layer_idx + 1}")
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
        elapsed = time.time() - layer_t0
        logging.info(f"  Layer {layer_idx} done in {elapsed:.1f}s")

    # Restore use_cache
    if hasattr(model.config, 'use_cache') and use_cache is not None:
        model.config.use_cache = use_cache
    cleanup_memory(verbos=True)
    logging.info('-----GPTQ Quantization (Fast / Multi-GPU) Done-----\n')
    return quantizers


# ============================================================================
# Unified quantization interface (fast version)
# ============================================================================

def quantize_model_fast(
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
    quantize_shared_experts: bool = False,
    quantize_vision: bool = False,
    quantize_vision_projector: bool = False,
    vision_bits: int = 4,
    expert_bit_map: Optional[Dict[int, Dict[int, int]]] = None,
) -> Dict[str, WeightQuantizer]:
    """
    Unified interface for model quantization (fast version).

    Uses multi-GPU parallel expert quantization for GPTQ.
    Falls back to standard rtn_fwrd for RTN (already fast).

    Args:
        (same as quantize_model in gptq_utils.py)

    Returns:
        Dictionary of quantizers for each layer.
    """
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
        return gptq_fwrd_fast(
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
            quantize_shared_experts=quantize_shared_experts,
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
            quantize_shared_experts=quantize_shared_experts,
            expert_bit_map=expert_bit_map,
        )
    else:
        raise ValueError(f"Unknown quantization method: {method}. Supported: 'gptq', 'rtn'")
