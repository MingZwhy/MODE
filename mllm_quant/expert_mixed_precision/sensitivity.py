"""Modality-wise expert quantization sensitivity collection.

This module intentionally keeps sensitivity collection outside the GPTQ path.
It runs controlled forward passes where only one expert is fake-quantized and
only tokens from one modality use the quantized weights.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from mllm_quant.calibration.multimodal_calib import load_image, process_calibration_item
from mllm_quant.moe_freq.record_freq import ExpertFrequencyRecorder, _extract_routing, load_model_and_processor
from mllm_quant.quantization.quant_utils import WeightQuantizer
from mllm_quant.rotation.quarot_gptq_compat import quarot_kimi_expert_mid_pre_down, quarot_moe_mid_pre_down

logger = logging.getLogger(__name__)


def load_coco_calibration(
    calib_data: str,
    calib_img: str,
    processor: Any,
    tokenizer: Any,
    *,
    model_type: str,
    n_samples: int,
    max_seq_length: int = 4096,
    human_only: bool = True,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load COCO calibration samples using the same processing path as record_freq."""
    import numpy as np

    with open(calib_data, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(dataset)

    samples = []
    for i in tqdm(range(n_samples), desc="Processing calibration data"):
        item = dataset[i % len(dataset)]
        if human_only:
            item = dict(item)
            item["conversations"] = [
                c for c in item.get("conversations", []) if c.get("from") == "human"
            ]
        images = None
        img_path = item.get("image")
        if img_path:
            full_path = os.path.join(calib_img, img_path)
            if os.path.exists(full_path):
                images = [load_image(full_path)]
        try:
            sample = process_calibration_item(
                images=images,
                data_item=item,
                processor=processor,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                model_type=model_type,
            )
            if sample is not None:
                samples.append(sample)
        except Exception as exc:
            logger.warning("Skip calibration sample %d: %s", i, exc)
    return samples


def _clone_wqp(base: Mapping[str, Any], bit: int, group_size: int) -> Dict[str, Any]:
    wqp = dict(base)
    wqp["w_bits"] = int(bit)
    wqp["w_groupsize"] = int(group_size)
    if int(group_size) != -1:
        wqp["perchannel"] = False
    return wqp


def _fake_quant_weight(weight: torch.Tensor, bit: int, group_size: int, base_wqp: Mapping[str, Any]) -> torch.Tensor:
    quantizer = WeightQuantizer()
    quantizer.configure(_clone_wqp(base_wqp, bit, group_size))
    return quantizer.forward(weight)


def _select_positions(mask: Optional[torch.Tensor], token_idx: torch.Tensor, modality: str) -> torch.Tensor:
    if mask is None:
        return torch.zeros(token_idx.shape[0], dtype=torch.bool, device=token_idx.device)
    mask = mask.detach().to("cpu")
    token_idx_cpu = token_idx.detach().to("cpu")
    if token_idx_cpu.numel() > 0 and mask.shape[0] <= int(token_idx_cpu.max().item()):
        return torch.zeros(token_idx.shape[0], dtype=torch.bool, device=token_idx.device)
    selected = mask[token_idx_cpu].to(token_idx.device)
    if modality == "text":
        return ~selected
    return selected


def _current_modality_mask(recorder: ExpertFrequencyRecorder, modality: str) -> Optional[torch.Tensor]:
    image_mask = recorder.current_image_mask
    if image_mask is not None:
        image_mask = image_mask.detach().to("cpu")
    if modality == "text":
        return image_mask
    # Key vision tokens are the adaptive dominant image tokens.
    if recorder.current_dominant_mask is None or image_mask is None:
        return None
    dominant_mask = recorder.current_dominant_mask.detach().to("cpu")
    if dominant_mask.shape[0] != image_mask.shape[0]:
        return None
    return dominant_mask & image_mask


def _current_output_mask(recorder: ExpertFrequencyRecorder, modality: str) -> Optional[torch.Tensor]:
    route_mask = _current_modality_mask(recorder, modality)
    if route_mask is None:
        return None
    if modality == "text":
        return ~route_mask
    return route_mask


def _accumulate_mse(
    collector: Optional[Dict[str, float]],
    base: torch.Tensor,
    test: torch.Tensor,
) -> None:
    if collector is None or base.numel() == 0:
        return
    diff = (base.detach().float() - test.detach().float()).reshape(-1)
    collector["sum"] = float(collector.get("sum", 0.0)) + diff.pow(2).sum().item()
    collector["count"] = float(collector.get("count", 0.0)) + float(diff.numel())


@contextmanager
def patch_qwen3_3d_expert(
    moe_module: torch.nn.Module,
    recorder: ExpertFrequencyRecorder,
    *,
    expert_idx: int,
    bit: int,
    modality: str,
    group_size: int,
    base_wqp: Mapping[str, Any],
):
    """Patch Qwen3-style 3D-Parameter experts for modality-isolated sensitivity."""
    experts = moe_module.experts
    original_forward = experts.forward
    q_gate_up = _fake_quant_weight(experts.gate_up_proj.data[expert_idx].t().contiguous(), bit, group_size, base_wqp)
    q_gate_up = q_gate_up.t().contiguous().to(experts.gate_up_proj.device)
    q_down = _fake_quant_weight(experts.down_proj.data[expert_idx].t().contiguous(), bit, group_size, base_wqp)
    q_down = q_down.t().contiguous().to(experts.down_proj.device)

    def _forward(self, hidden_states, routing_weights, router_indices):
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(router_indices.long(), num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        modality_mask = _current_modality_mask(recorder, modality)
        for expert_idx_tensor in expert_hit[:]:
            cur_expert_idx = expert_idx_tensor[0].item()
            with torch.no_grad():
                _, token_idx = torch.where(expert_mask[cur_expert_idx])
            current_state = hidden_states[token_idx]

            gate_up = current_state @ self.gate_up_proj[cur_expert_idx]
            gate, up = gate_up.chunk(2, dim=-1)
            gated_output = up * self.act_fn(gate)
            gated_output = quarot_moe_mid_pre_down(gated_output, self)
            out = gated_output @ self.down_proj[cur_expert_idx]

            if cur_expert_idx == expert_idx and current_state.numel() > 0:
                selected = _select_positions(modality_mask, token_idx, modality)
                if selected.any():
                    q_gate_up_out = current_state[selected] @ q_gate_up
                    q_gate, q_up = q_gate_up_out.chunk(2, dim=-1)
                    q_gated = q_up * self.act_fn(q_gate)
                    q_gated = quarot_moe_mid_pre_down(q_gated, self)
                    out[selected] = q_gated @ q_down

            weighted_output = out * routing_weights[token_idx, cur_expert_idx, None]
            next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))

        return next_states.view(batch_size, -1, self.hidden_size)

    experts.forward = types.MethodType(_forward, experts)
    try:
        yield
    finally:
        experts.forward = original_forward


@contextmanager
def patch_modulelist_expert(
    moe_module: torch.nn.Module,
    recorder: ExpertFrequencyRecorder,
    *,
    expert_idx: int,
    bit: int,
    modality: str,
    group_size: int,
    base_wqp: Mapping[str, Any],
):
    """Patch Deepseek/Kimi-style ModuleList experts in eval mode.

    Expert parallelism (ep_size > 1) is intentionally unsupported here because
    sensitivity collection runs on a single loaded model replica.
    """
    if getattr(moe_module, "ep_size", 1) != 1:
        raise NotImplementedError("ModuleList sensitivity patch only supports ep_size=1")
    original_moe_infer = moe_module.moe_infer
    expert = moe_module.experts[expert_idx]
    if expert is None:
        raise ValueError(f"Expert {expert_idx} is None on this rank")

    q_gate = _fake_quant_weight(expert.gate_proj.weight.data, bit, group_size, base_wqp).to(expert.gate_proj.weight.device)
    q_up = _fake_quant_weight(expert.up_proj.weight.data, bit, group_size, base_wqp).to(expert.up_proj.weight.device)
    q_down = _fake_quant_weight(expert.down_proj.weight.data, bit, group_size, base_wqp).to(expert.down_proj.weight.device)

    def _q_expert_forward(tokens):
        gate = F.linear(tokens, q_gate, expert.gate_proj.bias)
        up = F.linear(tokens, q_up, expert.up_proj.bias)
        mid = expert.act_fn(gate) * up
        mid = quarot_kimi_expert_mid_pre_down(mid, expert)
        return F.linear(mid, q_down, expert.down_proj.bias)

    @torch.no_grad()
    def _moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert_np = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        modality_mask = _current_modality_mask(recorder, modality)
        for cur_expert_idx, num_tokens in enumerate(tokens_per_expert_np):
            end_idx = start_idx + int(num_tokens)
            if num_tokens == 0:
                continue
            cur_expert = self.experts[cur_expert_idx]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = cur_expert(tokens_for_this_expert)
            if cur_expert_idx == expert_idx:
                token_positions = idxs[start_idx:end_idx] // topk_ids.shape[1]
                selected = _select_positions(modality_mask, token_positions, modality)
                if selected.any():
                    expert_out[selected] = _q_expert_forward(tokens_for_this_expert[selected])
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        return (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(x.dtype)
        )

    moe_module.moe_infer = types.MethodType(_moe_infer, moe_module)
    try:
        yield
    finally:
        moe_module.moe_infer = original_moe_infer


def _prepare_inputs(sample: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
        for k, v in sample.items()
    }


def _logits_for_kl(outputs: Any, mode: str) -> torch.Tensor:
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    logits = logits.float()
    if mode == "last":
        return logits[:, -1:, :]
    if mode == "all":
        return logits
    raise ValueError(f"Unknown kl_positions: {mode}")


def _kl_from_logits(base_logits: torch.Tensor, test_logits: torch.Tensor) -> float:
    base_logp = F.log_softmax(base_logits, dim=-1)
    test_logp = F.log_softmax(test_logits, dim=-1)
    return F.kl_div(test_logp, base_logp.exp(), reduction="batchmean", log_target=False).item()


class _StopAfterMoe(Exception):
    pass


def _moe_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _capture_moe_output(
    model: torch.nn.Module,
    sample: Mapping[str, Any],
    moe_module: torch.nn.Module,
    recorder: ExpertFrequencyRecorder,
) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
    """Run one sample until ``moe_module`` finishes and capture its output."""
    captured: Dict[str, Any] = {}

    def _hook(_module, _inp, output):
        out = _moe_tensor(output)
        captured["output"] = out.detach().float().cpu()
        captured["text_mask"] = _current_output_mask(recorder, "text")
        captured["vision_mask"] = _current_output_mask(recorder, "vision")
        raise _StopAfterMoe

    handle = moe_module.register_forward_hook(_hook)
    try:
        try:
            with torch.no_grad():
                model(**_prepare_inputs(sample, next(model.parameters()).device))
        except _StopAfterMoe:
            pass
    finally:
        handle.remove()

    if "output" not in captured:
        raise RuntimeError("Target MoE module did not run; cannot collect moe_mse sensitivity")

    return captured["output"], {
        "text": captured.get("text_mask"),
        "vision": captured.get("vision_mask"),
    }


def _copy_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    return mask.detach().cpu().bool().clone()


def _capture_all_moe_caches(
    model: torch.nn.Module,
    sample: Mapping[str, Any],
    moe_layers: List[Tuple[str, torch.nn.Module]],
    recorder: ExpertFrequencyRecorder,
) -> Dict[str, Dict[str, Any]]:
    """Run one full-precision sample once and cache every target MoE layer."""
    captured: Dict[str, Dict[str, Any]] = {}
    handles = []

    def _make_hook(layer_name: str, num_experts: int, top_k: int):
        def _hook(module, inp, output):
            hidden_states = inp[0]
            router_indices = _extract_routing(module, hidden_states, output, num_experts, top_k)
            if router_indices is None:
                raise RuntimeError(f"Failed to extract router indices for {layer_name}")
            captured[layer_name] = {
                "hidden_states": hidden_states.reshape(-1, hidden_states.shape[-1]).detach().cpu(),
                "router_indices": router_indices.reshape(-1, top_k).detach().cpu(),
                "masks": {
                    "text": _copy_mask(_current_output_mask(recorder, "text")),
                    "vision": _copy_mask(_current_output_mask(recorder, "vision")),
                },
            }
        return _hook

    for layer_name, moe_module in moe_layers:
        handles.append(
            moe_module.register_forward_hook(
                _make_hook(
                    layer_name,
                    recorder.num_experts_map[layer_name],
                    recorder.top_k_map[layer_name],
                )
            )
        )

    try:
        with torch.no_grad():
            model(**_prepare_inputs(sample, next(model.parameters()).device))
    finally:
        for handle in handles:
            handle.remove()

    missing = [layer_name for layer_name, _ in moe_layers if layer_name not in captured]
    if missing:
        raise RuntimeError(f"Failed to cache {len(missing)} MoE layers, first missing: {missing[0]}")
    return captured


def _cached_tokens_for_expert(
    caches: List[Dict[str, Any]],
    expert_idx: int,
    modality: str,
) -> Optional[torch.Tensor]:
    chunks = []
    for cache in caches:
        router_indices = cache["router_indices"]
        token_positions = (router_indices == int(expert_idx)).nonzero(as_tuple=False)
        if token_positions.numel() == 0:
            continue
        token_positions = token_positions[:, 0]
        mask = cache["masks"].get(modality)
        if mask is None:
            continue
        mask = mask.detach().cpu().bool().flatten()
        if mask.shape[0] != cache["hidden_states"].shape[0]:
            continue
        token_positions = token_positions[mask[token_positions]]
        if token_positions.numel() > 0:
            chunks.append(cache["hidden_states"].index_select(0, token_positions))
    if not chunks:
        return None
    return torch.cat(chunks, dim=0)


def _qwen3_expert_forward(
    experts: torch.nn.Module,
    expert_idx: int,
    tokens: torch.Tensor,
    gate_up_weight: Optional[torch.Tensor] = None,
    down_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    gate_up_weight = gate_up_weight if gate_up_weight is not None else experts.gate_up_proj[expert_idx]
    down_weight = down_weight if down_weight is not None else experts.down_proj[expert_idx]
    gate_up = tokens @ gate_up_weight
    gate, up = gate_up.chunk(2, dim=-1)
    mid = up * experts.act_fn(gate)
    mid = quarot_moe_mid_pre_down(mid, experts)
    return mid @ down_weight


def _modulelist_expert_forward(
    expert: torch.nn.Module,
    tokens: torch.Tensor,
    q_weights: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    if q_weights is None:
        return expert(tokens)
    q_gate, q_up, q_down = q_weights
    gate = F.linear(tokens, q_gate, expert.gate_proj.bias)
    up = F.linear(tokens, q_up, expert.up_proj.bias)
    mid = expert.act_fn(gate) * up
    mid = quarot_kimi_expert_mid_pre_down(mid, expert)
    return F.linear(mid, q_down, expert.down_proj.bias)


def _expert_output_mse_for_tokens(
    moe_module: torch.nn.Module,
    expert_idx: int,
    bit: int,
    tokens: torch.Tensor,
    *,
    group_size: int,
    base_wqp: Mapping[str, Any],
    chunk_size: int = 4096,
) -> float:
    collector = {"sum": 0.0, "count": 0.0}
    if hasattr(moe_module, "experts") and hasattr(moe_module.experts, "gate_up_proj"):
        experts = moe_module.experts
        device = experts.gate_up_proj.device
        dtype = experts.gate_up_proj.dtype
        q_gate_up = _fake_quant_weight(
            experts.gate_up_proj.data[expert_idx].t().contiguous(),
            bit,
            group_size,
            base_wqp,
        ).t().contiguous().to(device)
        q_down = _fake_quant_weight(
            experts.down_proj.data[expert_idx].t().contiguous(),
            bit,
            group_size,
            base_wqp,
        ).t().contiguous().to(device)
        with torch.no_grad():
            for start in range(0, tokens.shape[0], chunk_size):
                cur = tokens[start : start + chunk_size].to(device=device, dtype=dtype)
                base = _qwen3_expert_forward(experts, expert_idx, cur)
                test = _qwen3_expert_forward(experts, expert_idx, cur, q_gate_up, q_down)
                _accumulate_mse(collector, base, test)
    elif hasattr(moe_module, "experts") and isinstance(moe_module.experts, torch.nn.ModuleList):
        if getattr(moe_module, "ep_size", 1) != 1:
            raise NotImplementedError("Cached expert_mse only supports ModuleList MoE with ep_size=1")
        expert = moe_module.experts[expert_idx]
        if expert is None:
            raise ValueError(f"Expert {expert_idx} is None on this rank")
        device = expert.gate_proj.weight.device
        dtype = expert.gate_proj.weight.dtype
        q_weights = (
            _fake_quant_weight(expert.gate_proj.weight.data, bit, group_size, base_wqp).to(device),
            _fake_quant_weight(expert.up_proj.weight.data, bit, group_size, base_wqp).to(device),
            _fake_quant_weight(expert.down_proj.weight.data, bit, group_size, base_wqp).to(device),
        )
        with torch.no_grad():
            for start in range(0, tokens.shape[0], chunk_size):
                cur = tokens[start : start + chunk_size].to(device=device, dtype=dtype)
                base = _modulelist_expert_forward(expert, cur)
                test = _modulelist_expert_forward(expert, cur, q_weights)
                _accumulate_mse(collector, base, test)
    else:
        raise NotImplementedError(f"Unsupported MoE structure for cached expert_mse: {type(moe_module).__name__}")

    count = float(collector.get("count", 0.0))
    return float(collector.get("sum", 0.0)) / count if count > 0 else 0.0


def _collect_cached_expert_mse_for_layer(
    moe_module: torch.nn.Module,
    caches: List[Dict[str, Any]],
    *,
    layer_name: str,
    num_experts: int,
    bits: Tuple[int, ...],
    group_size: int,
    base_wqp: Mapping[str, Any],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    layer_sens: Dict[str, Dict[str, Dict[str, float]]] = {}
    layer_short = layer_name
    if ".layers." in layer_name:
        layer_short = "L" + layer_name.split(".layers.", 1)[1].split(".", 1)[0]
    progress = tqdm(
        range(num_experts),
        desc=f"Cached expert_mse {layer_short} experts",
        leave=True,
    )
    progress.set_postfix_str(layer_name)
    for expert_idx in progress:
        layer_sens[str(expert_idx)] = {str(bit): {} for bit in bits}
        modality_tokens = {
            modality: _cached_tokens_for_expert(caches, expert_idx, modality)
            for modality in ("text", "vision")
        }
        for bit in bits:
            for modality, tokens in modality_tokens.items():
                if tokens is None or tokens.numel() == 0:
                    loss = 0.0
                else:
                    loss = _expert_output_mse_for_tokens(
                        moe_module,
                        expert_idx,
                        int(bit),
                        tokens,
                        group_size=group_size,
                        base_wqp=base_wqp,
                    )
                layer_sens[str(expert_idx)][str(bit)][modality] = loss
    return layer_sens


def _mse_for_modality(base: torch.Tensor, test: torch.Tensor, mask: Optional[torch.Tensor]) -> float:
    if base.shape != test.shape:
        raise ValueError(f"MoE output shape mismatch: baseline={tuple(base.shape)} test={tuple(test.shape)}")

    diff = (base - test).float()
    if diff.dim() == 3 and diff.shape[0] == 1:
        diff_tokens = diff[0]
    else:
        diff_tokens = diff.reshape(-1, diff.shape[-1])

    if mask is not None:
        mask = mask.detach().to("cpu").bool().flatten()
        if mask.shape[0] == diff_tokens.shape[0] and mask.any():
            diff_tokens = diff_tokens[mask]

    return diff_tokens.pow(2).mean().item()


def _get_moe_layers(recorder: ExpertFrequencyRecorder, model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    modules = dict(model.named_modules())
    return [(name, modules[name]) for name in recorder.layer_names if name in modules]


def collect_sensitivity(
    *,
    model_path: str,
    model_type: str,
    calib_data: str,
    calib_img: str,
    output_path: str,
    n_samples: int = 128,
    candidate_bits: Iterable[int] = (2, 3, 4),
    group_size: int = 128,
    max_seq_length: int = 4096,
    dominant_ratio: float = 0.2,
    human_only: bool = True,
    seed: int = 42,
    metric: str = "logit_kl",
    kl_positions: str = "last",
    layer_limit: Optional[int] = None,
    expert_limit: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect modality-wise sensitivity on COCO calibration data."""
    if metric not in {"logit_kl", "moe_mse", "expert_mse"}:
        raise ValueError(f"Unknown sensitivity metric: {metric}")

    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, processor, tokenizer = load_model_and_processor(
        model_path,
        model_type,
        use_eager_attn=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    samples = load_coco_calibration(
        calib_data,
        calib_img,
        processor,
        tokenizer,
        model_type=model_type,
        n_samples=n_samples,
        max_seq_length=max_seq_length,
        human_only=human_only,
        seed=seed,
    )
    if not samples:
        raise ValueError("No calibration samples loaded")

    recorder = ExpertFrequencyRecorder(
        model_type,
        detail=True,
        dominant_ratio=dominant_ratio,
        attn_mode="adaptive",
        use_text_rater=False,
    )
    recorder.setup(model, tokenizer=tokenizer)
    moe_layers = _get_moe_layers(recorder, model)
    if layer_limit is not None:
        moe_layers = moe_layers[:layer_limit]
    if not moe_layers:
        raise ValueError("No MoE layers found")

    base_wqp = {
        "w_bits": 4,
        "w_groupsize": group_size,
        "w_asym": False,
        "w_clip": True,
        "perchannel": group_size == -1,
        "norm": 2.4,
        "grid": 100,
        "maxshrink": 0.8,
    }

    baseline_logits = []
    model_device = next(model.parameters()).device
    if model_device.type == "cpu" and device_obj.type != "cpu":
        logger.info("Model uses device_map; keeping original parameter device %s", model_device)
    if metric == "logit_kl":
        # Compute baseline logits with the same hooks active, so adaptive masks follow the same path.
        logger.info("Collecting baseline logits on %d samples", len(samples))
        for sample in tqdm(samples, desc="Baseline"):
            with torch.no_grad():
                outputs = model(**_prepare_inputs(sample, next(model.parameters()).device))
                baseline_logits.append(_logits_for_kl(outputs, kl_positions).cpu())

    sensitivities: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    bits = tuple(int(b) for b in candidate_bits)
    modalities = ("text", "vision")
    all_expert_mse_caches: Optional[Dict[str, List[Dict[str, Any]]]] = None
    if metric == "expert_mse":
        logger.info(
            "Caching full-precision MoE inputs for %d layers on %d samples",
            len(moe_layers),
            len(samples),
        )
        all_expert_mse_caches = {layer_name: [] for layer_name, _ in moe_layers}
        for sample in tqdm(samples, desc="Cache all MoE layers"):
            sample_caches = _capture_all_moe_caches(model, sample, moe_layers, recorder)
            for layer_name in all_expert_mse_caches:
                all_expert_mse_caches[layer_name].append(sample_caches[layer_name])

    for layer_name, moe_module in moe_layers:
        num_experts = recorder.num_experts_map[layer_name]
        if expert_limit is not None:
            num_experts = min(num_experts, expert_limit)
        sensitivities[layer_name] = {}

        if metric == "expert_mse":
            assert all_expert_mse_caches is not None
            caches = all_expert_mse_caches[layer_name]
            sensitivities[layer_name] = _collect_cached_expert_mse_for_layer(
                moe_module,
                caches,
                layer_name=layer_name,
                num_experts=num_experts,
                bits=bits,
                group_size=group_size,
                base_wqp=base_wqp,
            )
            del caches
            continue

        baseline_moe_outputs: List[Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]] = []
        if metric == "moe_mse":
            logger.info("Collecting baseline MoE outputs for %s on %d samples", layer_name, len(samples))
            for sample in tqdm(samples, desc=f"Baseline {layer_name}", leave=False):
                baseline_moe_outputs.append(_capture_moe_output(model, sample, moe_module, recorder))

        for expert_idx in range(num_experts):
            sensitivities[layer_name][str(expert_idx)] = {}
            for bit in bits:
                sensitivities[layer_name][str(expert_idx)][str(bit)] = {}
                for modality in modalities:
                    if hasattr(moe_module, "experts") and hasattr(moe_module.experts, "gate_up_proj"):
                        patcher = patch_qwen3_3d_expert
                    elif hasattr(moe_module, "experts") and isinstance(moe_module.experts, torch.nn.ModuleList):
                        patcher = patch_modulelist_expert
                    else:
                        raise NotImplementedError(f"Unsupported MoE structure for {layer_name}")

                    total_loss = 0.0
                    with patcher(
                        moe_module,
                        recorder,
                        expert_idx=expert_idx,
                        bit=bit,
                        modality=modality,
                        group_size=group_size,
                        base_wqp=base_wqp,
                    ):
                        for sample_idx, sample in enumerate(tqdm(samples, desc=f"{layer_name} e{expert_idx} {bit}b {modality}", leave=False)):
                            if metric == "logit_kl":
                                with torch.no_grad():
                                    outputs = model(**_prepare_inputs(sample, next(model.parameters()).device))
                                    test_logits = _logits_for_kl(outputs, kl_positions).cpu()
                                total_loss += _kl_from_logits(baseline_logits[sample_idx], test_logits)
                            else:
                                test_out, _ = _capture_moe_output(model, sample, moe_module, recorder)
                                base_out, base_masks = baseline_moe_outputs[sample_idx]
                                total_loss += _mse_for_modality(base_out, test_out, base_masks.get(modality))
                    loss = total_loss / len(samples)
                    sensitivities[layer_name][str(expert_idx)][str(bit)][modality] = loss

    payload = {
        "metadata": {
            "model_path": model_path,
            "model_type": model_type,
            "calib_data": calib_data,
            "calib_img": calib_img,
            "n_samples": len(samples),
            "candidate_bits": list(bits),
            "group_size": group_size,
            "dominant_ratio": dominant_ratio,
            "attn_mode": "adaptive",
            "human_only": human_only,
            "metric": metric,
            "kl_positions": kl_positions,
        },
        "sensitivities": sensitivities,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    recorder.remove_hooks()
    return payload
