"""
Align GPTQ / fast-GPTQ MoE calibration hooks with QuaRot ``experts.forward``.

When QuaRot is applied, expert ``down_proj`` expects activations after an online full Hadamard
on ``intermediate_dim``. The temporary ``hooked_experts_forward`` replacements in
``gptq_utils`` / ``gpt_utils_fast`` must apply the same transform; otherwise down_proj
Hessians / activations are wrong.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def quarot_moe_mid_pre_down(gated_output: torch.Tensor, experts: nn.Module) -> torch.Tensor:
    """
    Apply the same post-activation Hadamard as ``_patch_experts_forward`` before ``@ down_proj``.

    No-op unless ``experts._quarot_experts_patched`` is True (set after QuaRot).
    """
    if not getattr(experts, "_quarot_experts_patched", False):
        return gated_output
    # Lazy imports: hadamard_utils lives under 3rdparty/QuaRot/fake_quant (on sys.path from qwen3_vl_moe_quarot)
    from mllm_quant.rotation.qwen3_vl_moe_quarot import _apply_online_full_had

    import hadamard_utils  # noqa: WPS433 (runtime path)

    fp32_had = getattr(experts, "_quarot_fp32_had", False)
    inter = experts.expert_dim
    had_k, k = hadamard_utils.get_hadK(inter)
    if had_k is not None:
        had_k = had_k.to(device=gated_output.device, dtype=gated_output.dtype)
    return _apply_online_full_had(gated_output, had_k, k, fp32_had)


def quarot_kimi_expert_mid_pre_down(gated_output: torch.Tensor, expert_mlp: nn.Module) -> torch.Tensor:
    """
    Same online full Hadamard as ``_patch_deepseek_mlp_online_had`` before ``down_proj``,
    for Kimi-VL ``DeepseekV3MLP`` experts inside ``DeepseekV3MoE`` (ModuleList path).

    Used by GPTQ calibration re-forward where activations are built manually (gate×up) without
    calling the patched ``expert.forward``.
    """
    if expert_mlp is None or not getattr(expert_mlp, "_quarot_mlp_had_patched", False):
        return gated_output
    from mllm_quant.rotation.hadamard_ext import get_hadk_kimi
    from mllm_quant.rotation.qwen3_vl_moe_quarot import _apply_online_full_had

    fp32_had = getattr(expert_mlp, "_quarot_fp32_had", False)
    inter = expert_mlp.down_proj.in_features
    had_k, k = get_hadk_kimi(inter)
    if had_k is not None:
        had_k = had_k.to(device=gated_output.device, dtype=gated_output.dtype)
    return _apply_online_full_had(gated_output, had_k, k, fp32_had)


def save_quarot_aux(
    pretrained_model_dir: str,
    Q_cpu: torch.Tensor,
    *,
    fp32_had: bool,
    rotate_mode: str,
    model_type: str = "qwen3_vl_moe",
    ffn_had: Optional[bool] = None,
) -> str:
    """
    Save QuaRot matrix and metadata **inside** the HuggingFace model directory
    (the same path passed to ``model.save_pretrained(pretrained_model_dir)``),
    alongside ``config.json`` and weight shards as ``quarot_aux.pt``.

    Kimi-VL saves ``model_type='kimi_vl'`` and ``ffn_had`` for inference hook reload.
    """
    path = os.path.join(pretrained_model_dir, "quarot_aux.pt")
    payload: Dict[str, Any] = {
        "Q": Q_cpu.detach().cpu(),
        "fp32_had": fp32_had,
        "rotate_mode": rotate_mode,
        "model_type": model_type,
    }
    if ffn_had is not None:
        payload["ffn_had"] = bool(ffn_had)
    torch.save(payload, path)
    return path


def load_quarot_aux(pretrained_model_dir: str) -> Optional[Dict[str, Any]]:
    """Load ``<pretrained_model_dir>/quarot_aux.pt`` if present; else None."""
    path = os.path.join(pretrained_model_dir, "quarot_aux.pt")
    if not os.path.isfile(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
