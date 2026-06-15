"""QuaRot-style rotation helpers.

Heavy QuaRot modules depend on optional Hadamard kernels. Keep those imports
lazy so ordinary GPTQ / sensitivity utilities can import the package without
requiring the rotation extension.
"""

from .hadamard_ext import load_had44_from_txt, verify_had44
from .quarot_gptq_compat import (
    load_quarot_aux,
    quarot_kimi_expert_mid_pre_down,
    quarot_moe_mid_pre_down,
    save_quarot_aux,
)

__all__ = [
    "apply_quarot_rotation_qwen3_vl_moe",
    "apply_quarot_inference_hooks_qwen3_vl_moe",
    "apply_quarot_rotation_kimi_vl",
    "apply_quarot_inference_hooks_kimi_vl",
    "verify_had44",
    "load_had44_from_txt",
    "quarot_moe_mid_pre_down",
    "quarot_kimi_expert_mid_pre_down",
    "save_quarot_aux",
    "load_quarot_aux",
]


def __getattr__(name):
    if name in {
        "apply_quarot_rotation_qwen3_vl_moe",
        "apply_quarot_inference_hooks_qwen3_vl_moe",
    }:
        from .qwen3_vl_moe_quarot import (
            apply_quarot_inference_hooks_qwen3_vl_moe,
            apply_quarot_rotation_qwen3_vl_moe,
        )

        return {
            "apply_quarot_rotation_qwen3_vl_moe": apply_quarot_rotation_qwen3_vl_moe,
            "apply_quarot_inference_hooks_qwen3_vl_moe": apply_quarot_inference_hooks_qwen3_vl_moe,
        }[name]
    if name in {
        "apply_quarot_rotation_kimi_vl",
        "apply_quarot_inference_hooks_kimi_vl",
    }:
        from .kimi_vl_quarot import (
            apply_quarot_inference_hooks_kimi_vl,
            apply_quarot_rotation_kimi_vl,
        )

        return {
            "apply_quarot_rotation_kimi_vl": apply_quarot_rotation_kimi_vl,
            "apply_quarot_inference_hooks_kimi_vl": apply_quarot_inference_hooks_kimi_vl,
        }[name]
    raise AttributeError(name)
