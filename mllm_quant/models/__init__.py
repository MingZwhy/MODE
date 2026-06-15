# Models module
# Supports: Qwen3-VL, Qwen3-VL-MoE, InternVL, Kimi-VL
#
# Models are imported lazily to avoid loading all heavy dependencies at startup.
# Use get_model_class() / get_config_class() or import from submodules directly:
#   from mllm_quant.models.qwen3_vl import Qwen3VLForConditionalGeneration

AVAILABLE_MODEL_TYPES = ["qwen3_vl", "qwen3_vl_moe", "internvl", "kimi_vl"]


def get_model_class(model_type: str):
    """Get model class by model type (lazy import)."""
    if model_type == "qwen3_vl":
        from .qwen3_vl import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    elif model_type == "qwen3_vl_moe":
        from .qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
        return Qwen3VLMoeForConditionalGeneration
    elif model_type == "internvl":
        from .internvl import InternVLForConditionalGeneration
        return InternVLForConditionalGeneration
    elif model_type == "kimi_vl":
        from .kimi_vl import KimiVLForConditionalGeneration
        return KimiVLForConditionalGeneration
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available: {AVAILABLE_MODEL_TYPES}")


def get_config_class(model_type: str):
    """Get config class by model type (lazy import)."""
    if model_type == "qwen3_vl":
        from .qwen3_vl import Qwen3VLConfig
        return Qwen3VLConfig
    elif model_type == "qwen3_vl_moe":
        from .qwen3_vl_moe import Qwen3VLMoeConfig
        return Qwen3VLMoeConfig
    elif model_type == "internvl":
        from .internvl import InternVLConfig
        return InternVLConfig
    elif model_type == "kimi_vl":
        from .kimi_vl import KimiVLConfig
        return KimiVLConfig
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available: {AVAILABLE_MODEL_TYPES}")
