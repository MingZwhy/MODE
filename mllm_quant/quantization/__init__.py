# Quantization module
# Supports: GPTQ, RTN quantization methods
# Fast variant: multi-GPU parallel expert quantization for MoE models

from .gptq_utils import (
    GPTQ,
    GPTQWeight,
    gptq_fwrd,
    rtn_fwrd,
    quantize_model,
    get_model_type,
    is_moe_layer,
    get_moe_expert_params,
    get_num_experts_from_layer,
    generate_sequential_for_layer,
    cleanup_memory,
    QWEN3_VL_MODEL,
    QWEN3_VL_MOE_MODEL,
    INTERNVL_MODEL,
)

from .gpt_utils_fast import (
    gptq_fwrd_fast,
    quantize_model_fast,
    parallel_quantize_experts,
)

from .quant_utils import (
    WeightQuantizer,
    find_qlayers,
    get_minq_maxq,
    asym_quant,
    asym_dequant,
    asym_quant_dequant,
    sym_quant,
    sym_dequant,
    sym_quant_dequant,
    pack_i4,
    unpack_i4,
)

from .quant_linear import QuantLinear

# eval_utils depends on lmms_eval which may not always be installed;
# lazy import to avoid breaking the rest of the package.
def evaluate_model(*args, **kwargs):
    from .eval_utils import evaluate_model as _evaluate_model
    return _evaluate_model(*args, **kwargs)

__all__ = [
    # GPTQ related
    "GPTQ",
    "GPTQWeight",
    "gptq_fwrd",
    "rtn_fwrd",
    "quantize_model",
    "get_model_type",
    "is_moe_layer",
    "get_moe_expert_params",
    "get_num_experts_from_layer",
    "generate_sequential_for_layer",
    "cleanup_memory",
    # Fast GPTQ (multi-GPU parallel expert quantization)
    "gptq_fwrd_fast",
    "quantize_model_fast",
    "parallel_quantize_experts",
    # Model type constants
    "QWEN3_VL_MODEL",
    "QWEN3_VL_MOE_MODEL", 
    "INTERNVL_MODEL",
    # Quantization utilities
    "WeightQuantizer",
    "find_qlayers",
    "get_minq_maxq",
    "asym_quant",
    "asym_dequant",
    "asym_quant_dequant",
    "sym_quant",
    "sym_dequant",
    "sym_quant_dequant",
    "pack_i4",
    "unpack_i4",
    # Quant linear
    "QuantLinear",
    # Eval utilities (lazy)
    "evaluate_model",
]
