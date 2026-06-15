"""
mllm_quant - Multimodal Large Language Model Quantization

Submodules are imported lazily to avoid heavy dependency loading at startup.
Usage:
    from mllm_quant.quantization import quantize_model, GPTQ
    from mllm_quant.calibration import get_multimodal_calib_dataset
    from mllm_quant.models import get_model_class
"""

__version__ = "0.1.0"
