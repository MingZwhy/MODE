#!/usr/bin/env python3
"""
Script to quantize multimodal models using GPTQ or RTN.

Usage:
    # GPTQ quantization (Qwen3-VL-MoE)
    python quantize.py \
        --model_path Qwen/Qwen3-VL-30B-A3B-Instruct \
        --model_type qwen3_vl_moe \
        --calib_data_path /path/to/calibration.json \
        --calib_image_folder /path/to/images \
        --output_path ./quantized_model \
        --method gptq \
        --bits 4 \
        --group_size 128
    
    # RTN quantization (no calibration data needed)
    python quantize.py \
        --model_path Qwen/Qwen3-VL-30B-A3B-Instruct \
        --model_type qwen3_vl_moe \
        --output_path ./quantized_model \
        --method rtn \
        --bits 4
    
    # GPTQ quantization (InternVL - use HF format for save/load compatibility)
    python quantize.py \
        --model_path /path/to/InternVL3_5-30B-A3B-HF \
        --model_type internvl \
        --calib_data_path /path/to/calibration.json \
        --calib_image_folder /path/to/images \
        --output_path ./quantized_model \
        --method gptq \
        --bits 4 \
        --group_size 128
    
    # GPTQ quantization (Kimi-VL)
    python quantize.py \
        --model_path /path/to/Kimi-VL-A3B-Instruct \
        --model_type kimi_vl \
        --calib_data_path /path/to/calibration.json \
        --calib_image_folder /path/to/images \
        --output_path ./quantized_model \
        --method gptq \
        --bits 4 \
        --group_size 128
    
    # Mixed-precision quantization
    python quantize.py \
        --model_path Qwen/Qwen3-VL-30B-A3B-Instruct \
        --model_type qwen3_vl_moe \
        --calib_data_path /path/to/calibration.json \
        --calib_image_folder /path/to/images \
        --output_path ./quantized_model \
        --method gptq \
        --mix_bits \
        --attn_bits 4 \
        --up_gate_bits 2 \
        --down_bits 3

    # QuaRot-style LLM rotation sanity check (Qwen3-VL-MoE; needs fast_hadamard_transform + GPU)
    python quantize.py \
        --model_path Qwen/Qwen3-VL-30B-A3B-Instruct \
        --model_type qwen3_vl_moe \
        --verify_quarot \
        --verify_image /path/to/coco/000000000001.jpg \
        --device_map auto

    # Same for Kimi-VL (uses Hadamard-44 in mllm_quant/mllm_quant/rotation/had44.txt)
    python quantize.py \
        --model_path /path/to/Kimi-VL-A3B-Instruct \
        --model_type kimi_vl \
        --verify_quarot \
        --verify_image /path/to/coco/000000000009.jpg \
        --device_map auto
"""

import os
import sys
import argparse
import logging
import shutil
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoProcessor, AutoTokenizer

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mllm_quant.quantization import quantize_model, quantize_model_fast, gptq_fwrd, rtn_fwrd, evaluate_model, cleanup_memory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def copy_kimi_vl_remote_code_files(save_dir: str) -> None:
    """
    Kimi-VL ``config.json`` uses ``auto_map`` pointing at ``configuration_kimi_vl`` /
    ``modeling_kimi_vl``. ``save_pretrained`` does not copy those modules, so
    ``AutoConfig.from_pretrained(..., trust_remote_code=True)`` fails on the saved folder.
    Copy the mllm_quant Kimi sources next to weights for standalone eval / HF tools.
    """
    repo_root = Path(__file__).resolve().parent.parent
    kimi_pkg = repo_root / "mllm_quant" / "models" / "kimi_vl"
    out = Path(save_dir)
    for name in ("configuration_kimi_vl.py", "modeling_kimi_vl.py"):
        src = kimi_pkg / name
        if not src.is_file():
            logging.warning("Kimi remote-code copy skipped (missing %s)", src)
            continue
        dst = out / name
        shutil.copy2(src, dst)
        logging.info("Copied %s -> %s", src, dst)


def load_model(model_path: str, model_type: str, device_map: str = "auto"):
    """
    Load model based on model type.
    
    Args:
        model_path: HuggingFace model ID or local path
        model_type: Model type (qwen3_vl, qwen3_vl_moe, internvl, kimi_vl)
        device_map: Device map for model loading
        
    Returns:
        Tuple of (model, processor, tokenizer)
    """
    logging.info(f"Loading model: {model_path}")
    
    if model_type == "qwen3_vl":
        from mllm_quant.models.qwen3_vl import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
    elif model_type == "qwen3_vl_moe":
        from mllm_quant.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
    elif model_type == "kimi_vl":
        from mllm_quant.models.kimi_vl import KimiVLForConditionalGeneration
        model = KimiVLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )
    elif model_type == "internvl":
        from mllm_quant.models.internvl import InternVLForConditionalGeneration
        from transformers import AutoConfig
        
        # Prefer HuggingFace/transformers format (InternVL3_5-30B-A3B-HF) - no config conversion,
        # preserves original config for seamless save/load of quantized model
        try:
            base_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            logging.warning(f"Could not load config, using standard loading: {e}")
            base_config = None
        if base_config is not None and hasattr(base_config, 'model_type') and base_config.model_type == 'internvl':
            # HF format: model_type='internvl', has text_config/vision_config directly
            logging.info("Detected HuggingFace/transformers format (model_type=internvl), loading directly without conversion")
            model = InternVLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True,
            )
        elif base_config is not None and hasattr(base_config, 'model_type') and base_config.model_type == 'internvl_chat':
            # Legacy internvl_chat format: require config conversion
            from mllm_quant.models.internvl import InternVLConfig
            from transformers.models.internvl.configuration_internvl import InternVLVisionConfig
            from transformers import Qwen2Config
            from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
            
            logging.info("Detected internvl_chat model type, converting to internvl config...")
            logging.warning("Consider using InternVL3_5-30B-A3B-HF (HF format) to avoid config conversion for better save/load compatibility")
            
            llm_config = base_config.llm_config.to_dict()
            vision_config = base_config.vision_config.to_dict()
            vision_config["use_absolute_position_embeddings"] = True
            
            llm_arch = llm_config.get('architectures', [None])[0] if isinstance(llm_config.get('architectures'), list) else None
            llm_model_type = llm_config.get('model_type', '')
            logging.info(f"LLM architecture: {llm_arch}, model_type: {llm_model_type}")
            
            if llm_model_type == 'qwen3_moe' or llm_arch == 'Qwen3MoeForCausalLM':
                language_config_class = Qwen3MoeConfig
                image_token_id = 151667
            elif llm_model_type == 'qwen2' or llm_arch == 'Qwen2ForCausalLM':
                language_config_class = Qwen2Config
                image_token_id = 151667
            else:
                language_config_class = Qwen2Config
                image_token_id = 151667
                logging.warning(f"Unknown LLM type {llm_model_type}/{llm_arch}, defaulting to Qwen2Config")
            
            UNNECESSARY_CONFIG_KEYS = [
                "_name_or_path", "_attn_implementation_autoset", "auto_map",
                "use_bfloat16", "use_flash_attn", "bias", "laux_allreduce",
                "moe_coeff_ratio", "moe_output_scale", "noisy_gate_policy",
                "shared_expert_intermediate_size", "use_residual", "use_moe",
                "use_rts", "use_weighted_residual", "moe_config", "num_routed_experts",
                "num_shared_experts", "capacity_factor", "eval_capacity_factor",
                "drop_path_rate", "architectures"
            ]
            llm_config = {k: v for k, v in llm_config.items() if k not in UNNECESSARY_CONFIG_KEYS}
            llm_config["use_cache"] = True
            if "InternVL3" in model_path or "InternVL3_5" in model_path:
                if llm_model_type == 'qwen3_moe' or llm_arch == 'Qwen3MoeForCausalLM':
                    llm_config["eos_token_id"] = 151645
            
            vision_config = {k: v for k, v in vision_config.items() if k not in UNNECESSARY_CONFIG_KEYS}
            if "attention_probs_dropout_prob" in vision_config:
                d = vision_config.pop("attention_probs_dropout_prob")
                vision_config["attention_dropout"] = vision_config["projection_dropout"] = d
            if "qk_normalization" in vision_config:
                vision_config["use_qk_norm"] = vision_config.pop("qk_normalization")
            if "qkv_bias" in vision_config:
                vision_config["attention_bias"] = vision_config.pop("qkv_bias")
            
            converted_config = InternVLConfig(
                text_config=language_config_class(**llm_config),
                vision_config=InternVLVisionConfig(**vision_config),
                image_token_id=image_token_id,
            )
            model = InternVLForConditionalGeneration.from_pretrained(
                model_path,
                config=converted_config,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True,
            )
        else:
            # Fallback: try standard loading
            logging.info("Using standard InternVL loading")
            model = InternVLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True,
            )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load processor and tokenizer
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    return model, processor, tokenizer


def _build_qwen3_vl_moe_verify_inputs(processor, image_path: str, user_prompt: str) -> dict:
    """One multimodal sample (image + text) for QuaRot equivalence check."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]
    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
    except Exception as e:
        logging.warning("qwen_vl_utils / processor path failed (%s); using apply_chat_template only.", e)
        batch = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    return dict(batch)


def _move_batch_to_model_device(batch: dict, model: torch.nn.Module) -> dict:
    dev = next(model.parameters()).device
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(dev)
        else:
            out[k] = v
    return out


def run_verify_quarot_qwen3_vl_moe(args) -> None:
    """
    Load Qwen3-VL-MoE twice (sequential): baseline vs QuaRot-rotated LLM; compare next-token argmax
    at the last position (same multimodal batch). Uses attn_implementation=eager for stability.
    """
    import gc
    from mllm_quant.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
    from mllm_quant.rotation.qwen3_vl_moe_quarot import apply_quarot_rotation_qwen3_vl_moe

    if args.model_type != "qwen3_vl_moe":
        raise ValueError("run_verify_quarot_qwen3_vl_moe 仅支持 model_type=qwen3_vl_moe")
    if not args.verify_image or not os.path.isfile(args.verify_image):
        raise FileNotFoundError(f"--verify_image 无效或不存在: {args.verify_image}")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    inputs = _build_qwen3_vl_moe_verify_inputs(processor, args.verify_image, args.verify_prompt)

    def _load():
        return Qwen3VLMoeForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map=args.device_map,
            attn_implementation="eager",
            trust_remote_code=True,
        )

    logging.info("QuaRot verify: loading baseline model...")
    model0 = _load()
    model0.eval()
    with torch.inference_mode():
        b0 = _move_batch_to_model_device(inputs, model0)
        o0 = model0(**b0, logits_to_keep=1)
        la = o0.logits[:, -1, :].float().cpu()
        tok0 = la.argmax(dim=-1)

    tok0_list = tok0.tolist()
    logging.info("Baseline next-token id(s): %s", tok0_list)

    del model0
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logging.info("QuaRot verify: loading second copy + applying LLM rotation...")
    model1 = _load()
    model1.eval()
    fp32_had = bool(getattr(args, "verify_fp32_had", False))
    if getattr(args, "verify_no_fp32_had", False):
        fp32_had = False
    apply_quarot_rotation_qwen3_vl_moe(
        model1,
        rotate_mode=args.quarot_rotate_mode,
        fp32_had=fp32_had,
    )
    with torch.inference_mode():
        b1 = _move_batch_to_model_device(inputs, model1)
        o1 = model1(**b1, logits_to_keep=1)
        lb = o1.logits[:, -1, :].float().cpu()
        tok1 = lb.argmax(dim=-1)

    tok1_list = tok1.tolist()
    max_diff = (la - lb).abs().max().item()
    match = bool(torch.equal(tok0, tok1))
    strict = match and max_diff < args.verify_max_logit_diff

    logging.info("Rotated  next-token id(s): %s", tok1_list)
    logging.info("Argmax match: %s | max abs logit diff: %.6f", match, max_diff)
    logging.info(
        "Strict pass (argmax + diff < %.4f): %s",
        args.verify_max_logit_diff,
        strict,
    )
    if tok0_list and tok1_list:
        try:
            logging.info("Baseline decoded: %r", processor.tokenizer.decode(tok0_list))
            logging.info("Rotated  decoded: %r", processor.tokenizer.decode(tok1_list))
        except Exception as e:
            logging.debug("decode skip: %s", e)

    if not match:
        raise SystemExit("QuaRot verify FAILED: next-token argmax differs.")
    if not strict:
        logging.warning(
            "QuaRot verify: argmax OK but logit diff >= threshold (may still be acceptable in bf16)."
        )


def _build_kimi_vl_verify_inputs(processor, image_path: str, user_prompt: str) -> dict:
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    batch = processor(text=[text], images=[img], return_tensors="pt", padding=True)
    return dict(batch)


def run_verify_quarot_kimi_vl(args) -> None:
    """Same as Qwen verify: baseline vs rotated Kimi-VL; last-position next-token (multimodal).

    Both copies load with ``attn_implementation=eager``. ChartQA accelerate eval defaults to
    ``flash_attention_2`` unless you pass ``--eval_attn_implementation eager``.
    """
    import gc
    from mllm_quant.models.kimi_vl import KimiVLForConditionalGeneration
    from mllm_quant.rotation.kimi_vl_quarot import apply_quarot_rotation_kimi_vl

    if args.model_type != "kimi_vl":
        raise ValueError("run_verify_quarot_kimi_vl 仅支持 model_type=kimi_vl")
    if not args.verify_image or not os.path.isfile(args.verify_image):
        raise FileNotFoundError(f"--verify_image 无效或不存在: {args.verify_image}")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    inputs = _build_kimi_vl_verify_inputs(processor, args.verify_image, args.verify_prompt)

    def _load():
        return KimiVLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map=args.device_map,
            trust_remote_code=True,
            attn_implementation="eager",
        )

    logging.info("QuaRot verify (Kimi-VL): loading baseline model...")
    model0 = _load()
    model0.eval()
    with torch.inference_mode():
        b0 = _move_batch_to_model_device(inputs, model0)
        o0 = model0(**b0)
        la = o0.logits[:, -1, :].float().cpu()
        tok0 = la.argmax(dim=-1)

    tok0_list = tok0.tolist()
    logging.info("Baseline next-token id(s): %s", tok0_list)

    del model0
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logging.info("QuaRot verify (Kimi-VL): loading second copy + applying rotation...")
    model1 = _load()
    model1.eval()
    fp32_had = bool(getattr(args, "verify_fp32_had", False))
    if getattr(args, "verify_no_fp32_had", False):
        fp32_had = False
    apply_quarot_rotation_kimi_vl(
        model1,
        rotate_mode=args.quarot_rotate_mode,
        fp32_had=fp32_had,
        ffn_had=not getattr(args, "verify_quarot_no_ffn_had", False),
    )
    with torch.inference_mode():
        b1 = _move_batch_to_model_device(inputs, model1)
        o1 = model1(**b1)
        lb = o1.logits[:, -1, :].float().cpu()
        tok1 = lb.argmax(dim=-1)

    tok1_list = tok1.tolist()
    max_diff = (la - lb).abs().max().item()
    match = bool(torch.equal(tok0, tok1))
    strict = match and max_diff < args.verify_max_logit_diff

    logging.info("Rotated  next-token id(s): %s", tok1_list)
    logging.info("Argmax match: %s | max abs logit diff: %.6f", match, max_diff)
    logging.info(
        "Strict pass (argmax + diff < %.4f): %s",
        args.verify_max_logit_diff,
        strict,
    )
    if tok0_list and tok1_list:
        try:
            tok = processor.tokenizer
            logging.info("Baseline decoded: %r", tok.decode(tok0_list))
            logging.info("Rotated  decoded: %r", tok.decode(tok1_list))
        except Exception as e:
            logging.debug("decode skip: %s", e)

    if not match:
        raise SystemExit("QuaRot verify FAILED: next-token argmax differs.")
    if not strict:
        logging.warning(
            "QuaRot verify: argmax OK but logit diff >= threshold (may still be acceptable in bf16)."
        )


def main():
    parser = argparse.ArgumentParser(description="Quantize multimodal models using GPTQ or RTN")
    
    # Eval-only mode (skip quantization, directly evaluate)
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip quantization, only run evaluation on model_path"
    )
    parser.add_argument(
        "--verify_quarot",
        action="store_true",
        help="Qwen3-VL-MoE or Kimi-VL: load twice, apply QuaRot-style LLM rotation on second, compare last-token argmax (needs --verify_image)",
    )
    parser.add_argument(
        "--verify_image",
        type=str,
        default=None,
        help="Image path (e.g. COCO sample) for --verify_quarot",
    )
    parser.add_argument(
        "--verify_prompt",
        type=str,
        default="What is in the image? Answer in one short phrase.",
        help="User text for --verify_quarot",
    )
    parser.add_argument(
        "--quarot_rotate_mode",
        type=str,
        default="hadamard",
        choices=["hadamard", "random"],
        help="Orthogonal matrix for QuaRot hidden rotation",
    )
    parser.add_argument(
        "--verify_fp32_had",
        action="store_true",
        help="With --verify_quarot: run online Hadamard in fp32 (tighter match; more VRAM). Default is bf16.",
    )
    parser.add_argument(
        "--verify_no_fp32_had",
        action="store_true",
        help="With --verify_quarot: force bf16 online Had (redundant with new default; overrides --verify_fp32_had).",
    )
    parser.add_argument(
        "--verify_quarot_no_ffn_had",
        action="store_true",
        help="Kimi-VL --verify_quarot only: skip FFN Hadamard-44 (isolate global-Q + visual merge)",
    )
    parser.add_argument(
        "--verify_max_logit_diff",
        type=float,
        default=5.0,
        help="With --verify_quarot, also require max |Δlogit| < this for strict pass",
    )
    parser.add_argument(
        "--quarot_before_quant",
        action="store_true",
        help="Qwen3-VL-MoE / Kimi-VL: apply QuaRot before RTN/GPTQ; "
        "writes quarot_aux.pt next to model.save_pretrained (config / shards)",
    )
    parser.add_argument(
        "--quarot_fp32_had",
        action="store_true",
        help="With --quarot_before_quant: online Hadamard in fp32 (more VRAM). Default is bf16.",
    )
    parser.add_argument(
        "--no_quarot_fp32_had",
        action="store_true",
        help="With --quarot_before_quant: force bf16 online Had (redundant with new default; overrides --quarot_fp32_had).",
    )
    parser.add_argument(
        "--no_quarot_kimi_ffn_had",
        action="store_true",
        help="With --quarot_before_quant on kimi_vl: skip FFN Hadamard-44 (global Q + visual merge only)",
    )

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="HuggingFace model ID or local path (model to quantize or evaluate)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type for quantization: qwen3_vl, qwen3_vl_moe, internvl, kimi_vl (required when not eval_only)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for quantized model (required when not eval_only)"
    )
    
    # Quantization method
    parser.add_argument(
        "--method",
        type=str,
        default="gptq",
        choices=["gptq", "gptq_fast", "rtn"],
        help="Quantization method (gptq, gptq_fast, or rtn). gptq_fast uses multi-GPU parallel expert quantization."
    )
    
    # Calibration arguments (only for GPTQ)
    parser.add_argument(
        "--calib_data_path",
        type=str,
        default=None,
        help="Path to calibration data (JSON/JSONL). Required for GPTQ."
    )
    parser.add_argument(
        "--calib_image_folder",
        type=str,
        default=None,
        help="Root folder for calibration images. Required for GPTQ."
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=128,
        help="Number of calibration samples"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Maximum sequence length"
    )
    
    # Quantization arguments
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[1, 2, 3, 4, 8],
        help="Default quantization bits (used when mix_bits is False)"
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="Group size for quantization (-1 for per-channel)"
    )
    parser.add_argument(
        "--asym",
        action="store_true",
        help="Use asymmetric quantization (default: symmetric)"
    )
    parser.add_argument(
        "--no_clip",
        action="store_true",
        help="Disable MSE-based scale optimization"
    )
    parser.add_argument(
        "--damp_percent",
        type=float,
        default=0.01,
        help="Damping percentage for Hessian (GPTQ)"
    )
    parser.add_argument(
        "--act_order",
        action="store_true",
        help="Use descending activation order (GPTQ)"
    )
    parser.add_argument(
        "--quantize_vision",
        action="store_true",
        help="With GPTQ/GPTQ_FAST, also quantize the vision encoder transformer blocks "
             "(e.g. model.visual.blocks / vision_tower.encoder.blocks) before LLM quantization"
    )
    parser.add_argument(
        "--quantize_vision_projector",
        action="store_true",
        help="In addition to --quantize_vision, also quantize the vision projector / merger "
             "modules (e.g. model.visual.merger, model.visual.deepstack_merger_list, "
             "multi_modal_projector). Requires --quantize_vision."
    )
    parser.add_argument(
        "--vision_bits",
        type=int,
        default=4,
        choices=[2, 3, 4, 8, 16],
        help="Bit-width for all vision encoder Linear layers when --quantize_vision is set"
    )
    
    # Mixed-precision arguments
    parser.add_argument(
        "--mix_bits",
        action="store_true",
        help="Enable mixed-precision quantization"
    )
    parser.add_argument(
        "--attn_bits",
        type=int,
        default=4,
        help="Bits for attention layers (q, k, v, o) when mix_bits is enabled"
    )
    parser.add_argument(
        "--up_gate_bits",
        type=int,
        default=4,
        help="Bits for up and gate projections when mix_bits is enabled"
    )
    parser.add_argument(
        "--down_bits",
        type=int,
        default=4,
        help="Bits for down projections when mix_bits is enabled"
    )
    parser.add_argument(
        "--expert_bits_json",
        type=str,
        default=None,
        help="Expert-level bit assignment JSON from scripts/solve_expert_bits.py. "
             "Overrides routed expert bits per layer/expert."
    )
    parser.add_argument(
        "--shared_expert_bits",
        type=int,
        default=4,
        help="When --expert_bits_json is used, quantize shared_experts at this bit-width (default: 4)."
    )
    
    # Other arguments
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for model loading"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for quantization"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    # MoE expert token balancing arguments
    parser.add_argument(
        "--keep_min",
        action="store_true",
        help="Enable minimum token balancing for MoE experts (pad experts with fewer tokens)"
    )
    parser.add_argument(
        "--skip_rare_expert",
        action="store_true",
        help="Skip quantization for experts with fewer tokens than avg*percentage (set bits to 16)"
    )
    parser.add_argument(
        "--percentage",
        type=float,
        default=0.1,
        help="Minimum percentage of average tokens per expert (used when keep_min=True or skip_rare_expert=True)"
    )
    parser.add_argument(
        "--quantize_shared_experts",
        action="store_true",
        help="Also quantize MoE shared_experts (gate_proj/up_proj use up_gate_bits, down_proj uses down_bits)"
    )

    # Evaluation arguments (used when --eval is set)
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run lmms-eval evaluation on quantized model after saving"
    )
    parser.add_argument(
        "--eval_tasks",
        type=str,
        default="chartqa",
        help="Comma-separated evaluation tasks (e.g. chartqa,textvqa_val)"
    )
    parser.add_argument(
        "--eval_model_type",
        type=str,
        default="vllm",
        choices=["hf", "vllm", "accelerate"],
        help="Eval backend: 'vllm' (tensor-parallel, fast), "
             "'accelerate' (HF + accelerate data-parallel, flexible), "
             "or 'hf' (legacy single-process HF)"
    )
    parser.add_argument(
        "--eval_output_path",
        type=str,
        default=None,
        help="Output path for eval results. Default: quantization/results"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation (default: 64 for vllm, 1 for accelerate)"
    )
    parser.add_argument(
        "--eval_max_model_len",
        type=int,
        default=16384,
        help="vLLM max_model_len (default 16k) to reduce OOM"
    )
    parser.add_argument(
        "--eval_num_gpus",
        type=int,
        default=None,
        help="Number of GPUs for accelerate eval (default: all visible GPUs)"
    )
    parser.add_argument(
        "--eval_accelerate_model_parallel",
        action="store_true",
        help="accelerate 评测：单进程 + HuggingFace device_map=auto，把一份模型切到多张可见 GPU；"
             "默认多卡会为 data parallel 起多个进程、每卡一整模易 OOM。"
             "启用后固定 --num_processes 1，eval_num_gpus 仅作提示；用 CUDA_VISIBLE_DEVICES 限定参与 auto 切分的 GPU。",
    )
    parser.add_argument(
        "--eval_limit",
        type=int,
        default=None,
        help="Max samples per eval task (default: no limit, use full dataset)"
    )
    parser.add_argument(
        "--eval_apply_quarot",
        action="store_true",
        help="With --eval_only + accelerate: load model then apply in-place QuaRot (Qwen3-VL-MoE / Kimi-VL only; needs mllm_quant + lmms-eval patches)"
    )
    parser.add_argument(
        "--eval_attn_implementation",
        type=str,
        default=None,
        choices=["eager", "flash_attention_2", "sdpa"],
        help="accelerate + qwen/kimi only: attn_implementation for lmms-eval (default: flash_attention_2). "
             "Use eager to align with --verify_quarot (verify loads Kimi/Qwen MoE with eager).",
    )

    # Expert protection experiment arguments
    parser.add_argument(
        "--protect_experiment",
        action="store_true",
        help="Run expert protection experiment: progressively protect top-K experts per layer"
    )
    parser.add_argument(
        "--freq_json",
        type=str,
        default=None,
        help="Path to expert frequency JSON (required for --protect_experiment)"
    )
    parser.add_argument(
        "--protect_step",
        type=int,
        default=16,
        help="Number of experts to protect in each step (default: 16)"
    )
    parser.add_argument(
        "--protect_begin",
        type=int,
        default=0,
        help="Base count added to the first sweep: first run protects protect_begin+protect_step experts "
             "per layer, then +protect_step each step (default: 0)"
    )
    parser.add_argument(
        "--protect_bits",
        type=int,
        default=4,
        help="Bit-width for protected experts (default: 4)"
    )
    parser.add_argument(
        "--ranking_strategy",
        type=str,
        default=None,
        help="Ranking strategy for --protect_experiment. "
             "Comma-separated list from: total, text_image, text_dominant, text_redundant. "
             "If not set, defaults to all four strategies."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for text frequency in mixed ranking: p = alpha*p_text + (1-alpha)*p_vision (default: 0.5)"
    )
    
    args = parser.parse_args()

    # QuaRot rotation equivalence (Qwen3-VL-MoE LLM only)
    if args.verify_quarot:
        if args.model_type is None:
            parser.error("--verify_quarot 需要指定 --model_type（qwen3_vl_moe 或 kimi_vl）")
        if args.model_type not in ("qwen3_vl_moe", "kimi_vl"):
            parser.error("--verify_quarot 支持 --model_type qwen3_vl_moe 或 kimi_vl")
        if not args.verify_image:
            parser.error("--verify_quarot 需要 --verify_image")
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        if args.model_type == "qwen3_vl_moe":
            run_verify_quarot_qwen3_vl_moe(args)
        else:
            run_verify_quarot_kimi_vl(args)
        return

    # Eval-only mode: skip quantization, run evaluation only
    if args.eval_only:

        # 清理 sys.path 中的本地 transformers 路径，避免子进程导入冲突
        local_tf_path = os.environ.get("LOCAL_TRANSFORMERS_PATH")
        while local_tf_path and local_tf_path in sys.path:
            sys.path.remove(local_tf_path)

        logging.info("eval_only mode: skipping quantization, running evaluation...")
        if getattr(args, "eval_apply_quarot", False):
            if args.eval_model_type != "accelerate":
                parser.error("--eval_apply_quarot 仅支持 --eval_model_type accelerate")
            if args.model_type not in ("qwen3_vl_moe", "kimi_vl"):
                parser.error("--eval_apply_quarot 需要 --model_type 为 qwen3_vl_moe 或 kimi_vl")
        _eval_batch = args.eval_batch_size
        if _eval_batch is None:
            _eval_batch = 1 if args.eval_model_type == "accelerate" else 64
        evaluate_model(
            model_path=args.model_path,
            model_type=args.eval_model_type,
            tasks=args.eval_tasks,
            output_path=args.eval_output_path,
            batch_size=_eval_batch,
            max_model_len=args.eval_max_model_len,
            model_arch=args.model_type,
            num_gpus=args.eval_num_gpus,
            limit=args.eval_limit,
            apply_quarot=getattr(args, "eval_apply_quarot", False),
            attn_implementation=args.eval_attn_implementation,
            accelerate_model_parallel=getattr(args, "eval_accelerate_model_parallel", False),
        )
        return

    # ================================================================
    # Expert protection experiment mode
    # ================================================================
    if args.protect_experiment:
        if args.freq_json is None:
            parser.error("--freq_json is required for --protect_experiment")

        import json as _json
        import gc
        import copy as _copy

        valid_model_types = ["qwen3_vl", "qwen3_vl_moe", "internvl", "kimi_vl"]
        if args.model_type is None or args.model_type not in valid_model_types:
            parser.error(f"--model_type is required for --protect_experiment (choose from {valid_model_types})")

        with open(args.freq_json, 'r') as f:
            freq_data = _json.load(f)

        # Parse layer indices and num_experts from freq_data keys
        layer_indices = []
        for key in freq_data:
            parts = key.split('.')
            for i, p in enumerate(parts):
                if p == 'layers' and i + 1 < len(parts):
                    layer_indices.append(int(parts[i + 1]))
                    break
        layer_indices = sorted(set(layer_indices))
        first_key = list(freq_data.keys())[0]
        num_experts = len(freq_data[first_key].get('dominant_image', freq_data[first_key].get('total', [])))
        logging.info(f"Freq data: {len(layer_indices)} layers, {num_experts} experts per layer")

        def _parse_layer_idx(layer_key):
            parts = layer_key.split('.')
            for i, p in enumerate(parts):
                if p == 'layers' and i + 1 < len(parts):
                    return int(parts[i + 1])
            return None

        def build_ranking_single(freq_data, key_name):
            """Rank experts by a single frequency key (e.g. 'total')."""
            ranking = {}
            for layer_key, layer_data in freq_data.items():
                layer_idx = _parse_layer_idx(layer_key)
                if layer_idx is None:
                    continue
                freqs = layer_data.get(key_name, [])
                if not freqs:
                    continue
                indexed = list(enumerate(freqs))
                indexed.sort(key=lambda x: x[1], reverse=True)
                ranking[layer_idx] = [idx for idx, _ in indexed]
            return ranking

        def build_ranking_mixed(freq_data, text_key, vision_key, alpha):
            """Rank experts by alpha*normalize(text) + (1-alpha)*normalize(vision)."""
            ranking = {}
            for layer_key, layer_data in freq_data.items():
                layer_idx = _parse_layer_idx(layer_key)
                if layer_idx is None:
                    continue
                text_freqs = layer_data.get(text_key, [])
                vision_freqs = layer_data.get(vision_key, [])
                if not text_freqs or not vision_freqs:
                    continue
                t_sum = sum(text_freqs) or 1.0
                v_sum = sum(vision_freqs) or 1.0
                combined = []
                for idx in range(len(text_freqs)):
                    p_t = text_freqs[idx] / t_sum
                    p_v = vision_freqs[idx] / v_sum
                    combined.append((idx, alpha * p_t + (1 - alpha) * p_v))
                combined.sort(key=lambda x: x[1], reverse=True)
                ranking[layer_idx] = [idx for idx, _ in combined]
            return ranking

        alpha = args.alpha
        all_strategies = {
            'total':          lambda: build_ranking_single(freq_data, 'total'),
            'text_image':     lambda: build_ranking_mixed(freq_data, 'text', 'image', alpha),
            'text_dominant':  lambda: build_ranking_mixed(freq_data, 'text', 'dominant_image', alpha),
            'text_redundant': lambda: build_ranking_mixed(freq_data, 'text', 'redundant_image', alpha),
        }

        if args.ranking_strategy:
            strategy_names = [s.strip() for s in args.ranking_strategy.split(',')]
        else:
            strategy_names = list(all_strategies.keys())

        rankings = {}
        for sname in strategy_names:
            if sname not in all_strategies:
                parser.error(f"Unknown ranking strategy: {sname}. Choose from: {list(all_strategies.keys())}")
            rankings[sname] = all_strategies[sname]()
            logging.info(f"Built ranking for strategy '{sname}' (alpha={alpha})")

        step = args.protect_step
        first_topk = args.protect_begin + step
        topk_values = []
        k = first_topk
        while k <= num_experts:
            topk_values.append(k)
            k += step
        if not topk_values:
            topk_values = [num_experts]
        elif topk_values[-1] != num_experts:
            topk_values.append(num_experts)

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        weight_quant_params = {
            'w_bits': args.bits,
            'w_groupsize': args.group_size,
            'w_asym': args.asym,
            'w_clip': not args.no_clip,
            'perchannel': True,
            'percdamp': args.damp_percent,
            'act_order': args.act_order,
            'norm': 2.4,
            'grid': 100,
            'maxshrink': 0.8,
        }

        mix_w_bits = args.mix_bits
        mix_w_bits_dict = None
        if mix_w_bits:
            mix_w_bits_dict = {
                'attn': args.attn_bits,
                'up_and_gate': args.up_gate_bits,
                'down': args.down_bits,
            }

        results_dir = args.eval_output_path or './rtn_results'
        os.makedirs(results_dir, exist_ok=True)
        summary = {}

        local_tf_path = os.environ.get("LOCAL_TRANSFORMERS_PATH")

        for group_name, ranking in rankings.items():
            summary[group_name] = {}
            for topk in topk_values:
                logging.info(f"\n{'='*60}")
                logging.info(f"[{group_name}] protect_topk={topk}")
                logging.info(f"{'='*60}")

                # Build protect_map: layer_idx -> set of top-K expert indices
                protect_map = {}
                for layer_idx, ranked_experts in ranking.items():
                    protect_map[layer_idx] = set(ranked_experts[:topk])

                # Reload model from scratch
                logging.info("Loading fresh model...")
                model, processor, tokenizer = load_model(
                    args.model_path,
                    args.model_type,
                    args.device_map,
                )
                model.seqlen = args.max_seq_length

                # Quantize with protection
                quantizers = quantize_model(
                    model=model,
                    method="rtn",
                    dev=device,
                    weight_quant_params=weight_quant_params,
                    mix_w_bits=mix_w_bits,
                    mix_w_bits_dict=mix_w_bits_dict,
                    expert_protect_map=protect_map,
                    protect_bits=args.protect_bits,
                )

                logging.info(f"Quantization complete. Quantized {len(quantizers)} layers.")

                # Save model temporarily for eval
                _attn_b = mix_w_bits_dict['attn'] if mix_w_bits_dict else args.bits
                _expert_b = args.bits
                _protect_b = args.protect_bits
                _freq_tag = os.path.splitext(os.path.basename(args.freq_json))[0]
                _alpha_str = f'_a{args.alpha}' if group_name != 'total' else ''
                tmp_dir = os.path.join(results_dir, f'attn{_attn_b}b_expert{_expert_b}b_protect{_protect_b}b_{_freq_tag}_{group_name}{_alpha_str}_top{topk}')
                os.makedirs(tmp_dir, exist_ok=True)
                logging.info(f"Saving temp model to {tmp_dir}...")
                model.save_pretrained(tmp_dir)
                if args.model_type == "kimi_vl":
                    copy_kimi_vl_remote_code_files(tmp_dir)
                processor.save_pretrained(tmp_dir)
                tokenizer.save_pretrained(tmp_dir)

                # Free model memory before eval
                del model, processor, tokenizer, quantizers
                gc.collect()
                cleanup_memory(verbos=True)

                # Run eval in a subprocess so vLLM GPU memory is freed on exit
                import subprocess as _sp
                _eval_batch = args.eval_batch_size
                if _eval_batch is None:
                    _eval_batch = 1 if args.eval_model_type == "accelerate" else 64
                eval_cmd = [
                    sys.executable, os.path.abspath(__file__),
                    '--model_path', tmp_dir,
                    '--model_type', args.model_type,
                    '--eval_only',
                    '--eval_tasks', args.eval_tasks,
                    '--eval_batch_size', str(_eval_batch),
                    '--eval_model_type', args.eval_model_type,
                    '--eval_max_model_len', str(args.eval_max_model_len),
                    '--eval_output_path', results_dir,
                ]
                if args.eval_num_gpus is not None:
                    eval_cmd += ['--eval_num_gpus', str(args.eval_num_gpus)]
                if getattr(args, "eval_accelerate_model_parallel", False):
                    eval_cmd += ["--eval_accelerate_model_parallel"]
                if args.eval_limit is not None:
                    eval_cmd += ['--eval_limit', str(args.eval_limit)]
                if getattr(args, "eval_attn_implementation", None):
                    eval_cmd += [
                        "--eval_attn_implementation",
                        str(args.eval_attn_implementation),
                    ]

                logging.info(f"Evaluating {group_name} top{topk} in subprocess...")
                env = os.environ.copy()
                env['PYTHONPATH'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ret = _sp.run(eval_cmd, env=env)
                if ret.returncode != 0:
                    logging.warning(f"Eval subprocess exited with code {ret.returncode}")

                summary[group_name][topk] = f"eval done, results in {results_dir}"
                logging.info(f"[{group_name}] top{topk} eval complete.")

                # Clean up temp model
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

        # Write summary
        summary_path = os.path.join(results_dir, 'protect_experiment_summary.json')
        with open(summary_path, 'w') as f:
            _json.dump(summary, f, indent=2)
        logging.info(f"Experiment complete. Summary saved to {summary_path}")
        return
    
    # Validate arguments (for quantization mode)
    valid_model_types = ["qwen3_vl", "qwen3_vl_moe", "internvl", "kimi_vl"]
    if args.output_path is None:
        parser.error("--output_path is required when not using --eval_only")
    if args.model_type is None or args.model_type not in valid_model_types:
        parser.error(f"--model_type is required when not using --eval_only (choose from {valid_model_types})")
    if args.method in ("gptq", "gptq_fast") and (args.calib_data_path is None or args.calib_image_folder is None):
        parser.error("GPTQ requires --calib_data_path and --calib_image_folder")
    if args.quantize_vision and args.method not in ("gptq", "gptq_fast"):
        parser.error("--quantize_vision is only supported with --method gptq or gptq_fast")
    if args.quantize_vision and args.model_type not in ("qwen3_vl_moe", "kimi_vl"):
        parser.error("--quantize_vision currently supports --model_type qwen3_vl_moe or kimi_vl")
    if args.quantize_vision_projector and not args.quantize_vision:
        parser.error("--quantize_vision_projector requires --quantize_vision")
    if getattr(args, "quarot_before_quant", False):
        if args.model_type not in ("qwen3_vl_moe", "kimi_vl"):
            parser.error("--quarot_before_quant 需要 --model_type 为 qwen3_vl_moe 或 kimi_vl")

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load model
    model, processor, tokenizer = load_model(
        args.model_path,
        args.model_type,
        args.device_map,
    )
    
    # Set seqlen for model (used in gptq_fwrd)
    model.seqlen = args.max_seq_length
    
    logging.info(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

    quarot_Q_cpu = None
    quarot_aux_model_type: Optional[str] = None
    quarot_ffn_had_save: Optional[bool] = None
    quarot_fp32_had = bool(getattr(args, "quarot_fp32_had", False))
    if getattr(args, "no_quarot_fp32_had", False):
        quarot_fp32_had = False
    if getattr(args, "quarot_before_quant", False):
        if args.model_type == "qwen3_vl_moe":
            from mllm_quant.rotation.qwen3_vl_moe_quarot import apply_quarot_rotation_qwen3_vl_moe

            logging.info("Applying QuaRot before quantization (Qwen3-VL-MoE)...")
            quarot_Q_cpu = apply_quarot_rotation_qwen3_vl_moe(
                model,
                rotate_mode=args.quarot_rotate_mode,
                fp32_had=quarot_fp32_had,
            )
            quarot_aux_model_type = "qwen3_vl_moe"
            quarot_ffn_had_save = None
        else:
            from mllm_quant.rotation.kimi_vl_quarot import apply_quarot_rotation_kimi_vl

            kimi_ffn = not getattr(args, "no_quarot_kimi_ffn_had", False)
            logging.info("Applying QuaRot before quantization (Kimi-VL)...")
            quarot_Q_cpu = apply_quarot_rotation_kimi_vl(
                model,
                rotate_mode=args.quarot_rotate_mode,
                fp32_had=quarot_fp32_had,
                ffn_had=kimi_ffn,
            )
            quarot_aux_model_type = "kimi_vl"
            quarot_ffn_had_save = kimi_ffn
    
    # Build weight quantization parameters
    weight_quant_params = {
        'w_bits': args.bits,
        'w_groupsize': args.group_size,
        'w_asym': args.asym,
        'w_clip': not args.no_clip,
        'perchannel': True,
        'percdamp': args.damp_percent,
        'act_order': args.act_order,
        'norm': 2.4,
        'grid': 100,
        'maxshrink': 0.8,
    }
    
    # Build mixed-precision config
    mix_w_bits = args.mix_bits
    mix_w_bits_dict = None
    if mix_w_bits:
        mix_w_bits_dict = {
            'attn': args.attn_bits,
            'up_and_gate': args.up_gate_bits,
            'down': args.down_bits,
        }
        logging.info(f"Mixed-precision config: {mix_w_bits_dict}")

    expert_bit_map = None
    if args.expert_bits_json:
        from mllm_quant.expert_mixed_precision import load_expert_bit_config

        expert_bit_map = load_expert_bit_config(args.expert_bits_json)
        logging.info(
            "Loaded expert bit assignment from %s (%d layers)",
            args.expert_bits_json,
            len(expert_bit_map),
        )
        # Shared experts are protected at a fixed bit and do not participate in ILP.
        args.quantize_shared_experts = True
        if mix_w_bits_dict is None:
            mix_w_bits = True
            mix_w_bits_dict = {
                'attn': args.attn_bits,
                'up_and_gate': args.shared_expert_bits,
                'down': args.shared_expert_bits,
            }
        else:
            mix_w_bits_dict['up_and_gate'] = args.shared_expert_bits
            mix_w_bits_dict['down'] = args.shared_expert_bits
    
    logging.info(f"Weight quantization params: w_bits={args.bits}, group_size={args.group_size}, "
                 f"asym={args.asym}, clip={not args.no_clip}")
    if args.quantize_vision:
        logging.info(
            f"Vision encoder GPTQ enabled: vision_bits={args.vision_bits}, "
            f"quantize_vision_projector={args.quantize_vision_projector}"
        )
    
    # Load calibration data for GPTQ / GPTQ_fast (RTN does not need calibration)
    dataloader = None
    if args.method in ("gptq", "gptq_fast"):
        logging.info(f"Loading calibration data from {args.calib_data_path}...")
        from mllm_quant.calibration import get_multimodal_calib_dataset
        
        calibration_data = get_multimodal_calib_dataset(
            data_path=args.calib_data_path,
            image_folder=args.calib_image_folder,
            processor=processor,
            tokenizer=tokenizer,
            n_samples=args.n_samples,
            max_seq_length=args.max_seq_length,
            seed=args.seed,
            model_type=args.model_type,
        )
        
        logging.info(f"Loaded {len(calibration_data)} calibration samples")
        dataloader = calibration_data
    
    # Perform quantization
    logging.info(f"Starting {args.method.upper()} quantization...")
    
    if args.method == "gptq_fast":
        # Multi-GPU parallel expert quantization (same algorithm, faster on MoE models)
        quantizers = quantize_model_fast(
            model=model,
            method="gptq",  # underlying method is still gptq
            dataloader=dataloader,
            dev=device,
            nsamples=args.n_samples,
            weight_quant_params=weight_quant_params,
            mix_w_bits=mix_w_bits,
            mix_w_bits_dict=mix_w_bits_dict,
            keep_min=args.keep_min,
            skip_rare_expert=args.skip_rare_expert,
            percentage=args.percentage,
            quantize_shared_experts=args.quantize_shared_experts,
            quantize_vision=args.quantize_vision,
            quantize_vision_projector=args.quantize_vision_projector,
            vision_bits=args.vision_bits,
            expert_bit_map=expert_bit_map,
        )
    else:
        quantizers = quantize_model(
            model=model,
            method=args.method,
            dataloader=dataloader,
            dev=device,
            nsamples=args.n_samples,
            weight_quant_params=weight_quant_params,
            mix_w_bits=mix_w_bits,
            mix_w_bits_dict=mix_w_bits_dict,
            keep_min=args.keep_min,
            skip_rare_expert=args.skip_rare_expert,
            percentage=args.percentage,
            quantize_shared_experts=args.quantize_shared_experts,
            quantize_vision=args.quantize_vision,
            quantize_vision_projector=args.quantize_vision_projector,
            vision_bits=args.vision_bits,
            expert_bit_map=expert_bit_map,
        )
    
    logging.info(f"Quantization complete. Quantized {len(quantizers)} layers.")
    
    # Save quantized model (all artifacts in the same directory as save_pretrained)
    weights_save_dir = os.path.abspath(os.path.realpath(args.output_path))
    os.makedirs(weights_save_dir, exist_ok=True)

    logging.info(f"Saving quantized model weights to {weights_save_dir}...")
    if args.model_type == "kimi_vl" and getattr(args, "quarot_before_quant", False):
        from mllm_quant.rotation.kimi_vl_quarot import (
            materialize_kimi_rmsn_as_deepseek_rmsnorm_for_save,
        )

        logging.info(
            "Kimi QuaRot: replacing RMSN with DeepseekV3RMSNorm(γ=1) so layernorm weights are saved "
            "(avoids uninitialized norm weights on from_pretrained)."
        )
        materialize_kimi_rmsn_as_deepseek_rmsnorm_for_save(model)
    model.save_pretrained(weights_save_dir)
    if args.model_type == "kimi_vl":
        copy_kimi_vl_remote_code_files(weights_save_dir)

    # QuaRot: same folder as config.json / *.safetensors so from_pretrained(dir) + quarot_aux.pt 同路径即可
    if getattr(args, "quarot_before_quant", False) and quarot_Q_cpu is not None:
        from mllm_quant.rotation.quarot_gptq_compat import save_quarot_aux

        aux_path = save_quarot_aux(
            weights_save_dir,
            quarot_Q_cpu,
            fp32_had=quarot_fp32_had,
            rotate_mode=args.quarot_rotate_mode,
            model_type=quarot_aux_model_type or args.model_type,
            ffn_had=quarot_ffn_had_save,
        )
        logging.info(f"Saved quarot_aux.pt next to model weights: {aux_path}")

    processor.save_pretrained(weights_save_dir)
    tokenizer.save_pretrained(weights_save_dir)

    # Save quantization config
    import json
    quant_config = {
        'method': args.method,
        'bits': args.bits,
        'group_size': args.group_size,
        'asym': args.asym,
        'mix_bits': mix_w_bits,
        'mix_bits_config': mix_w_bits_dict,
        'quantize_vision': args.quantize_vision,
        'quantize_vision_projector': args.quantize_vision_projector,
        'vision_bits': args.vision_bits if args.quantize_vision else None,
        'n_samples': args.n_samples if args.method in ('gptq', 'gptq_fast') else 0,
        'quarot_before_quant': bool(getattr(args, "quarot_before_quant", False)),
        'quarot_rotate_mode': args.quarot_rotate_mode if getattr(args, "quarot_before_quant", False) else None,
        'quarot_fp32_had': quarot_fp32_had if getattr(args, "quarot_before_quant", False) else None,
        'quarot_aux_model_type': quarot_aux_model_type if getattr(args, "quarot_before_quant", False) else None,
        'quarot_kimi_ffn_had': quarot_ffn_had_save if getattr(args, "quarot_before_quant", False) else None,
        'expert_bits_json': args.expert_bits_json,
        'shared_expert_bits': args.shared_expert_bits if args.expert_bits_json else None,
    }
    with open(os.path.join(weights_save_dir, 'quant_config.json'), 'w') as f:
        json.dump(quant_config, f, indent=2)

    logging.info(f"Quantized model saved to {weights_save_dir}")

    # Optional: run evaluation on quantized model
    if args.eval:
        import gc
        logging.info("Cleaning up model from memory before evaluation...")
        del model, processor, tokenizer, quantizers
        gc.collect()
        cleanup_memory(verbos=True)

        # 清理 sys.path 中的本地 transformers 路径，避免子进程导入冲突
        local_tf_path = os.environ.get("LOCAL_TRANSFORMERS_PATH")
        while local_tf_path and local_tf_path in sys.path:
            sys.path.remove(local_tf_path)

        logging.info("Starting evaluation on quantized model...")
        _eval_batch = args.eval_batch_size
        if _eval_batch is None:
            _eval_batch = 1 if args.eval_model_type == "accelerate" else 64
        evaluate_model(
            model_path=args.output_path,
            model_type=args.eval_model_type,
            tasks=args.eval_tasks,
            output_path=args.eval_output_path,
            batch_size=_eval_batch,
            max_model_len=args.eval_max_model_len,
            model_arch=args.model_type,
            num_gpus=args.eval_num_gpus,
            limit=args.eval_limit,
            attn_implementation=args.eval_attn_implementation,
            accelerate_model_parallel=getattr(args, "eval_accelerate_model_parallel", False),
        )


if __name__ == "__main__":
    main()
