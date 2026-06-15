#!/usr/bin/env python3
"""
Record MoE expert selection frequencies across different evaluation datasets.

Supported models:
  - qwen3_vl_moe (Qwen3-VL-30B-A3B-Instruct)
  - kimi_vl       (Kimi-VL-A3B-Instruct)
  - internvl      (InternVL3_5-30B-A3B-HF)

For each model, the script:
  1. Creates the lmms-eval model wrapper (loads the HF model).
  2. Registers forward hooks on every MoE layer to capture expert routing.
  3. Runs each evaluation task one by one, recording per-layer expert counts
     separated into total / text-token / image-token categories.
  4. Saves per-task results and an aggregated result as JSON files.

Usage:
    python record_freq.py \
        --model_path /path/to/model \
        --model_type qwen3_vl_moe \
        --tasks chartqa,textvqa_val \
        --output_dir ./moe_freq/Qwen3-VL-30B-A3B-Instruct \
        --batch_size 1
"""

import os
import re
import sys
import json
import argparse
import logging
import functools
from collections import OrderedDict, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# HF mirror (same as eval_utils.py) - set early, before any HF import.
# Set HF_TOKEN in the environment yourself when private models/datasets need it.
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("record_freq")

# ---------------------------------------------------------------------------
# Mapping: our model_type  →  lmms-eval model name
# ---------------------------------------------------------------------------
LMMS_MODEL_TYPE_MAP = {
    "qwen3_vl_moe": "qwen3_vl",       # simple model (is_simple=True)
    "kimi_vl":       "huggingface",     # chat model, needs trust_remote_code
    "internvl":      "internvl3_5",     # simple model (is_simple=True)
}

ALL_TASKS = [
    "chartqa", "textvqa_val", "mmstar", "mmbench_en_dev",
    "mmvet", "mme", "realworldqa", "mmmu", "pope",
]


# ====================================================================
# ExpertFrequencyRecorder
# ====================================================================
class ExpertFrequencyRecorder:
    """Attach hooks to every MoE block and count expert selections."""

    def __init__(self, model_type: str, detail: bool = False,
                 judge_layer: int = 2, dominant_ratio: float = 1 / 3,
                 attn_mode: str = "vision", use_text_rater: bool = False):
        self.model_type = model_type
        self.detail = detail
        self.judge_layer = judge_layer
        self.dominant_ratio = dominant_ratio
        self.attn_mode = attn_mode  # "vision" / "left" / "adaptive"
        self.use_text_rater = use_text_rater
        self.hooks: list = []
        self.layer_names: list = []
        self.num_experts_map: dict = {}
        self.top_k_map: dict = {}
        self._judge_hidden_states = None   # captured at judge layer for rater selection
        self._tokenizer = None             # for debug printing
        self._printed_current_sample = False  # one print per sample for rater debug

        # per-layer counters  {layer_name: LongTensor[num_experts]}
        self.total_counts: OrderedDict = OrderedDict()
        self.text_counts:  OrderedDict = OrderedDict()
        self.image_counts: OrderedDict = OrderedDict()
        self.dominant_image_counts: OrderedDict = OrderedDict()
        self.redundant_image_counts: OrderedDict = OrderedDict()

        # set by the input-hook on each forward call
        self.current_image_mask = None      # 1-D bool tensor or None
        self.current_dominant_mask = None    # 1-D bool tensor or None (True = top-K important image token)
        self.current_redundant_mask = None   # 1-D bool tensor or None (True = bottom-K least important image token)
        self.current_input_ids = None       # 1-D long tensor for rater debug printing
        self.image_token_id = None
        self._setup_done = False

        # --- adaptive mode tracking ---
        # Maps decoder_layer_idx → dominant absolute positions for current sample
        self._adaptive_current_sample: dict = {}
        # Accumulated pairwise Jaccard: {(i,j): cumulative_sum}
        self._adaptive_overlap_accum: dict = defaultdict(float)
        # Per decoder-layer: list of n_dominant per sample
        self._adaptive_count_accum: dict = defaultdict(list)
        # Per sample n_image_tokens
        self._adaptive_n_img_accum: list = []
        self._adaptive_sample_count: int = 0
        # Sorted decoder layer indices that have MoE blocks
        self._adaptive_moe_decoder_indices: list = []
        # Special/template token IDs to exclude from text-based importance
        self._special_token_ids: set = set()

    # ------------------------------------------------------------------
    def setup(self, model: nn.Module, tokenizer=None):
        """Register all hooks.  Call once, right after model creation."""
        if self._setup_done:
            return

        self._tokenizer = tokenizer
        self._detect_image_token_id(model, tokenizer=tokenizer)
        self._collect_special_token_ids(tokenizer)
        moe_layers = self._find_moe_layers(model)

        if not moe_layers:
            logger.warning("No MoE layers found in the model!")
            return

        # 1) hook the top-level model to capture input_ids
        self._register_input_hook(model)

        # 2) register MoE layer metadata (needed before adaptive hook registration)
        for name, module, num_experts, top_k in moe_layers:
            self.layer_names.append(name)
            self.num_experts_map[name] = num_experts
            self.top_k_map[name] = top_k
            self.total_counts[name] = torch.zeros(num_experts, dtype=torch.long)
            self.text_counts[name]  = torch.zeros(num_experts, dtype=torch.long)
            self.image_counts[name] = torch.zeros(num_experts, dtype=torch.long)
            self.dominant_image_counts[name] = torch.zeros(num_experts, dtype=torch.long)
            self.redundant_image_counts[name] = torch.zeros(num_experts, dtype=torch.long)

        # 3) if detail mode, register attention hooks for dominant token detection
        if self.detail:
            if self.attn_mode == "adaptive":
                self._register_adaptive_hooks(model)
            else:
                self._register_judge_hook(model)

        # 4) hook each MoE block
        for name, module, num_experts, top_k in moe_layers:
            hook = module.register_forward_hook(
                self._make_moe_hook(name, num_experts, top_k)
            )
            self.hooks.append(hook)

        self._setup_done = True
        logger.info(
            "Registered hooks on %d MoE layers  |  image_token_id=%s  |  detail=%s",
            len(moe_layers), self.image_token_id, self.detail,
        )
        if self.detail:
            logger.info(
                "  judge_layer=%d  dominant_ratio=%.2f  attn_mode=%s  text_rater=%s",
                self.judge_layer, self.dominant_ratio, self.attn_mode, self.use_text_rater,
            )
        for n, _, ne, tk in moe_layers:
            logger.info("  %s  num_experts=%d  top_k=%d", n, ne, tk)

    # ------------------------------------------------------------------
    def reset(self):
        """Zero all counters (call before each new task)."""
        for name in self.layer_names:
            self.total_counts[name].zero_()
            self.text_counts[name].zero_()
            self.image_counts[name].zero_()
            self.dominant_image_counts[name].zero_()
            self.redundant_image_counts[name].zero_()
        self.current_dominant_mask = None
        self.current_redundant_mask = None
        self._judge_hidden_states = None
        self._printed_current_sample = False
        # adaptive mode
        self._adaptive_current_sample.clear()
        self._adaptive_overlap_accum.clear()
        self._adaptive_count_accum.clear()
        self._adaptive_n_img_accum.clear()
        self._adaptive_sample_count = 0

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _detect_image_token_id(self, model, tokenizer=None):
        # --- 1. Check model config ----------------------------------------
        config = model.config
        for attr in ("image_token_id", "media_placeholder_token_id"):
            if hasattr(config, attr):
                self.image_token_id = getattr(config, attr)
                return
        # nested config (e.g. InternVL wraps a text_config)
        for sub in ("text_config", "llm_config"):
            if hasattr(config, sub):
                sub_cfg = getattr(config, sub)
                for attr in ("image_token_id", "media_placeholder_token_id"):
                    if hasattr(sub_cfg, attr):
                        self.image_token_id = getattr(sub_cfg, attr)
                        return

        # --- 2. Check model attributes (e.g. InternVL img_context_token_id)
        for attr in ("img_context_token_id", "image_token_id"):
            val = getattr(model, attr, None)
            if val is not None:
                self.image_token_id = val
                return

        # --- 3. Tokenizer fallback: resolve known image token strings -----
        # InternVL uses <IMG_CONTEXT>, others may use <image> or <|image|>
        if tokenizer is not None:
            _KNOWN_IMG_TOKENS = ("<IMG_CONTEXT>", "<image>", "<|image|>")
            unk_id = getattr(tokenizer, "unk_token_id", None)
            for tok_str in _KNOWN_IMG_TOKENS:
                try:
                    tid = tokenizer.convert_tokens_to_ids(tok_str)
                    if tid is not None and tid != unk_id:
                        self.image_token_id = tid
                        logger.info(
                            "Resolved image_token_id=%d from tokenizer token '%s'",
                            tid, tok_str,
                        )
                        return
                except Exception:
                    pass

    def _collect_special_token_ids(self, tokenizer):
        """Build a set of special / template token IDs to exclude from
        text-based importance computation (system prompt already filtered
        by position; this catches remaining tokens like <|im_start|>,
        <|im_end|>, <|vision_end|>, role names, etc.)."""
        if tokenizer is None:
            return
        sids = set(getattr(tokenizer, "all_special_ids", None) or [])
        _EXTRA_SPECIAL = (
            "<|im_start|>", "<|im_end|>",
            "<|vision_start|>", "<|vision_end|>",
            "<|endoftext|>", "<|object_ref_start|>", "<|object_ref_end|>",
            "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
            "<s>", "</s>", "<pad>", "<unk>",
        )
        unk_id = getattr(tokenizer, "unk_token_id", None)
        for tok_str in _EXTRA_SPECIAL:
            try:
                tid = tokenizer.convert_tokens_to_ids(tok_str)
                if tid is not None and tid != unk_id:
                    sids.add(tid)
            except Exception:
                pass
        # Role-indicator tokens (not special per tokenizer, but semantically
        # irrelevant for vision-token importance computation)
        for role_str in ("assistant", "system", "user"):
            try:
                role_ids = tokenizer.encode(role_str, add_special_tokens=False)
                for rid in role_ids:
                    if rid != unk_id:
                        sids.add(rid)
            except Exception:
                pass
        sids.discard(self.image_token_id)
        self._special_token_ids = sids
        if sids:
            logger.info("Collected %d special token IDs for text filtering", len(sids))

    def _find_moe_layers(self, model):
        """Return list of (name, module, num_experts, top_k)."""
        results = []
        for name, module in model.named_modules():
            cls_name = type(module).__name__
            is_moe = False

            # explicit class-name matching
            if "SparseMoeBlock" in cls_name or "SparseMoe" in cls_name:
                is_moe = True
            elif cls_name in (
                "DeepseekV3MoE", "DeepseekV2MoE", "MixtralSparseMoeBlock",
            ):
                is_moe = True
            elif (
                "MoE" in cls_name
                and not any(x in cls_name for x in ("MoEGate", "Config", "Token"))
                and hasattr(module, "gate")
                and hasattr(module, "experts")
            ):
                is_moe = True

            if is_moe:
                ne = self._get_num_experts(module)
                tk = self._get_top_k(module)
                if ne and tk:
                    results.append((name, module, ne, tk))
                else:
                    logger.warning(
                        "MoE-like layer %s (%s) but num_experts=%s, top_k=%s  – skipped",
                        name, cls_name, ne, tk,
                    )
        return results

    @staticmethod
    def _get_num_experts(module):
        if hasattr(module, "num_experts"):
            return module.num_experts
        gate = getattr(module, "gate", None)
        if gate is not None:
            if isinstance(gate, nn.Linear):
                return gate.out_features
            for a in ("n_routed_experts", "num_experts"):
                if hasattr(gate, a):
                    return getattr(gate, a)
        cfg = getattr(module, "config", None)
        if cfg is not None:
            for a in ("num_experts", "n_routed_experts"):
                if hasattr(cfg, a):
                    return getattr(cfg, a)
        experts = getattr(module, "experts", None)
        if isinstance(experts, nn.ModuleList):
            return len(experts)
        return None

    @staticmethod
    def _get_top_k(module):
        for a in ("top_k", "num_experts_per_tok"):
            if hasattr(module, a):
                return getattr(module, a)
        gate = getattr(module, "gate", None)
        if gate is not None:
            for a in ("top_k", "num_experts_per_tok"):
                if hasattr(gate, a):
                    return getattr(gate, a)
        cfg = getattr(module, "config", None)
        if cfg is not None:
            for a in ("num_experts_per_tok", "top_k"):
                if hasattr(cfg, a):
                    return getattr(cfg, a)
        return None

    # ------------------------------------------------------------------
    def _register_input_hook(self, model):
        """
        Capture ``input_ids`` so we can build an image-token mask.

        Two complementary mechanisms are used:

        1. **forward pre-hook** on the top-level model – works for models
           whose ``forward()`` receives ``input_ids`` directly (e.g. Qwen3-VL,
           Kimi-VL).
        2. **generate() wrapper** – for models like InternVL whose custom
           ``generate()`` converts ``input_ids`` → ``inputs_embeds`` *before*
           calling ``forward()``, so the forward hook never sees ``input_ids``.
           The wrapper captures ``input_ids`` at the ``generate()`` entry point.

        The forward hook only *updates* the mask when it finds ``input_ids``;
        it never clears an existing mask (which may have been set by the
        generate wrapper for the prefill step).
        """
        recorder = self

        # ---- (a) forward pre-hook ----------------------------------------
        def _pre_hook(_module, args, kwargs):
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                candidate = args[0]
                if isinstance(candidate, torch.Tensor) and candidate.dtype in (
                    torch.long, torch.int, torch.int32, torch.int64,
                ):
                    input_ids = candidate

            # Only UPDATE the mask when input_ids is available.
            # Do NOT clear it – the generate wrapper may have set a valid
            # prefill mask that should persist until the first decode step.
            if input_ids is not None and recorder.image_token_id is not None:
                recorder.current_image_mask = (
                    input_ids == recorder.image_token_id
                ).reshape(-1)
                recorder.current_input_ids = input_ids.reshape(-1)
                recorder.current_dominant_mask = None
                recorder.current_redundant_mask = None
                recorder._printed_current_sample = False

        hook = model.register_forward_pre_hook(_pre_hook, with_kwargs=True)
        self.hooks.append(hook)

        # ---- (b) generate() wrapper --------------------------------------
        original_generate = model.generate

        @functools.wraps(original_generate)
        def _generate_wrapper(*args, **kwargs):
            # Try to find input_ids from kwargs or positional args.
            input_ids = kwargs.get("input_ids")
            if input_ids is None:
                for a in args:
                    if (
                        isinstance(a, torch.Tensor)
                        and a.dtype in (torch.long, torch.int, torch.int32, torch.int64)
                        and a.dim() >= 1
                    ):
                        input_ids = a
                        break

            if input_ids is not None and recorder.image_token_id is not None:
                recorder.current_image_mask = (
                    input_ids == recorder.image_token_id
                ).reshape(-1)
                recorder.current_input_ids = input_ids.reshape(-1)
                recorder.current_dominant_mask = None
                recorder.current_redundant_mask = None
                recorder._printed_current_sample = False

            return original_generate(*args, **kwargs)

        model.generate = _generate_wrapper

    # ------------------------------------------------------------------
    #  Dominant-token detection (detail mode)
    # ------------------------------------------------------------------
    @staticmethod
    def _find_decoder_layers(model) -> "nn.ModuleList | None":
        """Locate the nn.ModuleList of LLM decoder layers."""
        candidates = []
        for _name, module in model.named_modules():
            if not isinstance(module, nn.ModuleList) or len(module) < 4:
                continue
            first = module[0]
            if hasattr(first, "self_attn") and hasattr(first, "input_layernorm"):
                candidates.append((_name, module))
        if not candidates:
            return None
        return max(candidates, key=lambda x: len(x[1]))[1]

    def _register_judge_hook(self, model):
        """Register a forward hook on the **self_attn** sub-module of the
        decoder layer at *judge_layer*.

        Why self_attn and not the full decoder layer?
        - Modern decoder layers (e.g. Qwen3-VL-MoE) return a single tensor
          and silently discard attn_weights (``hidden, _ = self.self_attn(...)``).
        - The self_attn module still returns ``(attn_output, attn_weights)``
          when eager attention is used, so we can capture weights there.
        - The self_attn hook fires *before* the MoE/FFN in the same layer,
          which means even the judge_layer's own MoE gets the dominant mask.
        """
        decoder_layers = self._find_decoder_layers(model)
        if decoder_layers is None:
            logger.warning("Cannot find decoder layers – skipping dominant token detection")
            return
        if self.judge_layer >= len(decoder_layers):
            logger.warning(
                "judge_layer=%d >= num_decoder_layers=%d – skipping dominant token detection",
                self.judge_layer, len(decoder_layers),
            )
            return

        target_attn = decoder_layers[self.judge_layer].self_attn
        recorder = self

        # --- pre-hook: force output_attentions=True so that models like
        #     DeepseekV3 (Kimi-VL) don't discard the computed weights.
        #     Also capture hidden_states for Text Rater selection.
        def _attn_pre_hook(_module, args, kwargs):
            kwargs["output_attentions"] = True
            # Capture hidden_states for rater selection
            hs = kwargs.get("hidden_states")
            if hs is None and args:
                hs = args[0]
            if hs is not None:
                recorder._judge_hidden_states = hs
            return args, kwargs

        pre_h = target_attn.register_forward_pre_hook(_attn_pre_hook, with_kwargs=True)
        self.hooks.append(pre_h)

        # --- post-hook: capture the attention weights
        def _attn_hook(_module, _input, output):
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                return
            attn_weights = output[1]
            if attn_weights is None:
                return
            if not isinstance(attn_weights, torch.Tensor) or attn_weights.dim() != 4:
                return

            # Only process prefill (seq_len > 1); skip decode steps
            if attn_weights.shape[-1] <= 1:
                return

            recorder._compute_dominant_mask(attn_weights)

        hook = target_attn.register_forward_hook(_attn_hook)
        self.hooks.append(hook)
        logger.info(
            "Registered attention hook on decoder_layer[%d].self_attn for dominant-token detection",
            self.judge_layer,
        )

    def _register_adaptive_hooks(self, model):
        """Register self_attn hooks on ALL decoder layers that contain MoE blocks.

        Each layer computes its own dominant mask from its own attention weights.
        Within a single forward pass the execution order is:
            layer[i].self_attn (pre → fwd → post)  →  layer[i].MoE hook
        so the MoE hook always reads the dominant_mask computed by the same layer.
        """
        decoder_layers = self._find_decoder_layers(model)
        if decoder_layers is None:
            logger.warning("Cannot find decoder layers – skipping adaptive detection")
            return

        moe_decoder_indices = set()
        for name in self.layer_names:
            m = re.search(r"layers\.(\d+)", name)
            if m:
                moe_decoder_indices.add(int(m.group(1)))

        self._adaptive_moe_decoder_indices = sorted(moe_decoder_indices)
        recorder = self

        for idx in self._adaptive_moe_decoder_indices:
            if idx >= len(decoder_layers):
                continue
            target_attn = decoder_layers[idx].self_attn

            def _make_pre_hook(layer_idx):
                def _attn_pre_hook(_module, args, kwargs):
                    kwargs["output_attentions"] = True
                    hs = kwargs.get("hidden_states")
                    if hs is None and args:
                        hs = args[0]
                    if hs is not None:
                        recorder._judge_hidden_states = hs
                    return args, kwargs
                return _attn_pre_hook

            def _make_post_hook(layer_idx):
                def _attn_hook(_module, _input, output):
                    if not isinstance(output, (tuple, list)) or len(output) < 2:
                        return
                    attn_weights = output[1]
                    if attn_weights is None:
                        return
                    if not isinstance(attn_weights, torch.Tensor) or attn_weights.dim() != 4:
                        return
                    if attn_weights.shape[-1] <= 1:
                        return

                    recorder._compute_dominant_mask(attn_weights)

                    # track dominant set for adaptive stats (cpu-safe for multi-GPU)
                    if (recorder.current_dominant_mask is not None
                            and recorder.current_image_mask is not None):
                        dom_abs = (recorder.current_dominant_mask.cpu()
                                   & recorder.current_image_mask.cpu()
                                   ).nonzero(as_tuple=True)[0]
                        recorder._adaptive_current_sample[layer_idx] = frozenset(
                            dom_abs.tolist()
                        )
                return _attn_hook

            pre_h = target_attn.register_forward_pre_hook(
                _make_pre_hook(idx), with_kwargs=True,
            )
            self.hooks.append(pre_h)

            post_h = target_attn.register_forward_hook(_make_post_hook(idx))
            self.hooks.append(post_h)

        logger.info(
            "Registered adaptive attention hooks on %d decoder layers: %s",
            len(self._adaptive_moe_decoder_indices),
            self._adaptive_moe_decoder_indices,
        )

    def _select_text_raters(self, img_pos, text_pos):
        """Select Text Raters following SparseVLM §3.2.

        Uses hidden_states at the judge layer to compute vision→text
        embedding similarity, then picks text tokens above the mean as raters.

        Returns:
            rater_pos: 1-D LongTensor of sequence positions for selected raters,
                       or *text_pos* unchanged if rater selection is disabled or fails.
        """
        hs = self._judge_hidden_states
        if hs is None or hs.dim() != 3:
            return text_pos

        v_t = hs[:, img_pos, :]    # (B, #img, D)
        t_t = hs[:, text_pos, :]   # (B, #text, D)
        # vision→text similarity: (B, #img, #text) → softmax over text → mean over img
        sim = torch.matmul(v_t, t_t.transpose(1, 2))   # (B, #img, #text)
        sim = sim.softmax(dim=-1).mean(dim=1)            # (B, #text)
        score = sim[0]                                    # (#text,)  (assume B=1)

        rater_mask = score > score.mean()
        if not rater_mask.any():
            return text_pos

        rater_pos = text_pos[rater_mask]

        # Debug print: once per sample (skip duplicate prints in adaptive mode)
        if not self._printed_current_sample and self._tokenizer is not None:
            self._printed_current_sample = True
            try:
                ids = self.current_input_ids
                logger.info(
                    "[TextRater] %d/%d text tokens selected as raters (threshold=%.4f)",
                    len(rater_pos), len(text_pos), score.mean().item(),
                )
                if ids is not None:
                    text_ids = ids.cpu()[text_pos.cpu()].tolist()
                    rater_ids = ids.cpu()[rater_pos.cpu()].tolist()
                    all_text = self._tokenizer.decode(text_ids)
                    rater_text = self._tokenizer.decode(rater_ids)
                    logger.info("[TextRater] All text: %s", all_text[:300])
                    logger.info("[TextRater] Rater text: %s", rater_text[:300])
            except Exception as e:
                logger.warning("[TextRater] Debug print failed: %s", e)

        return rater_pos

    def _compute_dominant_mask(self, attn_weights: torch.Tensor):
        """Classify image tokens into dominant / redundant based on
        attention weights at the current layer.

        Modes (``self.attn_mode``):
        - ``"vision"`` / ``"adaptive"``: importance = mean attention from text
          (or Text Raters) to each image token.
        - ``"left"``:  importance = mean attention from all other tokens to
          each image token (excluding self-attention).

        ``adaptive`` uses the same algorithm as ``vision`` but is applied
        independently at every MoE-containing decoder layer rather than a
        single judge layer.

        When ``self.use_text_rater`` is True and mode is ``"vision"`` or
        ``"adaptive"``, only Text Raters are used as query tokens.

        Args:
            attn_weights: (B, H, L, L)  – attention probabilities.
        """
        mask = self.current_image_mask
        if mask is None:
            return

        L = attn_weights.shape[-1]
        if mask.shape[0] != L:
            return

        # align mask to the same device as attn_weights (multi-GPU safe)
        aw_dev = attn_weights.device
        mask = mask.to(aw_dev)

        img_pos = mask.nonzero(as_tuple=True)[0]
        text_pos = (~mask).nonzero(as_tuple=True)[0]
        if len(img_pos) == 0:
            return

        with torch.no_grad():
            if self.attn_mode in ("vision", "adaptive"):
                if len(text_pos) == 0:
                    return
                # 1) Exclude system prompt tokens (before first image token)
                instr_mask = text_pos > img_pos[0]
                if instr_mask.any():
                    text_pos = text_pos[instr_mask]
                # 2) Exclude special / template tokens (<|im_end|>, etc.)
                if self._special_token_ids and self.current_input_ids is not None:
                    ids_at = self.current_input_ids.cpu()[text_pos.cpu()]
                    special_t = torch.tensor(
                        sorted(self._special_token_ids), dtype=ids_at.dtype,
                    )
                    keep = ~torch.isin(ids_at, special_t)
                    if keep.any():
                        text_pos = text_pos[keep.to(text_pos.device)]
                if len(text_pos) == 0:
                    return
                query_pos = text_pos
                if self.use_text_rater:
                    query_pos = self._select_text_raters(img_pos, text_pos)
                    query_pos = query_pos.to(aw_dev)
                t2i = attn_weights[:, :, query_pos][:, :, :, img_pos]
                importance = t2i.float().mean(dim=(0, 1, 2))  # (#img,)
            else:
                # "left": all other tokens → each image token
                all_to_img = attn_weights[:, :, :, img_pos].float()
                for j, pos in enumerate(img_pos):
                    all_to_img[:, :, pos, j] = 0.0
                importance = all_to_img.mean(dim=(0, 1, 2))

            num_dominant = max(1, int(len(img_pos) * self.dominant_ratio))
            _, topk_idx = importance.topk(num_dominant)
            _, bottomk_idx = importance.topk(num_dominant, largest=False)

            dominant_mask = torch.zeros(L, dtype=torch.bool)
            dominant_mask[img_pos.cpu()[topk_idx.cpu()]] = True
            self.current_dominant_mask = dominant_mask

            redundant_mask = torch.zeros(L, dtype=torch.bool)
            redundant_mask[img_pos.cpu()[bottomk_idx.cpu()]] = True
            self.current_redundant_mask = redundant_mask

    # ------------------------------------------------------------------
    def _make_moe_hook(self, layer_name: str, num_experts: int, top_k: int):
        """Create a forward-hook closure for one MoE block."""
        recorder = self

        def _hook(module, inp, output):
            hidden_states = inp[0]  # original (possibly 3-D) tensor

            if hidden_states.dim() == 3:
                num_tokens = hidden_states.shape[0] * hidden_states.shape[1]
            elif hidden_states.dim() == 2:
                num_tokens = hidden_states.shape[0]
            else:
                return

            with torch.no_grad():
                router_indices = _extract_routing(
                    module, hidden_states, output, num_experts, top_k,
                )
                if router_indices is None:
                    return

                # router_indices: [num_tokens, top_k]  (long)
                router_indices = router_indices.reshape(-1, top_k)
                actual_tokens = router_indices.shape[0]

                # --- accumulate counts -----------------------------------
                mask = recorder.current_image_mask
                dmask = recorder.current_dominant_mask
                for k_idx in range(top_k):
                    col = router_indices[:, k_idx].long()
                    counts = torch.bincount(col, minlength=num_experts)
                    recorder.total_counts[layer_name] += counts.cpu()

                    if mask is not None and mask.shape[0] == actual_tokens:
                        dev = col.device
                        m = mask.to(dev)
                        if m.any():
                            ic = torch.bincount(col[m],  minlength=num_experts)
                            recorder.image_counts[layer_name] += ic.cpu()
                        if (~m).any():
                            tc = torch.bincount(col[~m], minlength=num_experts)
                            recorder.text_counts[layer_name] += tc.cpu()

                        # detail mode: split image tokens into dominant / redundant
                        if recorder.detail and dmask is not None and dmask.shape[0] == actual_tokens:
                            dm = dmask.to(dev)
                            dom_sel = m & dm       # dominant image
                            red_sel = m & (~dm)    # redundant image
                            if dom_sel.any():
                                dc = torch.bincount(col[dom_sel], minlength=num_experts)
                                recorder.dominant_image_counts[layer_name] += dc.cpu()
                            if red_sel.any():
                                rc = torch.bincount(col[red_sel], minlength=num_experts)
                                recorder.redundant_image_counts[layer_name] += rc.cpu()

                    elif mask is not None and mask.shape[0] != actual_tokens:
                        recorder.text_counts[layer_name] += counts.cpu()

        return _hook

    # ------------------------------------------------------------------
    #  Adaptive mode: per-sample stats accumulation
    # ------------------------------------------------------------------
    def accumulate_adaptive_stats(self):
        """Call after each sample's forward pass in adaptive mode.

        Computes pairwise Jaccard overlap between the dominant sets of
        different decoder layers for this sample, and accumulates running
        statistics for later export.
        """
        if self.attn_mode != "adaptive" or not self._adaptive_current_sample:
            return

        indices = sorted(self._adaptive_current_sample.keys())

        # record n_image for this sample
        if self.current_image_mask is not None:
            n_img = int(self.current_image_mask.sum().item())
            self._adaptive_n_img_accum.append(n_img)

        for idx in indices:
            dom_set = self._adaptive_current_sample[idx]
            self._adaptive_count_accum[idx].append(len(dom_set))

        # pairwise Jaccard
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                li, lj = indices[i], indices[j]
                si = self._adaptive_current_sample[li]
                sj = self._adaptive_current_sample[lj]
                inter = len(si & sj)
                union = len(si | sj)
                jacc = inter / union if union > 0 else 0.0
                self._adaptive_overlap_accum[(li, lj)] += jacc

        self._adaptive_sample_count += 1
        self._adaptive_current_sample.clear()

    def get_adaptive_token_stats(self) -> dict:
        """Return accumulated adaptive stats as a JSON-serialisable dict.

        Structure::

            {
                "config": { "num_samples", "dominant_ratio", "attn_mode" },
                "moe_decoder_layers": [1, 3, 5, ...],
                "per_layer": { "1": { "avg_n_dominant", "avg_n_image" }, ... },
                "pairwise_jaccard": { "1-3": 0.75, ... },
                "consecutive_jaccard": [ {"layer_a": 1, "layer_b": 3, "jaccard": 0.75}, ... ]
            }
        """
        n = max(self._adaptive_sample_count, 1)
        avg_n_img = (sum(self._adaptive_n_img_accum) / n
                     if self._adaptive_n_img_accum else 0)

        per_layer = {}
        for idx in self._adaptive_moe_decoder_indices:
            counts = self._adaptive_count_accum.get(idx, [])
            avg_dom = sum(counts) / len(counts) if counts else 0
            per_layer[str(idx)] = {
                "avg_n_dominant": round(avg_dom, 2),
                "avg_n_image": round(avg_n_img, 2),
            }

        pairwise = {}
        for (li, lj), total_jacc in self._adaptive_overlap_accum.items():
            pairwise[f"{li}-{lj}"] = round(total_jacc / n, 4)

        # consecutive pairs among MoE decoder layers
        consecutive = []
        sorted_indices = self._adaptive_moe_decoder_indices
        for k in range(len(sorted_indices) - 1):
            li, lj = sorted_indices[k], sorted_indices[k + 1]
            key = (li, lj)
            jacc = self._adaptive_overlap_accum.get(key, 0.0) / n
            consecutive.append({
                "layer_a": li, "layer_b": lj,
                "jaccard": round(jacc, 4),
            })

        return {
            "config": {
                "num_samples": self._adaptive_sample_count,
                "dominant_ratio": self.dominant_ratio,
                "attn_mode": "adaptive",
                "text_rater": self.use_text_rater,
            },
            "moe_decoder_layers": self._adaptive_moe_decoder_indices,
            "per_layer": per_layer,
            "pairwise_jaccard": pairwise,
            "consecutive_jaccard": consecutive,
        }

    # ------------------------------------------------------------------
    def get_results(self) -> OrderedDict:
        results = OrderedDict()
        for name in self.layer_names:
            entry = {
                "total": self.total_counts[name].tolist(),
                "text":  self.text_counts[name].tolist(),
                "image": self.image_counts[name].tolist(),
            }
            if self.detail:
                entry["dominant_image"]  = self.dominant_image_counts[name].tolist()
                entry["redundant_image"] = self.redundant_image_counts[name].tolist()
            results[name] = entry
        return results

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ====================================================================
# Routing extraction  (works for Qwen3 / Qwen3-VL-MoE / DeepseekV3)
# ====================================================================
def _extract_routing(module, hidden_states, output, num_experts, top_k):
    """
    Try to obtain router_indices [num_tokens, top_k] from *output* first
    (cheap); fall back to recomputing via module.gate (more general).
    """
    if hidden_states.dim() == 3:
        num_tokens = hidden_states.shape[0] * hidden_states.shape[1]
    else:
        num_tokens = hidden_states.shape[0]

    # ----- Method 1: extract from output (Qwen3 style) ---------------
    if isinstance(output, tuple) and len(output) >= 2:
        candidate = output[1]
        if (
            isinstance(candidate, torch.Tensor)
            and candidate.dim() == 2
            and candidate.shape[-1] == num_experts
            and candidate.shape[0] == num_tokens
        ):
            # candidate is router_logits  [num_tokens, num_experts]
            weights = torch.softmax(candidate, dim=-1, dtype=torch.float)
            _, indices = torch.topk(weights, top_k, dim=-1)
            return indices

    # ----- Method 2: recompute from gate ------------------------------
    gate = getattr(module, "gate", None)
    if gate is None:
        return None

    try:
        if isinstance(gate, nn.Linear):
            hs = hidden_states.reshape(-1, hidden_states.shape[-1])
            logits = gate(hs)
            weights = torch.softmax(logits, dim=-1, dtype=torch.float)
            _, indices = torch.topk(weights, top_k, dim=-1)
            return indices
        else:
            # Complex gate (e.g. MoEGate in DeepseekV3 / Kimi-VL)
            gate_out = gate(hidden_states)
            if isinstance(gate_out, tuple):
                idx_candidate = gate_out[0]
                if idx_candidate.dtype in (
                    torch.long, torch.int, torch.int32, torch.int64,
                ):
                    return idx_candidate.reshape(-1, top_k)
                # else treat as logits
                weights = torch.softmax(idx_candidate, dim=-1, dtype=torch.float)
                _, indices = torch.topk(weights, top_k, dim=-1)
                return indices
            else:
                weights = torch.softmax(gate_out, dim=-1, dtype=torch.float)
                _, indices = torch.topk(weights, top_k, dim=-1)
                return indices
    except Exception as exc:
        logger.warning("Routing extraction failed for %s: %s", type(module).__name__, exc)
        return None


# ====================================================================
# trust_remote_code monkey-patch  (needed for kimi_vl)
# ====================================================================
_originals = {}

def patch_trust_remote_code():
    """Make all AutoXxx.from_pretrained calls default to trust_remote_code=True."""
    import transformers

    targets = [
        transformers.AutoConfig,
        transformers.AutoModel,
        transformers.AutoModelForCausalLM,
        transformers.AutoTokenizer,
        transformers.AutoProcessor,
    ]
    if hasattr(transformers, "AutoModelForImageTextToText"):
        targets.append(transformers.AutoModelForImageTextToText)

    for cls in targets:
        orig = cls.from_pretrained
        _originals[cls] = orig

        def _make_patched(original):
            @classmethod
            def _patched(klass, *a, **kw):
                kw.setdefault("trust_remote_code", True)
                return original.__func__(klass, *a, **kw)
            return _patched

        cls.from_pretrained = _make_patched(orig)


def unpatch_trust_remote_code():
    for cls, orig in _originals.items():
        cls.from_pretrained = orig
    _originals.clear()


# ====================================================================
# lmms-eval integration
# ====================================================================
def create_lmms_model(model_path: str, model_type: str, batch_size: int = 1,
                      use_eager_attn: bool = False):
    """Instantiate the lmms-eval model wrapper (loads the HF model)."""
    from lmms_eval.models import get_model

    lmms_type = LMMS_MODEL_TYPE_MAP[model_type]
    logger.info("Creating lmms-eval model  type=%s  path=%s", lmms_type, model_path)

    model_args = f"pretrained={model_path}"
    if model_type == "kimi_vl":
        model_args += ",trust_remote_code=True"
    if use_eager_attn:
        if model_type == "internvl":
            model_args += ",use_flash_attn=False"
        else:
            model_args += ",attn_implementation=eager"

    lm = get_model(lmms_type).create_from_arg_string(
        model_args,
        {"batch_size": batch_size},
    )

    return lm


def run_single_task(lm, task_name: str, batch_size: int = 1, limit: int = None):
    """Run one lmms-eval task on a pre-loaded model."""
    from lmms_eval import evaluator
    from lmms_eval.tasks import TaskManager

    task_manager = TaskManager(verbosity="INFO")

    print(f"Running task {task_name} with limit {limit}")

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=[task_name],
        batch_size=batch_size,
        task_manager=task_manager,
        log_samples=False,
        limit=limit,
    )
    return results


def print_task_metrics(results: dict, task_name: str):
    """Extract and print evaluation metrics from lmms-eval results."""
    if results is None or "results" not in results:
        logger.warning("No evaluation results available for %s", task_name)
        return

    task_metrics = results["results"].get(task_name, {})
    if not task_metrics:
        logger.warning("No metrics found for task %s in results", task_name)
        return

    metrics = {
        k: v for k, v in task_metrics.items()
        if not k.startswith("alias") and "stderr" not in k
        and "submission" not in k and v != "N/A"
    }
    parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
             for k, v in metrics.items()]
    logger.info(">> [%s] %s", task_name, "  ".join(parts) if parts else "(no metrics)")


# ====================================================================
# Aggregation helpers
# ====================================================================
def aggregate_results(aggregated, task_results):
    """In-place add task_results into aggregated."""
    _KEYS = ["total", "text", "image", "dominant_image", "redundant_image"]
    if not aggregated:
        for layer_name, data in task_results.items():
            entry = {}
            for key in _KEYS:
                if key in data:
                    entry[key] = list(data[key])
            aggregated[layer_name] = entry
    else:
        for layer_name, data in task_results.items():
            for key in _KEYS:
                if key not in data:
                    continue
                if key not in aggregated[layer_name]:
                    aggregated[layer_name][key] = list(data[key])
                else:
                    for i in range(len(data[key])):
                        aggregated[layer_name][key][i] += data[key][i]


def run_summary(directory: str) -> None:
    """Read all per-task JSON files from *directory*, aggregate and save.

    This is useful when a recording run was interrupted and did not produce
    the final ``aggregated.json``.

    Usage::

        python record_freq.py --summary /path/to/output_dir
    """
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        logger.error("Directory does not exist: %s", directory)
        sys.exit(1)

    json_files = sorted(
        f for f in os.listdir(directory)
        if f.endswith(".json") and f != "aggregated.json"
    )
    if not json_files:
        logger.error("No per-task JSON files found in %s", directory)
        sys.exit(1)

    aggregated: OrderedDict = OrderedDict()
    for fname in json_files:
        fpath = os.path.join(directory, fname)
        with open(fpath, "r") as fp:
            task_results = json.load(fp, object_pairs_hook=OrderedDict)
        aggregate_results(aggregated, task_results)
        logger.info("Loaded  %s", fpath)

    save_path = os.path.join(directory, "aggregated.json")
    with open(save_path, "w") as fp:
        json.dump(aggregated, fp, indent=2)
    logger.info("Saved aggregated result (%d tasks) -> %s", len(json_files), save_path)


# ====================================================================
# COCO calibration mode
# ====================================================================
_MODEL_CLS_MAP = {
    "qwen3_vl_moe": "Qwen2_5_VLForConditionalGeneration",
    "kimi_vl":       "AutoModelForCausalLM",
    "internvl":      "AutoModel",
}


def load_model_and_processor(model_path: str, model_type: str,
                             use_eager_attn: bool = False):
    """Load model + processor directly (no lmms-eval wrapper).

    Uses the same custom model classes as quantize.py to ensure
    architecture compatibility (e.g. Qwen3VLMoe != Qwen2.5VL).
    """
    from transformers import AutoTokenizer, AutoProcessor

    _project_root = str(Path(__file__).resolve().parents[2])
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if model_type == "qwen3_vl_moe":
        from mllm_quant.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
        if use_eager_attn:
            load_kwargs["attn_implementation"] = "eager"
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_path, **load_kwargs,
        ).eval()
    elif model_type == "kimi_vl":
        from mllm_quant.models.kimi_vl import KimiVLForConditionalGeneration
        load_kwargs["trust_remote_code"] = True
        model = KimiVLForConditionalGeneration.from_pretrained(
            model_path, **load_kwargs,
        ).eval()
    elif model_type == "internvl":
        from mllm_quant.models.internvl import InternVLForConditionalGeneration
        load_kwargs["trust_remote_code"] = True
        if use_eager_attn:
            load_kwargs["use_flash_attn"] = False
        model = InternVLForConditionalGeneration.from_pretrained(
            model_path, **load_kwargs,
        ).eval()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    return model, processor, tokenizer


def run_coco_recording(args):
    """Record expert selection frequencies on COCO calibration data."""
    _project_root = str(Path(__file__).resolve().parents[2])
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from mllm_quant.calibration.multimodal_calib import (
        load_image, process_calibration_item,
    )
    import numpy as np

    model_name = Path(args.model_path).name
    n_samples = args.n_samples

    # --- output file name ---
    # e.g. Qwen3-VL-30B-A3B-Instruct_coco512_vision_r0.2_rater_honly.json
    if args.detail:
        ratio_str = f"{args.dominant_ratio:.2f}".rstrip("0").rstrip(".")
        parts = [model_name, f"coco{n_samples}", args.attn_mode, f"r{ratio_str}"]
        if args.text_rater:
            parts.append("rater")
        if args.human_only:
            parts.append("honly")
        fname = "_".join(parts) + ".json"
    else:
        parts = [model_name, f"coco{n_samples}"]
        if args.human_only:
            parts.append("honly")
        fname = "_".join(parts) + ".json"

    coco_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "coco")
    save_path = os.path.join(coco_dir, fname)

    if not args.debug:
        os.makedirs(coco_dir, exist_ok=True)

    # --- 1. load model & processor ---
    logger.info("Loading model directly (no lmms-eval) ...")
    model, processor, tokenizer = load_model_and_processor(
        args.model_path, args.model_type,
        use_eager_attn=args.detail,
    )

    # --- 2. load & process calibration data ---
    logger.info(
        "Loading COCO calibration data: %s  n_samples=%d  human_only=%s",
        args.calib_data, n_samples, args.human_only,
    )
    with open(args.calib_data, "r") as f:
        dataset = json.load(f)
    rng = np.random.default_rng(seed=42)
    rng.shuffle(dataset)

    from tqdm import tqdm
    calib_data = []
    for i in tqdm(range(n_samples), desc="Processing calibration data"):
        item = dataset[i % len(dataset)]
        # --human_only: strip assistant response
        if args.human_only:
            item = dict(item)
            item["conversations"] = [
                c for c in item.get("conversations", []) if c.get("from") == "human"
            ]
        # load image
        images = None
        img_path = item.get("image")
        if img_path:
            full_path = os.path.join(args.calib_img, img_path)
            if os.path.exists(full_path):
                images = [load_image(full_path)]
        try:
            d = process_calibration_item(
                images=images, data_item=item,
                processor=processor, tokenizer=tokenizer,
                max_seq_length=4096, model_type=args.model_type,
            )
            if d is not None:
                calib_data.append(d)
        except Exception as e:
            logger.warning("Skip sample %d: %s", i, e)

    logger.info("Got %d calibration samples", len(calib_data))

    # --- 3. attach recorder ---
    recorder = ExpertFrequencyRecorder(
        args.model_type,
        detail=args.detail,
        judge_layer=args.judge_layer,
        dominant_ratio=args.dominant_ratio,
        attn_mode=args.attn_mode,
        use_text_rater=args.text_rater,
    )
    recorder.setup(model, tokenizer=tokenizer)

    if not recorder.layer_names:
        logger.error("No MoE layers found – nothing to record.  Exiting.")
        return

    # --- 4. run prefill on each sample ---
    is_adaptive = (args.attn_mode == "adaptive")
    logger.info("Running prefill on %d samples ...  adaptive=%s", len(calib_data), is_adaptive)
    device = next(model.parameters()).device

    from tqdm import tqdm
    for sample in tqdm(calib_data, desc="Prefill", unit="sample"):
        inputs = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in sample.items()}
        with torch.no_grad():
            model(**inputs)
        if is_adaptive:
            recorder.accumulate_adaptive_stats()

    # --- 5. save results ---
    results = recorder.get_results()

    if not args.debug:
        with open(save_path, "w") as fp:
            json.dump(results, fp, indent=2)
        logger.info("Saved  %s", save_path)

        # adaptive mode: save per-layer token stats as separate file
        if is_adaptive:
            stats = recorder.get_adaptive_token_stats()
            stats_path = save_path.replace(".json", "_token_stats.json")
            with open(stats_path, "w") as fp:
                json.dump(stats, fp, indent=2)
            logger.info("Saved adaptive token stats  %s", stats_path)
    else:
        logger.info("*** DEBUG MODE: results NOT saved ***")
        if is_adaptive:
            stats = recorder.get_adaptive_token_stats()
            logger.info("Adaptive token stats (not saved):")
            for item in stats.get("consecutive_jaccard", []):
                logger.info("  layer %d ↔ %d  jaccard=%.4f",
                            item["layer_a"], item["layer_b"], item["jaccard"])

    recorder.remove_hooks()
    logger.info("COCO recording done. (%s)", fname)


# ====================================================================
# gradient verification
# ====================================================================
def run_grad_verification(args):
    """Compare gradient norms of dominant / redundant / text / image tokens.

    Uses adaptive mode to determine per-layer dominant vision tokens, then
    computes the next-token prediction loss and backpropagates to obtain
    per-position gradient norms of hidden_states at each decoder layer.
    """
    import torch.nn.functional as F
    import numpy as np
    from tqdm import tqdm

    _project_root = str(Path(__file__).resolve().parents[2])
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from mllm_quant.calibration.multimodal_calib import (
        load_image, process_calibration_item,
    )

    model_name = Path(args.model_path).name
    n_samples = args.n_samples

    # --- output file name ---
    ratio_str = f"{args.dominant_ratio:.2f}".rstrip("0").rstrip(".")
    parts = [model_name, f"coco{n_samples}", "grad", "adaptive", f"r{ratio_str}"]
    if args.human_only:
        parts.append("honly")
    fname = "_".join(parts) + ".json"

    coco_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coco")
    save_path = os.path.join(coco_dir, fname)
    os.makedirs(coco_dir, exist_ok=True)

    # --- 1. load model ---
    logger.info("Loading model for gradient verification ...")
    model, processor, tokenizer = load_model_and_processor(
        args.model_path, args.model_type, use_eager_attn=True,
    )

    # --- 2. load & process calibration data ---
    logger.info("Loading COCO calibration data: %s  n_samples=%d", args.calib_data, n_samples)
    with open(args.calib_data, "r") as f:
        dataset = json.load(f)
    rng = np.random.default_rng(seed=42)
    rng.shuffle(dataset)

    calib_data = []
    for i in tqdm(range(n_samples), desc="Processing calibration data"):
        item = dataset[i % len(dataset)]
        if args.human_only:
            item = dict(item)
            item["conversations"] = [
                c for c in item.get("conversations", []) if c.get("from") == "human"
            ]
        images = None
        img_path = item.get("image")
        if img_path:
            full_path = os.path.join(args.calib_img, img_path)
            if os.path.exists(full_path):
                images = [load_image(full_path)]
        try:
            d = process_calibration_item(
                images=images, data_item=item,
                processor=processor, tokenizer=tokenizer,
                max_seq_length=4096, model_type=args.model_type,
            )
            if d is not None:
                calib_data.append(d)
        except Exception as e:
            logger.warning("Skip sample %d: %s", i, e)

    logger.info("Got %d calibration samples", len(calib_data))

    # --- 3. setup recorder (adaptive mode, for dom/red classification) ---
    recorder = ExpertFrequencyRecorder(
        args.model_type, detail=True,
        dominant_ratio=args.dominant_ratio,
        attn_mode="adaptive",
        use_text_rater=args.text_rater,
    )
    recorder.setup(model, tokenizer=tokenizer)
    if not recorder.layer_names:
        logger.error("No MoE layers found – exiting.")
        return

    # --- 4. register gradient capture hooks on decoder layers ---
    decoder_layers = ExpertFrequencyRecorder._find_decoder_layers(model)
    if decoder_layers is None:
        logger.error("Cannot find decoder layers")
        return

    num_dec_layers = len(decoder_layers)
    hidden_storage: dict = {}  # layer_idx → Tensor (with grad retained)

    grad_hooks = []
    for idx in range(num_dec_layers):
        def _make_grad_hook(layer_idx):
            def _hook(_module, args, kwargs):
                hs = kwargs.get("hidden_states")
                if hs is None and args:
                    hs = args[0]
                if isinstance(hs, torch.Tensor) and hs.requires_grad:
                    hs.retain_grad()
                    hidden_storage[layer_idx] = hs
            return _hook
        h = decoder_layers[idx].register_forward_pre_hook(
            _make_grad_hook(idx), with_kwargs=True,
        )
        grad_hooks.append(h)

    # --- 5. per-sample gradient collection ---
    layer_dom_grads = defaultdict(list)
    layer_red_grads = defaultdict(list)
    layer_img_grads = defaultdict(list)
    layer_text_grads = defaultdict(list)

    device = next(model.parameters()).device

    logger.info("Running gradient verification on %d samples ...", len(calib_data))
    for si, sample in enumerate(tqdm(calib_data, desc="Grad verify")):
        hidden_storage.clear()
        recorder._adaptive_current_sample.clear()

        inputs = {
            k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
            for k, v in sample.items()
        }
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            continue

        # forward WITH gradients (no torch.no_grad)
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # next-token prediction loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous().to(shift_logits.device)
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        loss.backward()

        # read per-layer gradient norms
        img_mask = recorder.current_image_mask
        adaptive_doms = dict(recorder._adaptive_current_sample)

        for li in range(num_dec_layers):
            hs = hidden_storage.get(li)
            if hs is None or hs.grad is None:
                continue

            gn = hs.grad[0].float().norm(dim=-1).cpu()  # (seq_len,)
            seq_len = gn.shape[0]

            if img_mask is None or img_mask.shape[0] != seq_len:
                continue

            im = img_mask[:seq_len].cpu()
            tm = ~im

            dom_pos = adaptive_doms.get(li, frozenset())
            dm = torch.zeros(seq_len, dtype=torch.bool)
            for p in dom_pos:
                if p < seq_len:
                    dm[p] = True
            rm = im & (~dm)

            if dm.any():
                layer_dom_grads[li].append(gn[dm].mean().item())
            if rm.any():
                layer_red_grads[li].append(gn[rm].mean().item())
            if im.any():
                layer_img_grads[li].append(gn[im].mean().item())
            if tm.any():
                layer_text_grads[li].append(gn[tm].mean().item())

        # cleanup to avoid OOM
        model.zero_grad()
        for hs_val in hidden_storage.values():
            if isinstance(hs_val, torch.Tensor) and hs_val.grad is not None:
                hs_val.grad = None
        hidden_storage.clear()
        del outputs, logits, loss
        torch.cuda.empty_cache()

    # --- 6. aggregate ---
    per_layer = {}
    all_dom, all_red, all_img, all_text = [], [], [], []

    for li in range(num_dec_layers):
        d = float(np.mean(layer_dom_grads[li])) if layer_dom_grads[li] else 0.0
        r = float(np.mean(layer_red_grads[li])) if layer_red_grads[li] else 0.0
        i = float(np.mean(layer_img_grads[li])) if layer_img_grads[li] else 0.0
        t = float(np.mean(layer_text_grads[li])) if layer_text_grads[li] else 0.0

        per_layer[str(li)] = {
            "dom_grad_norm": round(d, 8),
            "red_grad_norm": round(r, 8),
            "img_grad_norm": round(i, 8),
            "text_grad_norm": round(t, 8),
            "dom_over_red": round(d / max(r, 1e-12), 4),
            "text_over_img": round(t / max(i, 1e-12), 4),
        }
        all_dom.append(d)
        all_red.append(r)
        all_img.append(i)
        all_text.append(t)

    avg_dom = float(np.mean(all_dom)) if all_dom else 0.0
    avg_red = float(np.mean(all_red)) if all_red else 0.0
    avg_img = float(np.mean(all_img)) if all_img else 0.0
    avg_text = float(np.mean(all_text)) if all_text else 0.0

    results = {
        "config": {
            "n_samples": len(calib_data),
            "dominant_ratio": args.dominant_ratio,
            "attn_mode": "adaptive",
            "text_rater": args.text_rater,
            "human_only": args.human_only,
            "num_decoder_layers": num_dec_layers,
        },
        "per_layer": per_layer,
        "global": {
            "avg_dom_grad": round(avg_dom, 8),
            "avg_red_grad": round(avg_red, 8),
            "avg_img_grad": round(avg_img, 8),
            "avg_text_grad": round(avg_text, 8),
            "avg_dom_over_red": round(avg_dom / max(avg_red, 1e-12), 4),
            "avg_text_over_img": round(avg_text / max(avg_img, 1e-12), 4),
        },
    }

    # --- 7. print summary ---
    logger.info("=" * 60)
    logger.info("Gradient Verification Summary (%d samples, r=%.2f):",
                len(calib_data), args.dominant_ratio)
    logger.info("  avg dom grad norm:  %.8f", avg_dom)
    logger.info("  avg red grad norm:  %.8f", avg_red)
    logger.info("  avg img grad norm:  %.8f", avg_img)
    logger.info("  avg text grad norm: %.8f", avg_text)
    logger.info("  dom / red  ratio:   %.4f", results["global"]["avg_dom_over_red"])
    logger.info("  text / img ratio:   %.4f", results["global"]["avg_text_over_img"])
    logger.info("=" * 60)

    # per-layer table (every 4th layer for readability)
    logger.info("Per-layer dom/red/text/img grad norms:")
    logger.info("  %4s  %12s  %12s  %12s  %12s  %8s  %8s",
                "Lyr", "Dom", "Red", "Img", "Text", "D/R", "T/I")
    for li in range(num_dec_layers):
        p = per_layer[str(li)]
        logger.info("  %4d  %12.8f  %12.8f  %12.8f  %12.8f  %8.3f  %8.3f",
                     li, p["dom_grad_norm"], p["red_grad_norm"],
                     p["img_grad_norm"], p["text_grad_norm"],
                     p["dom_over_red"], p["text_over_img"])

    # --- 8. save ---
    with open(save_path, "w") as fp:
        json.dump(results, fp, indent=2)
    logger.info("Saved gradient verification results: %s", save_path)

    # cleanup
    for h in grad_hooks:
        h.remove()
    recorder.remove_hooks()
    logger.info("Gradient verification done.")


# ====================================================================
# Pruning verification  (zero MoE output for dom / red vision tokens)
# ====================================================================
def _make_prune_hook(recorder, prune_target: str):
    """Forward hook that zeros the MoE block output for selected vision tokens.

    Args:
        recorder: ExpertFrequencyRecorder with adaptive hooks active –
            provides ``current_image_mask`` and ``current_dominant_mask``.
        prune_target: ``"dom"`` to prune dominant tokens,
            ``"red"`` to prune redundant tokens.
    """
    def _hook(module, inp, output):
        if prune_target == "dom":
            mask = recorder.current_dominant_mask
        else:
            mask = recorder.current_redundant_mask

        if mask is None:
            return

        prune = mask.cpu()
        if not prune.any():
            return

        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output

        if hs.dim() == 3:
            seq_len = hs.shape[1]
        elif hs.dim() == 2:
            seq_len = hs.shape[0]
        else:
            return

        if prune.shape[0] != seq_len:
            return

        pm = prune.to(hs.device)
        hs = hs.clone()
        if hs.dim() == 3:
            hs[:, pm, :] = 0.0
        else:
            hs[pm, :] = 0.0

        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    return _hook


def run_prune_verification(args):
    """Run lmms-eval with per-layer vision token pruning to verify importance.

    Uses adaptive mode to identify dominant / redundant vision tokens at each
    MoE layer, then zeros the MoE output for the chosen group.  By comparing
    the accuracy drop when pruning *dominant* vs *redundant* tokens we can
    verify that the dominant tokens are indeed more important.

    Results (per-task accuracy metrics) are saved under ``--prune_verify_dir``
    if set, otherwise ``moe_freq/accuracy_verify/``.
    """
    model_name = Path(args.model_path).name
    ratio_str = f"{args.dominant_ratio:.2f}".rstrip("0").rstrip(".")
    prune_target = args.prune_target

    if args.prune_verify_dir:
        verify_dir = os.path.abspath(args.prune_verify_dir)
    else:
        verify_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "accuracy_verify",
        )
    os.makedirs(verify_dir, exist_ok=True)

    parts = [model_name, f"prune_{prune_target}", f"r{ratio_str}"]
    if args.text_rater:
        parts.append("rater")
    fname = "_".join(parts) + ".json"
    save_path = os.path.join(verify_dir, fname)

    is_baseline = (prune_target == "none")

    # --- 1. create lmms-eval model ---------------------------------------
    lm = create_lmms_model(
        args.model_path, args.model_type, args.batch_size,
        use_eager_attn=(not is_baseline),
    )
    lm.clean = lambda: None
    underlying_model = lm.model

    tokenizer = (getattr(lm, "_tokenizer", None)
                 or getattr(lm, "tokenizer", None))

    # --- 2. setup recorder & prune hooks (skip for baseline) -------------
    recorder = None
    prune_hooks = []

    if not is_baseline:
        recorder = ExpertFrequencyRecorder(
            args.model_type, detail=True,
            dominant_ratio=args.dominant_ratio,
            attn_mode="adaptive",
            use_text_rater=args.text_rater,
        )
        recorder.setup(underlying_model, tokenizer=tokenizer)

        if not recorder.layer_names:
            logger.error("No MoE layers found – exiting.")
            return

        # --- 3. register prune hooks on MoE layers -----------------------
        moe_layers = recorder._find_moe_layers(underlying_model)
        for _name, module, _ne, _tk in moe_layers:
            h = module.register_forward_hook(
                _make_prune_hook(recorder, prune_target),
            )
            prune_hooks.append(h)

        logger.info(
            "Registered %d prune hooks (target=%s, ratio=%.2f)",
            len(prune_hooks), prune_target, args.dominant_ratio,
        )
    else:
        logger.info("Baseline mode (no pruning) – measuring original accuracy")

    # --- 4. run evaluation tasks -----------------------------------------
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    limit = None if args.limit < 0 else args.limit

    task_accuracies = {}

    for task in tasks:
        logger.info("=" * 70)
        logger.info("Prune-verify task: %s  (target=%s, r=%s)",
                     task, prune_target, ratio_str)
        if recorder is not None:
            recorder.reset()

        eval_results = None
        try:
            eval_results = run_single_task(lm, task, args.batch_size, limit=limit)
        except Exception:
            logger.exception("Evaluation failed for task %s", task)

        metrics = {}
        if eval_results and "results" in eval_results:
            task_metrics = eval_results["results"].get(task, {})
            metrics = {
                k: v for k, v in task_metrics.items()
                if not k.startswith("alias") and "stderr" not in k
                and "submission" not in k and v != "N/A"
            }
        task_accuracies[task] = metrics
        print_task_metrics(eval_results, task)

    # --- 5. save results -------------------------------------------------
    results = {
        "config": {
            "model_path": args.model_path,
            "model_type": args.model_type,
            "prune_target": prune_target,
            "dominant_ratio": args.dominant_ratio,
            "attn_mode": "adaptive",
            "text_rater": args.text_rater,
            "tasks": tasks,
            "limit": limit,
            "batch_size": args.batch_size,
        },
        "task_results": task_accuracies,
    }

    with open(save_path, "w") as fp:
        json.dump(results, fp, indent=2)
    logger.info("Saved prune verification results: %s", save_path)

    # --- cleanup ---------------------------------------------------------
    for h in prune_hooks:
        h.remove()
    if recorder is not None:
        recorder.remove_hooks()
    logger.info("Prune verification done.")


# ====================================================================
# main
# ====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Record MoE expert selection frequencies across eval datasets.",
    )
    # -- summary mode (no model loading / evaluation) ---------------------
    parser.add_argument(
        "--summary", metavar="DIR", default=None,
        help="Summary mode: read all per-task JSONs from DIR, "
             "aggregate and save aggregated.json.  "
             "Other arguments are ignored when this is set.",
    )
    # -- record mode arguments -------------------------------------------
    parser.add_argument(
        "--model_path", default=None,
        help="Path to the pretrained model directory (required for record mode).",
    )
    parser.add_argument(
        "--model_type", default=None,
        choices=["qwen3_vl_moe", "kimi_vl", "internvl"],
        help="Model type identifier (required for record mode).",
    )
    parser.add_argument(
        "--tasks", default=",".join(ALL_TASKS),
        help="Comma-separated evaluation tasks (default: all 9 benchmarks).",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Directory to save JSON results.  Default: <this_dir>/<model_name>.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for lmms-eval (default 1, recommended for HF).",
    )
    parser.add_argument(
        "--limit", type=int, default=-1,
        help="Max samples per task (-1 = no limit, use full dataset).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode: record freq & print accuracy, but do NOT save results to disk.",
    )
    parser.add_argument(
        "--no_aggregate", action="store_true",
        help="Skip computing and saving the aggregated result across tasks.",
    )
    # -- detail mode: dominant / redundant vision token classification ------
    parser.add_argument(
        "--detail", action="store_true",
        help="Enable dominant/redundant vision-token classification via LLM attention.",
    )
    parser.add_argument(
        "--judge_layer", type=int, default=2,
        help="0-indexed decoder layer whose attention is used for dominant-token detection (default: 2).",
    )
    parser.add_argument(
        "--dominant_ratio", type=float, default=1.0 / 3,
        help="Fraction of image tokens classified as dominant (default: 1/3).",
    )
    parser.add_argument(
        "--attn_mode", type=str, default="vision",
        choices=["vision", "left", "adaptive"],
        help="How to compute image-token importance: "
             "'vision' = text→img attention at judge_layer; "
             "'left' = all-other→img attention at judge_layer; "
             "'adaptive' = text→img attention per-layer (default: vision).",
    )
    parser.add_argument(
        "--text_rater", action="store_true",
        help="Enable SparseVLM-style Text Rater selection (only with attn_mode=vision). "
             "Select visually-relevant text tokens via embedding similarity before "
             "computing importance.",
    )
    # -- COCO calibration mode ------------------------------------------------
    parser.add_argument(
        "--coco", action="store_true",
        help="COCO calibration mode: record expert freq on COCO calib data (prefill only).",
    )
    parser.add_argument(
        "--calib_data", type=str,
        default="data/calibration/calib_coco_512.json",
        help="Path to COCO calibration JSON.",
    )
    parser.add_argument(
        "--calib_img", type=str,
        default="data",
        help="Root folder for calibration images.",
    )
    parser.add_argument(
        "--n_samples", type=int, default=512,
        help="Number of COCO calibration samples (default: 512).",
    )
    parser.add_argument(
        "--human_only", action="store_true",
        help="COCO mode: only include the human question in text (drop assistant answer). "
             "More realistic for eval-like prefill and cleaner Text Rater selection.",
    )
    # -- gradient verification mode ----------------------------------------
    parser.add_argument(
        "--grad_verify", action="store_true",
        help="Gradient verification mode (requires --coco). "
             "Compare gradient norms of dominant vs redundant vs text vs image tokens "
             "per layer. No expert counting is performed.",
    )
    # -- pruning verification mode -----------------------------------------
    parser.add_argument(
        "--prune_verify", action="store_true",
        help="Pruning verification mode: zero MoE output for dom/red vision "
             "tokens per layer (adaptive) and measure accuracy via lmms-eval. "
             "JSON results: see --prune_verify_dir (default moe_freq/accuracy_verify).",
    )
    parser.add_argument(
        "--prune_verify_dir", type=str, default=None,
        help="Directory for --prune_verify JSON outputs "
             "(default: <this_dir>/accuracy_verify).",
    )
    parser.add_argument(
        "--prune_target", type=str, default="dom",
        choices=["dom", "red", "none"],
        help="Which vision tokens to prune: "
             "'dom' (dominant), 'red' (redundant), or "
             "'none' (no pruning, baseline) (default: dom).",
    )
    args = parser.parse_args()

    # ---- summary mode: aggregate existing JSONs and exit ----------------
    if args.summary is not None:
        run_summary(args.summary)
        return

    # ---- pruning verification mode ---------------------------------------
    if args.prune_verify:
        if args.model_path is None or args.model_type is None:
            parser.error("--model_path and --model_type are required for --prune_verify mode.")
        run_prune_verification(args)
        return

    # ---- COCO calibration mode -------------------------------------------
    if args.coco:
        if args.model_path is None or args.model_type is None:
            parser.error("--model_path and --model_type are required for --coco mode.")
        if args.grad_verify:
            run_grad_verification(args)
        else:
            run_coco_recording(args)
        return

    # ---- record mode: validate required arguments -----------------------
    if args.model_path is None or args.model_type is None:
        parser.error("--model_path and --model_type are required for record mode.")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    limit = None if args.limit < 0 else args.limit

    # output directory (not needed in debug mode)
    if args.output_dir is None:
        model_name = Path(args.model_path).name
        if args.detail:
            ratio_str = f"{args.dominant_ratio:.2f}".rstrip("0").rstrip(".")
            suffix = f"-{args.attn_mode}-r{ratio_str}"
            if args.text_rater:
                suffix += "-rater"
        else:
            suffix = ""
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), model_name + suffix,
        )
    if not args.debug:
        os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. create model ------------------------------------------------
    lm = create_lmms_model(
        args.model_path, args.model_type, args.batch_size,
        use_eager_attn=args.detail,
    )

    # IMPORTANT: lmms-eval's evaluate() calls lm.clean() after each task,
    # which deletes all nn.Module attributes (including _model) to free
    # GPU memory for LLM-as-judge.  Since we reuse the model across tasks,
    # override clean() with a no-op to prevent the model from being destroyed.
    lm.clean = lambda: None

    # lmms-eval wraps the HF model; unwrap to get the raw nn.Module
    underlying_model = lm.model  # property; unwraps Accelerate if needed

    # ---- 2. attach recorder ---------------------------------------------
    # Pass tokenizer so _detect_image_token_id can resolve tokens like
    # <IMG_CONTEXT> (InternVL) when the config has no image_token_id field.
    tokenizer = getattr(lm, "_tokenizer", None) or getattr(lm, "tokenizer", None)
    recorder = ExpertFrequencyRecorder(
        args.model_type,
        detail=args.detail,
        judge_layer=args.judge_layer,
        dominant_ratio=args.dominant_ratio,
        attn_mode=args.attn_mode,
        use_text_rater=args.text_rater,
    )
    recorder.setup(underlying_model, tokenizer=tokenizer)

    if not recorder.layer_names:
        logger.error("No MoE layers found – nothing to record.  Exiting.")
        return

    # ---- 3. per-task evaluation -----------------------------------------
    aggregated: OrderedDict = OrderedDict()

    if args.debug:
        logger.info("*** DEBUG MODE: results will NOT be saved to disk ***")

    for task in tasks:
        logger.info("=" * 70)
        logger.info("Task: %s", task)
        recorder.reset()

        eval_results = None
        try:
            eval_results = run_single_task(lm, task, args.batch_size, limit=limit)
        except Exception:
            logger.exception("Evaluation failed for task %s (will still save any collected freq data).", task)

        print_task_metrics(eval_results, task)

        task_results = recorder.get_results()

        if not args.debug:
            save_path = os.path.join(args.output_dir, f"{task}.json")
            with open(save_path, "w") as fp:
                json.dump(task_results, fp, indent=2)
            logger.info("Saved  %s", save_path)

        if not args.no_aggregate:
            aggregate_results(aggregated, task_results)

    # ---- 4. aggregated result -------------------------------------------
    if aggregated and not args.debug and not args.no_aggregate:
        save_path = os.path.join(args.output_dir, "aggregated.json")
        with open(save_path, "w") as fp:
            json.dump(aggregated, fp, indent=2)
        logger.info("Saved aggregated  %s", save_path)

    # ---- cleanup --------------------------------------------------------
    recorder.remove_hooks()
    logger.info("All done.")


if __name__ == "__main__":
    main()
