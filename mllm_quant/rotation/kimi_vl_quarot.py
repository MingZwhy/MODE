# QuaRot-style rotation for Kimi-VL: global Q on LLM + MoE + Hadamard-44 FFN + visual merge @ Q.
# Skips MLA head-wise V/O Hadamard and o_proj online partial Had (see discussion).

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn

from .hadamard_ext import ensure_had44_verified, get_hadk_kimi

logger = logging.getLogger(__name__)

_FAKE_QUANT = Path(__file__).resolve().parents[2] / "3rdparty" / "QuaRot" / "fake_quant"
if str(_FAKE_QUANT) not in sys.path:
    sys.path.insert(0, str(_FAKE_QUANT))
import hadamard_utils  # noqa: E402

from mllm_quant.rotation.qwen3_vl_moe_quarot import (  # noqa: E402
    RMSN,
    _apply_online_full_had,
    _bake_mean_into_embedding,
    _fuse_rmsnorm_into_linear,
    _get_orthogonal_matrix,
    _rotate_linear_input_side,
    _rotate_linear_output_side,
)


def ensure_kimi_quarot_decoder_rmsnorm_weights_are_ones(model: nn.Module) -> None:
    """
    After QuaRot fusion, decoder norms are RMSN (no learnable γ); checkpoints often omit
    ``*.layernorm.weight`` keys. HuggingFace may report these as "newly initialized" but
    leave GPU Parameters uninitialized (garbage), which blows up the first RMSNorm forward.

    Filling γ=1 matches RMSN / fused-linears math. Safe for any Kimi-VL load where norms
    were fused away (re-save with materialize_kimi_rmsn_as_deepseek_rmsnorm_for_save).
    """
    inner = model.language_model.model
    for layer in inner.layers:
        for name in ("input_layernorm", "post_attention_layernorm"):
            mod = getattr(layer, name, None)
            w = getattr(mod, "weight", None) if mod is not None else None
            if isinstance(w, nn.Parameter):
                w.data.fill_(1.0)
    final_norm = getattr(inner, "norm", None)
    wn = getattr(final_norm, "weight", None) if final_norm is not None else None
    if isinstance(wn, nn.Parameter):
        wn.data.fill_(1.0)


def materialize_kimi_rmsn_as_deepseek_rmsnorm_for_save(model: nn.Module) -> None:
    """
    Replace QuaRot ``RMSN`` (no state_dict entries) with ``DeepseekV3RMSNorm`` and γ=1 so
    ``save_pretrained`` writes ``language_model.model.layers.*.layernorm.weight`` tensors.
    Forward is identical to RMSN when γ is all-ones.
    """
    from mllm_quant.models.kimi_vl.modeling_kimi_vl import DeepseekV3RMSNorm

    inner = model.language_model.model
    ref = next(inner.layers[0].parameters())
    dev, dt = ref.device, ref.dtype

    def replace_rmsn(mod: nn.Module) -> nn.Module:
        if not isinstance(mod, RMSN):
            return mod
        new = DeepseekV3RMSNorm(mod.dim, eps=mod.variance_epsilon)
        return new.to(device=dev, dtype=dt)

    for layer in inner.layers:
        layer.input_layernorm = replace_rmsn(layer.input_layernorm)
        layer.post_attention_layernorm = replace_rmsn(layer.post_attention_layernorm)
    inner.norm = replace_rmsn(inner.norm)


def _fuse_rmsnorm_into_moe_gate(layernorm: nn.Module, gate_module: nn.Module) -> None:
    """MoEGate.weight [E, H]; F.linear uses h @ w.T — fuse gamma on hidden columns."""
    gamma = layernorm.weight.data
    w = gate_module.weight.data
    w.mul_(gamma.to(dtype=w.dtype, device=w.device).unsqueeze(0))


def _apply_exact_had_kimi_linear(linear: nn.Linear, work_device: torch.device) -> None:
    """Fuse Walsh×H_44 into down_proj input dim (Kimi: 11264, 1408, 2816)."""
    w = linear.weight.data
    dtype, dev = w.dtype, w.device
    w_ = w.float().to(work_device)
    had_k, k = get_hadk_kimi(linear.in_features)
    w_ = hadamard_utils.matmul_hadU_cuda(w_, had_k.to(w_.device).to(w_.dtype), k)
    linear.weight.data = w_.to(device=dev, dtype=dtype)


def _patch_deepseek_mlp_online_had(mlp: nn.Module, fp32_had: bool) -> None:
    if getattr(mlp, "_quarot_mlp_had_patched", False):
        return
    inter = mlp.intermediate_size
    had_k, k = get_hadk_kimi(inter)
    mlp._quarot_fp32_had = fp32_had

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mid = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        mid = _apply_online_full_had(mid, had_k, k, fp32_had)
        return self.down_proj(mid)

    mlp.forward = types.MethodType(forward, mlp)
    mlp._quarot_mlp_had_patched = True


def _fuse_layer_norms_kimi_text(inner: nn.Module, lm_head: nn.Linear, bake_mean: bool = True) -> None:
    from mllm_quant.models.kimi_vl.modeling_kimi_vl import DeepseekV3DecoderLayer, DeepseekV3MLP, DeepseekV3MoE

    if bake_mean:
        _bake_mean_into_embedding(inner.embed_tokens)

    for layer in inner.layers:
        assert isinstance(layer, DeepseekV3DecoderLayer)
        attn = layer.self_attn
        iln = layer.input_layernorm
        if getattr(attn, "q_lora_rank", None) is None:
            _fuse_rmsnorm_into_linear(iln, attn.q_proj)
        else:
            _fuse_rmsnorm_into_linear(iln, attn.q_a_proj)
        _fuse_rmsnorm_into_linear(iln, attn.kv_a_proj_with_mqa)
        eps = iln.variance_epsilon
        layer.input_layernorm = RMSN(layer.hidden_size, eps=eps)

        mlp = layer.mlp
        paln = layer.post_attention_layernorm
        if isinstance(mlp, DeepseekV3MLP):
            _fuse_rmsnorm_into_linear(paln, mlp.gate_proj)
            _fuse_rmsnorm_into_linear(paln, mlp.up_proj)
        elif isinstance(mlp, DeepseekV3MoE):
            _fuse_rmsnorm_into_moe_gate(paln, mlp.gate)
            for ex in mlp.experts:
                if ex is not None:
                    _fuse_rmsnorm_into_linear(paln, ex.gate_proj)
                    _fuse_rmsnorm_into_linear(paln, ex.up_proj)
            # shared_experts sees the same post_attention_layernorm output as the gate (see MoE.forward).
            if getattr(mlp, "shared_experts", None) is not None:
                se = mlp.shared_experts
                _fuse_rmsnorm_into_linear(paln, se.gate_proj)
                _fuse_rmsnorm_into_linear(paln, se.up_proj)
        else:
            raise TypeError(type(mlp))
        layer.post_attention_layernorm = RMSN(layer.hidden_size, eps=paln.variance_epsilon)

    _fuse_rmsnorm_into_linear(inner.norm, lm_head)
    inner.norm = RMSN(inner.config.hidden_size, eps=inner.norm.variance_epsilon)


def _rotate_attention_global_q(attn: nn.Module, q_dev: torch.Tensor) -> None:
    if getattr(attn, "q_lora_rank", None) is None:
        attn.q_proj.weight.data = _rotate_linear_input_side(attn.q_proj.weight.data, q_dev)
    else:
        attn.q_a_proj.weight.data = _rotate_linear_input_side(attn.q_a_proj.weight.data, q_dev)
    attn.kv_a_proj_with_mqa.weight.data = _rotate_linear_input_side(attn.kv_a_proj_with_mqa.weight.data, q_dev)
    attn.o_proj.weight.data = _rotate_linear_output_side(attn.o_proj.weight.data, q_dev)
    if attn.o_proj.bias is not None:
        b = attn.o_proj.bias.data
        attn.o_proj.bias.data = torch.matmul(q_dev.T.double(), b.double().to(q_dev.device)).to(
            dtype=b.dtype, device=b.device
        )


def _rotate_dense_mlp_kimi(
    mlp: nn.Module, q_dev: torch.Tensor, work_dev: torch.device, fp32_had: bool, ffn_had: bool
) -> None:
    mlp.gate_proj.weight.data = _rotate_linear_input_side(mlp.gate_proj.weight.data, q_dev)
    mlp.up_proj.weight.data = _rotate_linear_input_side(mlp.up_proj.weight.data, q_dev)
    mlp.down_proj.weight.data = _rotate_linear_output_side(mlp.down_proj.weight.data, q_dev)
    if ffn_had:
        _apply_exact_had_kimi_linear(mlp.down_proj, work_device=work_dev)
        _patch_deepseek_mlp_online_had(mlp, fp32_had)


def _rotate_moe_kimi(
    moe: nn.Module, q_dev: torch.Tensor, work_dev: torch.device, fp32_had: bool, ffn_had: bool
) -> None:
    moe.gate.weight.data = _rotate_linear_input_side(moe.gate.weight.data, q_dev)
    for ex in moe.experts:
        if ex is None:
            continue
        ex.gate_proj.weight.data = _rotate_linear_input_side(ex.gate_proj.weight.data, q_dev)
        ex.up_proj.weight.data = _rotate_linear_input_side(ex.up_proj.weight.data, q_dev)
        ex.down_proj.weight.data = _rotate_linear_output_side(ex.down_proj.weight.data, q_dev)
        if ffn_had:
            _apply_exact_had_kimi_linear(ex.down_proj, work_device=work_dev)
            _patch_deepseek_mlp_online_had(ex, fp32_had)
    if getattr(moe, "shared_experts", None) is not None:
        _rotate_dense_mlp_kimi(moe.shared_experts, q_dev, work_dev, fp32_had, ffn_had)


def _patch_kimi_merge_image_features(kimi: nn.Module) -> None:
    if getattr(kimi, "_quarot_kimi_merge_patched", False):
        return
    orig = kimi._merge_with_image_features

    def _merge_with_image_features(self, inputs_embeds, input_ids, image_features):
        qm = getattr(self, "_quarot_visual_Q", None)
        if qm is not None:
            qm = qm.to(device=image_features.device, dtype=torch.float32)
            image_features = (image_features.float() @ qm).to(image_features.dtype)
        return orig(inputs_embeds, input_ids, image_features)

    kimi._merge_with_image_features = types.MethodType(_merge_with_image_features, kimi)
    kimi._quarot_kimi_merge_patched = True


@torch.inference_mode()
def apply_quarot_rotation_kimi_vl(
    model: nn.Module,
    rotate_mode: Literal["hadamard", "random"] = "hadamard",
    fp32_had: bool = False,
    work_device: Optional[torch.device] = None,
    verify_h44: bool = True,
    ffn_had: bool = True,
    bake_mean: bool = True,
) -> torch.Tensor:
    """
    KimiVLForConditionalGeneration: rotate DeepSeek text LLM with global Q; optional H_44 on FFN downs;
    right-multiply projected image features by Q before merge. ViT not rotated.
    """
    from mllm_quant.models.kimi_vl.modeling_kimi_vl import DeepseekV3DecoderLayer, DeepseekV3MLP, DeepseekV3MoE

    lm = model.language_model
    if getattr(lm.model, "_quarot_applied", False):
        logger.warning("QuaRot already applied to Kimi inner model; skipping.")
        return getattr(lm.model, "_quarot_Q_cpu", None)

    if verify_h44:
        ensure_had44_verified()

    inner = lm.model
    lm_head = lm.lm_head
    cfg = inner.config
    hidden = cfg.hidden_size

    dev = work_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = _get_orthogonal_matrix(hidden, rotate_mode, dev).cpu()

    if getattr(inner.layers[0].self_attn, "q_lora_rank", None) is not None:
        logger.warning(
            "QuaRot Kimi: q_lora_rank path is only partially tested (q_a fused; q_b/kv_b unchanged)."
        )

    logger.info("QuaRot Kimi: fusing RMSNorm into LLM linears + MoE gate/experts...")
    _fuse_layer_norms_kimi_text(inner, lm_head, bake_mean=bake_mean)

    logger.info("QuaRot Kimi: rotating embeddings + lm_head...")
    emb = inner.embed_tokens.weight.data
    inner.embed_tokens.weight.data = torch.matmul(emb.to(torch.float64).to(q.device), q).to(
        dtype=emb.dtype, device=emb.device
    )
    lh = lm_head.weight.data
    if lh.data_ptr() != inner.embed_tokens.weight.data.data_ptr():
        lm_head.weight.data = torch.matmul(lh.to(torch.float64).to(q.device), q).to(
            dtype=lh.dtype, device=lh.device
        )

    if ffn_had:
        logger.info("QuaRot Kimi: rotating decoder (global Q + H44 FFN; no MLA V/O Hadamard)...")
    else:
        logger.info("QuaRot Kimi: rotating decoder (global Q only; FFN Hadamard skipped)...")
    for layer in inner.layers:
        assert isinstance(layer, DeepseekV3DecoderLayer)
        attn = layer.self_attn
        wdev = attn.q_proj.weight.device if attn.q_lora_rank is None else attn.q_a_proj.weight.device
        q_dev = q.to(device=wdev)
        _rotate_attention_global_q(attn, q_dev)

        mlp = layer.mlp
        if isinstance(mlp, DeepseekV3MLP):
            _rotate_dense_mlp_kimi(mlp, q_dev, wdev, fp32_had, ffn_had)
        elif isinstance(mlp, DeepseekV3MoE):
            _rotate_moe_kimi(mlp, q_dev, wdev, fp32_had, ffn_had)
        else:
            raise TypeError(type(mlp))

    inner._quarot_Q_cpu = q.cpu()
    inner._quarot_applied = True

    q_vis = q.to(dtype=torch.float32)
    model._quarot_visual_Q = q_vis
    _patch_kimi_merge_image_features(model)

    logger.info("QuaRot Kimi: done (merge_with_image_features applies visual @ Q).")
    return inner._quarot_Q_cpu


def apply_quarot_inference_hooks_kimi_vl(
    model: nn.Module,
    Q_cpu: torch.Tensor,
    *,
    fp32_had: bool = False,
    ffn_had: bool = True,
    verify_h44: bool = True,
) -> None:
    """
    Reload online FFN Hadamard-44 + visual merge @ Q for a Kimi-VL checkpoint that was already
    rotated and saved (weights in QuaRot basis). Does not rotate weights again.
    """
    from mllm_quant.models.kimi_vl.modeling_kimi_vl import DeepseekV3DecoderLayer, DeepseekV3MLP, DeepseekV3MoE

    lm = model.language_model
    inner = lm.model
    if getattr(inner, "_quarot_inference_hooks_installed", False):
        logger.warning("QuaRot Kimi: inference hooks already installed; skipping.")
        return

    ensure_kimi_quarot_decoder_rmsnorm_weights_are_ones(model)

    if verify_h44 and ffn_had:
        ensure_had44_verified()

    for layer in inner.layers:
        assert isinstance(layer, DeepseekV3DecoderLayer)
        mlp = layer.mlp
        if isinstance(mlp, DeepseekV3MLP):
            if ffn_had:
                _patch_deepseek_mlp_online_had(mlp, fp32_had)
        elif isinstance(mlp, DeepseekV3MoE):
            if ffn_had:
                for ex in mlp.experts:
                    if ex is not None:
                        _patch_deepseek_mlp_online_had(ex, fp32_had)
                if getattr(mlp, "shared_experts", None) is not None:
                    _patch_deepseek_mlp_online_had(mlp.shared_experts, fp32_had)
        else:
            raise TypeError(type(mlp))

    inner._quarot_Q_cpu = Q_cpu.detach().cpu()
    inner._quarot_applied = True
    inner._quarot_inference_hooks_installed = True

    model._quarot_visual_Q = Q_cpu.to(dtype=torch.float32)
    _patch_kimi_merge_image_features(model)

    logger.info(
        "QuaRot Kimi: inference hooks installed (FFN online Had + merge @ Q), "
        f"fp32_had={fp32_had}, ffn_had={ffn_had}."
    )
