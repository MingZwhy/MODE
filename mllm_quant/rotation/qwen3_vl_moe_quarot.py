# QuaRot-style rotation for Qwen3-VL-MoE language model only (no ViT rotation).
# See QuaRot (NeurIPS 2024) / 3rdparty/QuaRot/fake_quant/rotation_utils.py

from __future__ import annotations

import logging
import math
import os
import sys
import types
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# QuaRot fake_quant (hadamard_utils + fast_hadamard_transform)
_FAKE_QUANT = Path(__file__).resolve().parents[2] / "3rdparty" / "QuaRot" / "fake_quant"
if str(_FAKE_QUANT) not in sys.path:
    sys.path.insert(0, str(_FAKE_QUANT))

import hadamard_utils  # noqa: E402
import fast_hadamard_transform  # noqa: E402


class RMSN(nn.Module):
    """RMSNorm without learnable scale (scale fused into following linear)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.variance_epsilon = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.variance_epsilon)
        return x.to(dtype)

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.variance_epsilon}"


def _fuse_rmsnorm_into_linear(layernorm: nn.Module, linear: nn.Linear) -> None:
    """Fuse RMSNorm gamma into Linear weight (QuaRot-style)."""
    if not hasattr(layernorm, "weight"):
        raise ValueError("layernorm must have weight (gamma)")
    gamma = layernorm.weight.data.double()
    w = linear.weight.data.double()
    linear.weight.data = (w * gamma.unsqueeze(0)).to(linear.weight.dtype)
    if linear.bias is not None and hasattr(layernorm, "bias") and layernorm.bias is not None:
        raise NotImplementedError("bias fusion not expected for Qwen3 MoE linears")


def _fuse_rmsnorm_into_gate_up_proj(layernorm: nn.Module, gate_up_proj: nn.Parameter) -> None:
    """Fuse post-attention RMSNorm into expert gate_up_proj [num_experts, hidden, 2*inter]."""
    gamma = layernorm.weight.data  # [hidden]
    g = gamma.to(dtype=gate_up_proj.dtype, device=gate_up_proj.device)
    gate_up_proj.data.mul_(g.view(1, -1, 1))


def _bake_mean_into_embedding(embed: nn.Embedding) -> None:
    """Zero-mean each embedding row (QuaRot fake_quant)."""
    w = embed.weight.data.double()
    embed.weight.data = (w - w.mean(dim=-1, keepdim=True)).to(embed.weight.dtype)


def _random_orthogonal_matrix(size: int, device: torch.device) -> torch.Tensor:
    torch.cuda.empty_cache() if device.type == "cuda" else None
    random_matrix = torch.randn(size, size, dtype=torch.float64, device=device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def _get_orthogonal_matrix(size: int, mode: str, device: torch.device) -> torch.Tensor:
    if mode == "random":
        return _random_orthogonal_matrix(size, device)
    if mode == "hadamard":
        return hadamard_utils.random_hadamard_matrix(size, device)
    raise ValueError(f"Unknown rotate_mode: {mode}")


def _rotate_linear_input_side(w: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """W is [out, hidden]; return W @ Q."""
    dtype, dev = w.dtype, w.device
    w64 = w.to(device=q.device, dtype=torch.float64)
    out = torch.matmul(w64, q).to(device=dev, dtype=dtype)
    return out


def _rotate_linear_output_side(w: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """W is [out, in] with hidden on 'out' for o_proj; return Q.T @ W."""
    dtype, dev = w.dtype, w.device
    w64 = w.to(device=q.device, dtype=torch.float64)
    out = torch.matmul(q.T, w64).to(device=dev, dtype=dtype)
    return out


def _apply_exact_had_to_linear_on_device(
    linear: nn.Linear, had_dim: int = -1, output: bool = False, out_device=None
) -> None:
    """Like QuaRot apply_exact_had_to_linear but uses linear.weight.device as compute device."""
    assert isinstance(linear, nn.Linear)
    in_features, out_features = linear.in_features, linear.out_features
    w = linear.weight.data
    dtype, dev = w.dtype, w.device
    work = out_device or dev
    w_ = w.float().to(work)

    if had_dim != -1:
        assert hadamard_utils.is_pow2(had_dim), "had_dim must be power of 2"

    if had_dim == -1:
        if output:
            had_k, k = hadamard_utils.get_hadK(out_features)
            w_ = hadamard_utils.matmul_hadU_cuda(w_.t(), had_k, k).t()
        if not output:
            had_k, k = hadamard_utils.get_hadK(in_features)
            w_ = hadamard_utils.matmul_hadU_cuda(w_, had_k, k)
    else:
        if output:
            w_ = w_.t()
            tshape = w_.shape
            w_ = fast_hadamard_transform.hadamard_transform(
                w_.reshape(-1, tshape[-1] // had_dim, had_dim),
                scale=1.0 / math.sqrt(had_dim),
            ).reshape(tshape)
            w_ = w_.t()
        else:
            raise NotImplementedError("had_dim!=-1 with output=False not used for Qwen3 ov")

    linear.weight.data = w_.to(device=dev, dtype=dtype)


def _online_full_had_chunk_rows() -> int:
    """Max flattened rows per chunk; 0 or negative disables chunking (legacy one-shot)."""
    v = os.environ.get("QUAROT_ONLINE_FULL_HAD_CHUNK")
    if v is not None:
        return int(v)
    return int(os.environ.get("QUAROT_MATMUL_HAD_CHUNK", "2048"))


def _apply_online_full_had(
    x: torch.Tensor, had_k: Optional[torch.Tensor], k: int, fp32: bool = False
) -> torch.Tensor:
    """
    Online full Hadamard on the last dim. When ``fp32`` is True, avoid ``x.float()`` on the
    whole tensor (doubles VRAM vs bf16); chunk along flattened batch rows instead — same math.

    Env: ``QUAROT_ONLINE_FULL_HAD_CHUNK`` (rows per chunk). If unset, uses
    ``QUAROT_MATMUL_HAD_CHUNK`` or default 2048. Use ``0`` for unchunked (original behavior).
    """
    dtype = x.dtype
    init_shape = x.shape
    n = x.shape[-1]
    flat = x.reshape(-1, n)
    b = flat.shape[0]
    chunk = _online_full_had_chunk_rows()

    work_dtype = torch.float32 if fp32 else dtype
    hk = had_k
    if hk is not None:
        hk = hk.to(device=x.device, dtype=work_dtype)

    def _run_rows(s: int, e: int) -> torch.Tensor:
        sl = flat[s:e]
        work = sl.float() if fp32 else sl
        return hadamard_utils.matmul_hadU_cuda(work, hk, k)

    if chunk <= 0 or b <= chunk:
        y = _run_rows(0, b)
        return y.reshape(init_shape).to(dtype)

    if fp32:
        out = torch.empty(b, n, device=x.device, dtype=torch.float32)
    else:
        out = torch.empty_like(flat)
    for s in range(0, b, chunk):
        e = min(s + chunk, b)
        out[s:e] = _run_rows(s, e)
    return out.reshape(init_shape).to(dtype)


def _apply_online_partial_had_o(
    x: torch.Tensor, num_heads: int, head_dim: int, had_k: Optional[torch.Tensor], k: int, fp32: bool
) -> torch.Tensor:
    """Match QuaRot ActQuantWrapper.online_partial_had for o_proj input."""
    dtype = x.dtype
    if fp32:
        x = x.float()
    init_shape = x.shape
    if k == 1:
        x = fast_hadamard_transform.hadamard_transform(
            x.reshape(-1, init_shape[-1] // head_dim, head_dim).transpose(1, 2),
            scale=1.0 / math.sqrt(init_shape[-1] // head_dim),
        ).transpose(1, 2)
    else:
        x = (had_k.to(dtype=x.dtype, device=x.device) @ x.reshape(-1, init_shape[-1] // head_dim, head_dim)) / math.sqrt(
            init_shape[-1] // head_dim
        )
    x = x.reshape(init_shape)
    return x.to(dtype)


def _fuse_layer_norms_qwen3_vl_moe_text(lm: nn.Module, lm_head: nn.Linear) -> None:
    from mllm_quant.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        Qwen3VLMoeTextDecoderLayer,
        Qwen3VLMoeTextMLP,
        Qwen3VLMoeTextSparseMoeBlock,
    )

    _bake_mean_into_embedding(lm.embed_tokens)

    for layer in lm.layers:
        assert isinstance(layer, Qwen3VLMoeTextDecoderLayer)
        attn = layer.self_attn
        _fuse_rmsnorm_into_linear(layer.input_layernorm, attn.q_proj)
        _fuse_rmsnorm_into_linear(layer.input_layernorm, attn.k_proj)
        _fuse_rmsnorm_into_linear(layer.input_layernorm, attn.v_proj)
        layer.input_layernorm = RMSN(layer.hidden_size, eps=layer.input_layernorm.variance_epsilon)

        mlp = layer.mlp
        if isinstance(mlp, Qwen3VLMoeTextSparseMoeBlock):
            _fuse_rmsnorm_into_linear(layer.post_attention_layernorm, mlp.gate)
            _fuse_rmsnorm_into_gate_up_proj(layer.post_attention_layernorm, mlp.experts.gate_up_proj)
        elif isinstance(mlp, Qwen3VLMoeTextMLP):
            _fuse_rmsnorm_into_linear(layer.post_attention_layernorm, mlp.gate_proj)
            _fuse_rmsnorm_into_linear(layer.post_attention_layernorm, mlp.up_proj)
        else:
            raise TypeError(type(mlp))
        layer.post_attention_layernorm = RMSN(layer.hidden_size, eps=layer.post_attention_layernorm.variance_epsilon)

    _fuse_rmsnorm_into_linear(lm.norm, lm_head)
    lm.norm = RMSN(lm.config.hidden_size, eps=lm.norm.variance_epsilon)


def _rotate_attention_layer(attn: nn.Module, q: torch.Tensor, num_heads: int, head_dim: int, work_device: torch.device):
    for mod in (attn.q_proj, attn.k_proj, attn.v_proj):
        mod.weight.data = _rotate_linear_input_side(mod.weight.data, q)
    attn.o_proj.weight.data = _rotate_linear_output_side(attn.o_proj.weight.data, q)
    if attn.o_proj.bias is not None:
        attn.o_proj.bias.data = torch.matmul(q.T.double(), attn.o_proj.bias.data.double().to(q.device)).to(
            dtype=attn.o_proj.bias.dtype, device=attn.o_proj.device
        )
    _apply_exact_had_to_linear_on_device(attn.v_proj, had_dim=head_dim, output=True, out_device=work_device)
    _apply_exact_had_to_linear_on_device(attn.o_proj, had_dim=-1, output=False, out_device=work_device)


def _rotate_dense_mlp(mlp: nn.Module, q: torch.Tensor, inter: int, work_device: torch.device):
    mlp.gate_proj.weight.data = _rotate_linear_input_side(mlp.gate_proj.weight.data, q)
    mlp.up_proj.weight.data = _rotate_linear_input_side(mlp.up_proj.weight.data, q)
    mlp.down_proj.weight.data = _rotate_linear_output_side(mlp.down_proj.weight.data, q)
    _apply_exact_had_to_linear_on_device(mlp.down_proj, had_dim=-1, output=False, out_device=work_device)


def _rotate_moe_block(moe: nn.Module, q: torch.Tensor, work_device: torch.device):
    moe.gate.weight.data = _rotate_linear_input_side(moe.gate.weight.data, q)
    gu = moe.experts.gate_up_proj.data  # [E, H, 2*inter]
    qt = q.T.to(dtype=torch.float64, device=gu.device)
    gu64 = gu.to(torch.float64)
    # For each expert: gate_up[e] <- Q.T @ gate_up[e] with shapes [H,H] @ [H, 2D]
    out = torch.einsum("ij,ejk->eik", qt, gu64).to(dtype=gu.dtype)
    moe.experts.gate_up_proj.data.copy_(out)
    down = moe.experts.down_proj.data
    e, inter, h = down.shape
    for ei in range(e):
        lin = nn.Linear(inter, h, bias=False)
        lin.weight.data = down[ei].t().clone()
        lin.weight.data = _rotate_linear_output_side(lin.weight.data, q)
        _apply_exact_had_to_linear_on_device(lin, had_dim=-1, output=False, out_device=work_device)
        down[ei].copy_(lin.weight.data.t())


def _register_down_proj_hook(module: nn.Linear, inter: int, fp32_had: bool) -> None:
    if getattr(module, "_quarot_down_prehook_registered", False):
        return
    had_k, k = hadamard_utils.get_hadK(inter)
    had_k = had_k.to(dtype=module.weight.dtype, device=module.weight.device) if had_k is not None else None

    def pre_hook(_m, inp):
        x = inp[0]
        return (_apply_online_full_had(x, had_k, k, fp32_had),)

    module.register_forward_pre_hook(pre_hook)
    module._quarot_down_prehook_registered = True


def _register_o_proj_hook(module: nn.Linear, num_heads: int, head_dim: int, fp32_had: bool) -> None:
    if getattr(module, "_quarot_o_prehook_registered", False):
        return
    had_k, k = hadamard_utils.get_hadK(num_heads)
    if had_k is not None:
        had_k = had_k.to(dtype=module.weight.dtype, device=module.weight.device)

    def pre_hook(_m, inp):
        x = inp[0]
        return (_apply_online_partial_had_o(x, num_heads, head_dim, had_k, k, fp32_had),)

    module.register_forward_pre_hook(pre_hook)
    module._quarot_o_prehook_registered = True


def _patch_experts_forward(experts: nn.Module, fp32_had: bool) -> None:
    if getattr(experts, "_quarot_experts_patched", False):
        return
    inter = experts.expert_dim
    had_k, k = hadamard_utils.get_hadK(inter)
    if had_k is not None:
        had_k = had_k.to(dtype=experts.down_proj.dtype, device=experts.down_proj.device)

    def forward(self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, router_indices: torch.Tensor):
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        if self.training:
            next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=self.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hit[:]:
                with torch.no_grad():
                    _, token_idx = torch.where(expert_mask[expert_idx[0]])
                current_state = hidden_states[token_idx]
                gate_up = current_state @ self.gate_up_proj[expert_idx]
                gate, up = gate_up.chunk(2, dim=-1)
                mid = up * self.act_fn(gate)
                mid = _apply_online_full_had(mid, had_k, k, fp32_had)
                out = mid @ self.down_proj[expert_idx]
                weighted_output = out[0] * routing_weights[token_idx, expert_idx, None]
                next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
            next_states = next_states.view(batch_size, -1, self.hidden_size)
        else:
            hidden_states = hidden_states.repeat(self.num_experts, 1)
            hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
            gate_up = torch.bmm(hidden_states, self.gate_up_proj)
            gate, up = gate_up.chunk(2, dim=-1)
            mid = up * self.act_fn(gate)
            mid = _apply_online_full_had(mid, had_k, k, fp32_had)
            next_states = torch.bmm(mid, self.down_proj)
            next_states = next_states.reshape(self.num_experts, batch_size, -1, self.hidden_size)
            next_states = (
                next_states * routing_weights.transpose(0, 1).view(self.num_experts, batch_size, -1)[..., None]
            )
            next_states = next_states.sum(dim=0)
        return next_states

    experts.forward = forward.__get__(experts, experts.__class__)
    experts._quarot_experts_patched = True
    experts._quarot_fp32_had = fp32_had


def apply_quarot_inference_hooks_qwen3_vl_moe(
    model: nn.Module,
    Q_cpu: torch.Tensor,
    *,
    fp32_had: bool = False,
) -> None:
    """
    Re-install online Hadamard / MoE forward / visual-Q hooks after ``from_pretrained``.

    Use when weights were already rotated + (optionally) quantized and saved; hooks are not
    persisted. Pass the same ``Q`` tensor stored in ``quarot_aux.pt``.
    """
    from mllm_quant.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        Qwen3VLMoeTextDecoderLayer,
        Qwen3VLMoeTextMLP,
        Qwen3VLMoeTextSparseMoeBlock,
    )

    lm = model.language_model
    if getattr(lm, "_quarot_inference_hooks_installed", False):
        logger.warning("QuaRot inference hooks already installed; skipping.")
        return

    cfg = lm.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    for layer in lm.layers:
        assert isinstance(layer, Qwen3VLMoeTextDecoderLayer)
        attn = layer.self_attn
        mlp = layer.mlp
        if isinstance(mlp, Qwen3VLMoeTextMLP):
            _register_down_proj_hook(mlp.down_proj, mlp.intermediate_size, fp32_had)
        elif isinstance(mlp, Qwen3VLMoeTextSparseMoeBlock):
            _patch_experts_forward(mlp.experts, fp32_had)
        else:
            raise TypeError(type(mlp))
        _register_o_proj_hook(attn.o_proj, cfg.num_attention_heads, head_dim, fp32_had)

    lm._quarot_Q_cpu = Q_cpu
    lm._quarot_applied = True
    lm._quarot_inference_hooks_installed = True

    moe_inner = getattr(model, "model", None)
    if moe_inner is not None and hasattr(moe_inner, "get_image_features"):
        moe_inner._quarot_visual_Q = Q_cpu.to(dtype=torch.float32)
        _patch_get_image_features_for_quarot(moe_inner)
    else:
        logger.warning("QuaRot inference hooks: no model.get_image_features; text-only.")

    logger.info("QuaRot: inference hooks re-installed (MoE/dense Had + o_proj + visual @ Q).")


def _rot_right_Q(t: torch.Tensor, qm: torch.Tensor) -> torch.Tensor:
    """Map vision-side LLM hidden vectors into QuaRot rotated subspace: h' = h @ Q."""
    qm = qm.to(device=t.device, dtype=torch.float32)
    return (t.float() @ qm).to(dtype=t.dtype)


def _patch_get_image_features_for_quarot(moe_model: nn.Module) -> None:
    """
    Vision tower output lives in the original LLM hidden basis; after rotating LLM weights,
    we must right-multiply injected image/video embeddings (and deepstack lists) by Q.

    get_video_features delegates to get_image_features, so one patch covers both.
    """
    if getattr(moe_model, "_quarot_get_image_features_patched", False):
        return
    orig = moe_model.get_image_features

    def get_image_features(self, pixel_values, image_grid_thw=None):
        image_embeds, deepstack_image_embeds = orig(pixel_values, image_grid_thw)
        Qm = getattr(self, "_quarot_visual_Q", None)
        if Qm is None:
            return image_embeds, deepstack_image_embeds
        image_embeds = tuple(_rot_right_Q(x, Qm) for x in image_embeds)
        if deepstack_image_embeds is not None:
            deepstack_image_embeds = [_rot_right_Q(x, Qm) for x in deepstack_image_embeds]
        return image_embeds, deepstack_image_embeds

    moe_model.get_image_features = types.MethodType(get_image_features, moe_model)
    moe_model._quarot_get_image_features_patched = True


@torch.inference_mode()
def apply_quarot_rotation_qwen3_vl_moe(
    model: nn.Module,
    rotate_mode: Literal["hadamard", "random"] = "hadamard",
    fp32_had: bool = False,
    work_device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Apply QuaRot-style rotation to Qwen3VLMoeForConditionalGeneration **language model only**
    (ViT weights unchanged). Injected image/video embeddings are right-multiplied by Q in
    ``model.model.get_image_features`` so they match the rotated ``embed_tokens`` basis.
    """
    from mllm_quant.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        Qwen3VLMoeTextDecoderLayer,
        Qwen3VLMoeTextMLP,
        Qwen3VLMoeTextSparseMoeBlock,
    )

    lm = model.language_model
    if getattr(lm, "_quarot_applied", False):
        logger.warning("QuaRot rotation already applied to this language_model; skipping.")
        return lm._quarot_Q_cpu
    lm_head = model.lm_head
    cfg = lm.config
    hidden = cfg.hidden_size
    head_dim = getattr(cfg, "head_dim", hidden // cfg.num_attention_heads)

    dev = work_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = _get_orthogonal_matrix(hidden, rotate_mode, dev).cpu()

    logger.info("QuaRot: fusing RMSNorm into adjacent LLM linears (language_model only)...")
    _fuse_layer_norms_qwen3_vl_moe_text(lm, lm_head)

    logger.info("QuaRot: rotating embedding + lm_head...")
    emb = lm.embed_tokens.weight.data
    lm.embed_tokens.weight.data = torch.matmul(emb.to(torch.float64).to(q.device), q).to(dtype=emb.dtype, device=emb.device)
    lh = lm_head.weight.data
    lm_head.weight.data = torch.matmul(lh.to(torch.float64).to(q.device), q).to(dtype=lh.dtype, device=lh.device)

    logger.info("QuaRot: rotating decoder layers...")
    for layer in lm.layers:
        assert isinstance(layer, Qwen3VLMoeTextDecoderLayer)
        attn = layer.self_attn
        wdev = attn.q_proj.weight.device
        q_dev = q.to(device=wdev)
        _rotate_attention_layer(attn, q_dev, cfg.num_attention_heads, head_dim, wdev)

        mlp = layer.mlp
        if isinstance(mlp, Qwen3VLMoeTextMLP):
            inter = mlp.intermediate_size
            _rotate_dense_mlp(mlp, q_dev, inter, wdev)
            _register_down_proj_hook(mlp.down_proj, inter, fp32_had)
        elif isinstance(mlp, Qwen3VLMoeTextSparseMoeBlock):
            _rotate_moe_block(mlp, q_dev, wdev)
            _patch_experts_forward(mlp.experts, fp32_had)
        else:
            raise TypeError(type(mlp))
        _register_o_proj_hook(attn.o_proj, cfg.num_attention_heads, head_dim, fp32_had)

    lm._quarot_Q_cpu = q.cpu()
    lm._quarot_applied = True

    # Multimodal: align ViT→LLM injected features (prefill scatter + deepstack) with rotated embed basis
    moe_inner = getattr(model, "model", None)
    if moe_inner is not None and hasattr(moe_inner, "get_image_features"):
        moe_inner._quarot_visual_Q = q.to(dtype=torch.float32)
        _patch_get_image_features_for_quarot(moe_inner)
    else:
        logger.warning(
            "QuaRot: no inner model.get_image_features; skipping visual injection Q (text-only models)."
        )

    logger.info(
        "QuaRot: applied to Qwen3-VL-MoE LLM (hooks + visual feature right-Q in get_image_features)."
    )
    return lm._quarot_Q_cpu


def compare_next_token_logits(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    inputs: dict,
    max_logit_diff_for_pass: float = 5.0,
) -> Tuple[bool, dict]:
    """
    Run forward on the same multimodal inputs; compare argmax next-token at last position
    and max absolute logit difference (bf16/FlashAttention may yield larger diffs).
    """
    dev_a = next(model_a.parameters()).device
    dev_b = next(model_b.parameters()).device

    def _to(m, d):
        b = {k: v for k, v in inputs.items() if v is not None}
        out = {}
        for k, v in b.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(d)
            else:
                out[k] = v
        return out

    with torch.inference_mode():
        oa = model_a(**_to(model_a, dev_a), logits_to_keep=1)
        ob = model_b(**_to(model_b, dev_b), logits_to_keep=1)
        la = oa.logits[:, -1, :].float()
        lb = ob.logits[:, -1, :].float()
    ta = la.argmax(dim=-1)
    tb = lb.argmax(dim=-1)
    max_diff = (la.cpu() - lb.cpu()).abs().max().item()
    match_token = bool(torch.equal(ta.cpu(), tb.cpu()))
    ok = match_token and max_diff < max_logit_diff_for_pass
    info = {
        "argmax_match": match_token,
        "token_a": ta.tolist(),
        "token_b": tb.tolist(),
        "max_abs_logit_diff": max_diff,
        "pass_argmax": match_token,
        "pass_strict": ok,
    }
    return ok, info
