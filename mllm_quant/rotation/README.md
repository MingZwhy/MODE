# QuaRot / 旋转支持说明

本文档总结当前仓库为 **多模态 MoE（Qwen3-VL-MoE、Kimi-VL）** 接入 **QuaRot 风格旋转**（全局正交矩阵 `Q`、RMSNorm 融合、FFN 侧 Hadamard 与在线 Hadamard、视觉特征与文本嵌入对齐）已完成的工程工作。  
**不旋转 ViT / 视觉 backbone**，仅处理语言模型子模块及多模态注入到 LLM 的 hidden 对齐。

---

## 1. 本目录文件

| 文件 | 作用 |
|------|------|
| `__init__.py` | 导出 Qwen/Kimi 的旋转与推理钩子 API、Hadamard-44 校验、`quarot_gptq_compat` 辅助函数。 |
| `qwen3_vl_moe_quarot.py` | Qwen3-VL-**MoE** 语言部分 QuaRot：共享 `Q`、LayerNorm 融合、Attention/MLP(MoE)/`lm_head` 权重侧吸收 `Q`，FFN down 与 QuaRot 一致的 Hadamard 融合 + 在线 Had；**patch `get_image_features`**，使视觉注入与旋转后的 `embed_tokens` 同基。 |
| `kimi_vl_quarot.py` | Kimi-VL（DeepSeek 风格 LLM + MoE）QuaRot：复用 `qwen3_vl_moe_quarot` 中通用算子；**MoE gate** 与 **post_attention_layernorm** 的 gamma 融合（含 **`shared_experts`**，与 routed experts 共用已归一化 hidden）；**Kimi 维度的 H₄₄×Walsh** 融进 `down_proj` + DeepSeek MLP 前向在线 Had；**patch `_merge_with_image_features`**，对 `image_features` 右乘 `Q`；跳过 MLA 的 head-wise V/O Hadamard 等（见源码注释）。 |
| `hadamard_ext.py` | Kimi FFN 中间维（如 44×2^k）的 **Hadamard-44** 扩展：`had44.txt` 加载、校验、`get_hadk_kimi`。 |
| `had44.txt` | 44×44 的 ±1 Hadamard（Sloane 文本格式），供 `hadamard_ext` 使用。 |

**外部依赖**：实现依赖 `3rdparty/QuaRot/fake_quant` 下的 `hadamard_utils`、`fast_hadamard_transform`（运行时 `sys.path` 注入，无需单独安装 QuaRot 包即可做旋转与等价性实验）。

---

## 2. 公开 API（`mllm_quant.rotation`）

- `apply_quarot_rotation_qwen3_vl_moe(model, rotate_mode=..., fp32_had=..., ...)`
- `apply_quarot_rotation_kimi_vl(model, rotate_mode=..., fp32_had=..., ffn_had=..., bake_mean=..., ...)`
- `verify_had44` / `load_had44_from_txt`（Hadamard-44 数据与自检）

`rotate_mode` 通常为 `"hadamard"` 或 `"random"`（与 QuaRot fake_quant 一致）。

---

## 3. 与 `scripts/quantize.py` 的衔接

- **`--verify_quarot`**（需 `--model_type qwen3_vl_moe` 或 `kimi_vl`，且 Kimi 需 `--verify_image`）：加载两份模型（baseline vs 原地旋转），在固定多模态输入上对比 **末位置 next-token**（及可选 logit 阈值），用于快速检查旋转是否数值等价。验证路径下 Attention 使用 **`eager`** 以稳定对齐。
- **`--quarot_before_quant`**：`model_type` 为 **`qwen3_vl_moe`** 或 **`kimi_vl`** 时，在 RTN / GPTQ / **gptq_fast** 前做 QuaRot，并在 `save_pretrained` 同目录写入 **`quarot_aux.pt`**（含 `model_type`、`Q`、`fp32_had`、`rotate_mode`；Kimi 另含 **`ffn_had`**）。
- **`--quarot_rotate_mode`**、`--verify_quarot_no_ffn_had` / **`--no_quarot_kimi_ffn_had`**（Kimi 上关掉 FFN H₄₄）等见 `quantize.py` help。

---

## 4. 与 lmms-eval / ChartQA 等评测的衔接

- **`mllm_quant/quantization/eval_utils.py`**：`evaluate_model_accelerate` 为 `qwen3_vl` / `qwen3_vl_moe` / `kimi_vl` 拼接 `model_args`，默认 **`attn_implementation=flash_attention_2`**（可用 `--eval_attn_implementation` 覆盖为 `eager` / `sdpa`）；**`--eval_apply_quarot`** 时在 `model_args` 中加 `apply_quarot=True`。
- **子进程 `PYTHONPATH`**：`apply_quarot` 或 **`model_arch=kimi_vl`** 时会注入 mllm_quant 仓库根，保证 lmms-eval 内能 `import mllm_quant`。
- **lmms-eval 补丁**（仓库内 `lmms-eval`）：
  - `models/simple/qwen3_vl.py`：`apply_quarot=True` 时用 **`mllm_quant.models.qwen3_vl_moe`** 加载并调用 `apply_quarot_rotation_qwen3_vl_moe`。
  - `models/chat/huggingface.py`：**Kimi** 在可 import 时**始终**用 **`mllm_quant.models.kimi_vl.KimiVLForConditionalGeneration`** 加载；`apply_quarot=True` 时调用 `apply_quarot_rotation_kimi_vl`；若目录下有 **`quarot_aux.pt`**（旋转+量化保存），则调用 **`apply_quarot_inference_hooks_kimi_vl`** 恢复 FFN 在线 Had + merge @ `Q`。

---

## 5. Kimi 建模与 checkpoint 对齐（支撑公平评测）

`mllm_quant.models.kimi_vl.modeling_kimi_vl` 曾一度为适配新 `Cache` API **误将 `max_cache_length` 写成当前序列长度**，导致 `generate` 时错误裁剪 `attention_mask`，ChartQA 等指标暴跌。现已与本地 checkpoint 中的 **`modeling_kimi_vl.py`** 在相关逻辑上对齐（`max_cache_length` 仍表示**缓存容量上限**，`DynamicCache` 无固定上限时为 `None`）。  
**评测时 baseline 与 QuaRot 应使用同一 modeling + 同一 `attn_implementation`。**

---

## 6. 已知注意点

- **`--verify_quarot`（单步、常配合 eager）≠ 多步 `generate`（bf16 / flash）逐 token 完全一致**；全任务评测应以统一 attn 与统一 modeling 下的指标为准。
- **Kimi MLA**：若 `q_lora_rank` 非空，代码中有「仅部分路径经充分测试」的告警；`q_a` 等已融合/旋转，深层 MLA 细节需按需再审计。
- **InternVL** 等架构在当前 accelerate+QuaRot 路径中 **未** 接 `apply_quarot`。

---

## 7. 相关脚本提示

`scripts/rotate.sh` 中示例命令注释了：ChartQA 默认与 `eval_utils` 一致使用 **flash_attention_2**；若要与 `--verify_quarot` 的 eager 加载严格对齐，可加 `--eval_attn_implementation eager`。

---

## 8. 旋转 + 量化（RTN / GPTQ / gptq_fast）

### Qwen3-VL-MoE

- **`--quarot_before_quant`** + `model_type=qwen3_vl_moe`：调用 `apply_quarot_rotation_qwen3_vl_moe`；MoE 校准时 `hooked_experts_forward` 使用 **`quarot_moe_mid_pre_down`** 与 QuaRot `experts.forward` 对齐。
- **推理**：`lmms-eval` **`qwen3_vl`** 检测到 **`quarot_aux.pt`** 时调用 **`apply_quarot_inference_hooks_qwen3_vl_moe`**。

### Kimi-VL

- **`--quarot_before_quant`** + `model_type=kimi_vl`：调用 `apply_quarot_rotation_kimi_vl`（可选 **`--no_quarot_kimi_ffn_had`** 关掉 FFN H₄₄）。GPTQ ModuleList MoE 在 **`gptq_utils` / `gpt_utils_fast`** 的手动 gate×up→down 重算路径中插入 **`quarot_kimi_expert_mid_pre_down`**，与已 patch 的 `DeepseekV3MLP.forward` 一致。
- **`quarot_aux.pt`** 字段：`model_type='kimi_vl'`、`ffn_had`（bool）、`Q`、`fp32_had`、`rotate_mode`。
- **推理**：`lmms-eval` **`huggingface`**（Kimi）检测到 **`quarot_aux.pt`** 时调用 **`apply_quarot_inference_hooks_kimi_vl`**（内含 **`ensure_kimi_quarot_decoder_rmsnorm_weights_are_ones`**：融合后 checkpoint 常不含 `layernorm.weight`，`from_pretrained`+`device_map` 可能留下未初始化的 γ，导致首层 RMSNorm 爆炸与 NaN logits）。
- **保存**：`quantize.py` 在 Kimi + `--quarot_before_quant` 时于 `save_pretrained` 前调用 **`materialize_kimi_rmsn_as_deepseek_rmsnorm_for_save`**，把无参数的 `RMSN` 换成 γ=1 的 `DeepseekV3RMSNorm`，使 norm 权重写入 safetensors，避免上述问题。

### 共用

- **`quarot_gptq_compat.py`**：`save_quarot_aux` / `load_quarot_aux`、`quarot_moe_mid_pre_down`、`quarot_kimi_expert_mid_pre_down`。

---

*文档随实现迭代更新；具体数学与层名以 `qwen3_vl_moe_quarot.py` / `kimi_vl_quarot.py` 源码为准。*
