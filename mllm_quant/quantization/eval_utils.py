"""
Evaluation utilities for quantized multimodal models using lmms-eval.

Supports two backends:
  - **vllm**:       vLLM tensor-parallel inference (fast, but model code is fixed).
  - **accelerate**: HuggingFace + accelerate data-parallel inference
                    (flexible – model code can be freely modified).

The accelerate backend is launched via ``accelerate launch`` in a
subprocess. By default, ``--num_processes=N`` uses **data parallel** (each
rank loads a full model on ``cuda:i``). With *model-parallel* mode, use
``--num_processes=1`` and ``device_map=auto`` so a **single** model shard
spreads across visible GPUs (lower per-GPU memory).
"""
import logging
import os
import subprocess
import sys

# HF mirror for dataset/model downloads. Set HF_TOKEN in the environment if needed.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from pathlib import Path
from typing import List, Optional, Union

import torch

logger = logging.getLogger(__name__)


# 默认结果保存目录
DEFAULT_OUTPUT_DIR = "results"


# ====================================================================
# Model architecture -> lmms-eval model wrapper name mapping
# ====================================================================
#   用户使用的 model_arch (与 quantize.py --model_type 一致)
#   → lmms-eval 注册的 model name
ARCH_TO_LMMS_MODEL = {
    "qwen3_vl":     "qwen3_vl",     # simple/qwen3_vl.py  (auto-detects MoE)
    "qwen3_vl_moe": "qwen3_vl",     # same wrapper, A\d+B regex triggers MoE path
    "kimi_vl":      "huggingface",   # chat/huggingface.py (generic HF chat model)
    "internvl":     "internvl3",     # simple/internvl3.py (dedicated InternVL wrapper)
}


def _build_hf_model_args(
    model_arch: str,
    model_path: str,
    *,
    apply_quarot: bool = False,
    attn_implementation: Optional[str] = None,
    model_parallel: bool = False,
) -> str:
    """Build the ``--model_args`` string for an accelerate-based lmms-eval run.

    Each model architecture may need different constructor kwargs
    (e.g. ``attn_implementation``, ``trust_remote_code``).
    When ``apply_quarot`` is True, lmms-eval model wrappers load the checkpoint then
    run in-place QuaRot (requires ``mllm_quant`` on ``PYTHONPATH``).

    ``attn_implementation``: if set, passed to qwen/kimi wrappers; default is
    ``flash_attention_2``. Use ``eager`` to match ``--verify_quarot`` loading in
    ``quantize.py`` (Kimi/Qwen MoE verify uses eager).
    """
    quarot_suffix = ",apply_quarot=True" if apply_quarot else ""
    _attn = attn_implementation or "flash_attention_2"
    if model_arch in ("qwen3_vl", "qwen3_vl_moe"):
        s = (
            f"pretrained={model_path},"
            f"attn_implementation={_attn},"
            f"interleave_visuals=False{quarot_suffix}"
        )
    elif model_arch == "kimi_vl":
        s = (
            f"pretrained={model_path},"
            f"attn_implementation={_attn},"
            f"trust_remote_code=True{quarot_suffix}"
        )
    elif model_arch == "internvl":
        if apply_quarot:
            raise ValueError("apply_quarot is not supported for internvl in this eval path.")
        s = f"pretrained={model_path},device_map=auto"
    else:
        if apply_quarot:
            raise ValueError(f"apply_quarot not wired for model_arch={model_arch!r}.")
        s = f"pretrained={model_path}"

    if model_parallel and "device_map=" not in s:
        s += ",device_map=auto"
    return s


# ====================================================================
# Accelerate backend  (subprocess-based)
# ====================================================================
def evaluate_model_accelerate(
    model_path: str,
    model_arch: str,
    tasks: Union[str, List[str]] = "chartqa",
    output_path: Optional[str] = None,
    batch_size: int = 1,
    num_gpus: Optional[int] = None,
    limit: Optional[Union[int, float]] = None,
    log_samples: bool = False,
    apply_quarot: bool = False,
    attn_implementation: Optional[str] = None,
    accelerate_model_parallel: bool = False,
) -> int:
    """Launch ``accelerate launch -m lmms_eval`` as a subprocess.

    Args:
        model_path:  Path to the pretrained / quantized model.
        model_arch:  Architecture identifier (qwen3_vl_moe / kimi_vl / internvl / …).
                     Mapped to the lmms-eval model wrapper name via *ARCH_TO_LMMS_MODEL*.
        tasks:       Comma-separated string or list of task names.
        output_path: Directory for lmms-eval result files.
        batch_size:  Per-GPU batch size (default 1, recommended for HF models).
        num_gpus:    Number of GPUs for **data-parallel** mode only (``--num_processes``).
                     Ignored when *accelerate_model_parallel* is True.
        limit:       Max samples per task (None = full dataset).
        log_samples: Whether lmms-eval should dump per-sample outputs.
        apply_quarot: After load, apply QuaRot in-place (Qwen3-VL-MoE / Kimi-VL only).
        attn_implementation: Optional override for qwen/kimi (default flash_attention_2).
        accelerate_model_parallel: If True, use ``--num_processes 1`` and pass
            ``device_map=auto`` so one model replica is sharded across visible GPUs
            (HF / Accelerate pipeline parallel style), instead of N× full-model data parallel.

    Returns:
        Subprocess return code (0 = success).
    """
    lmms_model = ARCH_TO_LMMS_MODEL.get(model_arch)
    if lmms_model is None:
        raise ValueError(
            f"Unknown model_arch '{model_arch}'. "
            f"Available: {list(ARCH_TO_LMMS_MODEL.keys())}"
        )

    model_args = _build_hf_model_args(
        model_arch,
        model_path,
        apply_quarot=apply_quarot,
        attn_implementation=attn_implementation,
        model_parallel=accelerate_model_parallel,
    )

    if isinstance(tasks, list):
        tasks_str = ",".join(tasks)
    else:
        tasks_str = tasks

    if num_gpus is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    if accelerate_model_parallel:
        launch_num_processes = 1
        if num_gpus is not None and num_gpus > 1:
            logger.warning(
                "accelerate_model_parallel=True: using --num_processes 1 + device_map=auto "
                "(one sharded model). Ignoring eval num_gpus=%s for process count; "
                "set CUDA_VISIBLE_DEVICES to choose which GPUs participate in auto sharding.",
                num_gpus,
            )
    else:
        launch_num_processes = num_gpus

    output_path = output_path or str(DEFAULT_OUTPUT_DIR)
    os.makedirs(output_path, exist_ok=True)

    # ---- build command --------------------------------------------------
    cmd = [
        sys.executable, "-m",
        "accelerate.commands.launch",
        "--num_processes", str(launch_num_processes),
        "--main_process_port", "12346",
        "-m", "lmms_eval",
        "--model", lmms_model,
        f"--model_args={model_args}",
        "--tasks", tasks_str,
        "--batch_size", str(batch_size),
        "--output_path", output_path,
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if log_samples:
        cmd += ["--log_samples"]

    logger.info("Launching accelerate eval:\n  %s", " \\\n    ".join(cmd))
    print(f"\n{'=' * 70}")
    print(f"  Accelerate eval: {lmms_model}  ({model_arch})")
    print(f"  model_path : {model_path}")
    print(f"  model_args : {model_args}")
    print(f"  tasks      : {tasks_str}")
    print(f"  num_processes : {launch_num_processes}")
    print(f"  model_parallel (HF shard): {accelerate_model_parallel}")
    print(f"  num_gpus (hint / DP only): {num_gpus}")
    print(f"  batch_size : {batch_size}")
    print(f"  limit      : {limit}")
    print(f"  apply_quarot: {apply_quarot}")
    print(f"  attn_implementation: {attn_implementation or 'flash_attention_2 (default)'}")
    print(f"  output_path: {output_path}")
    print(f"{'=' * 70}\n")

    # ---- run subprocess (inherit stdio for real-time output) ------------
    env = os.environ.copy()
    # Ensure the subprocess inherits HF mirror / token settings
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    # Kimi / Qwen3-VL-MoE may need mllm_quant (unified Kimi load, MoE quarot_aux, or apply_quarot).
    if apply_quarot or model_arch in ("kimi_vl", "qwen3_vl_moe"):
        _mllm_repo = Path(__file__).resolve().parents[2]
        _prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(_mllm_repo) + (os.pathsep + _prev if _prev else "")
        )

    result = subprocess.run(cmd, env=env)

    if result.returncode == 0:
        print(f"\nAccelerate eval completed.  Results saved to: {output_path}")
    else:
        print(f"\nAccelerate eval failed with return code {result.returncode}")

    return result.returncode


# ====================================================================
# vLLM backend  (in-process, original implementation)
# ====================================================================
def _evaluate_model_vllm(
    model_path: str,
    tasks: Union[str, List[str]] = "chartqa",
    output_path: Optional[str] = None,
    batch_size: int = 64,
    num_fewshot: Optional[int] = None,
    device: Optional[str] = None,
    tensor_parallel_size: Optional[int] = None,
    data_parallel_size: Optional[int] = None,
    gpu_memory_utilization: float = 0.95,
    max_model_len: int = 16384,
    include_path: Optional[str] = None,
    log_samples: bool = False,
    limit: Optional[Union[int, float]] = None,
) -> Optional[dict]:
    """Evaluate using vLLM backend (in-process)."""
    from lmms_eval import evaluator
    from lmms_eval.loggers import EvaluationTracker
    from lmms_eval.tasks import TaskManager
    from lmms_eval.utils import get_datetime_str, load_yaml_config, make_table

    output_path = output_path or str(DEFAULT_OUTPUT_DIR)
    os.makedirs(output_path, exist_ok=True)

    # 解析 tasks
    if isinstance(tasks, str):
        task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    else:
        task_list = list(tasks)
    if not task_list:
        task_list = ["chartqa"]

    # 构建 vLLM model_args
    if tensor_parallel_size is None:
        tensor_parallel_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if data_parallel_size is None:
        data_parallel_size = 1
    model_args = (
        f"model={model_path},"
        f"tensor_parallel_size={tensor_parallel_size},"
        f"data_parallel_size={data_parallel_size},"
        f"gpu_memory_utilization={gpu_memory_utilization},"
        # f"max_model_len={max_model_len}"
    )

    # 初始化 TaskManager 和 task_names
    task_manager = TaskManager(verbosity="INFO", include_path=include_path, model_name="vllm")
    task_names = list(task_manager.match_tasks(task_list))
    for task in [t for t in task_list if t not in task_names]:
        if os.path.isfile(task):
            config = load_yaml_config(task)
            task_names.append(config)
    task_missing = [t for t in task_list if t not in task_names and "*" not in t]
    if task_missing:
        missing = ", ".join(task_missing)
        raise ValueError(
            f"未找到任务: {missing}。可用 `lmms-eval --tasks list` 查看任务列表"
        )

    # 初始化 EvaluationTracker
    evaluation_tracker_args = {"output_path": output_path}
    if os.environ.get("HF_TOKEN"):
        evaluation_tracker_args["token"] = os.environ["HF_TOKEN"]
    evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)

    # 调用 simple_evaluate
    datetime_str = get_datetime_str()
    results = evaluator.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        evaluation_tracker=evaluation_tracker,
        task_manager=task_manager,
        log_samples=log_samples,
        datetime_str=datetime_str,
        limit=limit,
        distributed_executor_backend=(
            "torchrun"
            if (torch.distributed.is_available() and torch.distributed.is_initialized())
            else "accelerate"
        ),
    )

    if results is None:
        return None

    # 处理 results
    if log_samples:
        samples = results.pop("samples", None)
    else:
        samples = None

    # 保存到本地
    evaluation_tracker.save_results_aggregated(
        results=results,
        samples=samples if log_samples else None,
        datetime_str=datetime_str,
    )
    if log_samples and samples:
        for task_name, config in results.get("configs", {}).items():
            if task_name in samples:
                evaluation_tracker.save_results_samples(
                    task_name=task_name,
                    samples=samples[task_name],
                )

    if evaluation_tracker.push_results_to_hub or evaluation_tracker.push_samples_to_hub:
        evaluation_tracker.recreate_metadata_card()

    # 打印到终端
    print(f"model=vllm, model_args={model_args}, num_fewshot={num_fewshot}, batch_size={batch_size}")
    print(make_table(results))
    if "groups" in results:
        print(make_table(results, "groups"))
    print(f"\n结果已保存到: {output_path}")

    return results


# ====================================================================
# Unified entry point
# ====================================================================
def evaluate_model(
    model_path: str,
    model_type: str = "vllm",
    tasks: Union[str, List[str]] = "chartqa",
    output_path: Optional[str] = None,
    batch_size: int = 64,
    num_fewshot: Optional[int] = None,
    device: Optional[str] = None,
    *,
    # --- accelerate-specific ---
    model_arch: Optional[str] = None,
    num_gpus: Optional[int] = None,
    accelerate_model_parallel: bool = False,
    # --- vLLM-specific ---
    tensor_parallel_size: Optional[int] = None,
    data_parallel_size: Optional[int] = None,
    gpu_memory_utilization: float = 0.95,
    max_model_len: int = 16384,
    # --- common ---
    include_path: Optional[str] = None,
    log_samples: bool = False,
    limit: Optional[Union[int, float]] = None,
    apply_quarot: bool = False,
    attn_implementation: Optional[str] = None,
) -> Optional[Union[dict, int]]:
    """
    使用 lmms-eval 评估多模态模型（统一入口）。

    Args:
        model_path:  模型路径（本地路径或 HuggingFace 模型名）
        model_type:  评估后端:
                       - ``"vllm"``:       vLLM tensor-parallel (默认)
                       - ``"accelerate"``: HuggingFace + accelerate data-parallel
        tasks:       评估任务，逗号分隔字符串或列表
        output_path: 结果保存目录
        batch_size:  批大小 (vllm 默认 64, accelerate 默认 1)
        num_fewshot: few-shot 样本数量
        device:      设备 (仅 vllm)

        model_arch:  模型架构标识 (accelerate 必填):
                       qwen3_vl / qwen3_vl_moe / kimi_vl / internvl
        num_gpus:    accelerate **数据并行** 时的进程数 (默认全部可见 GPU)
        accelerate_model_parallel:  True 时单进程 + device_map=auto，模型切多卡

        tensor_parallel_size:   vLLM tensor parallel GPU 数
        data_parallel_size:     vLLM data parallel GPU 数
        gpu_memory_utilization: vLLM GPU 显存占用比例
        max_model_len:          vLLM 最大序列长度

        include_path: 自定义任务配置目录
        log_samples:  是否保存逐样本输出
        limit:        每个任务的最大样本数

    Returns:
        vllm 模式: 评估结果字典 (或 None)
        accelerate 模式: 子进程返回码 (0 = 成功)
    """
    backend = model_type.lower()

    if backend == "accelerate":
        if model_arch is None:
            raise ValueError(
                "使用 accelerate 后端时必须指定 model_arch "
                "(如 qwen3_vl_moe / kimi_vl / internvl)"
            )
        return evaluate_model_accelerate(
            model_path=model_path,
            model_arch=model_arch,
            tasks=tasks,
            output_path=output_path,
            batch_size=batch_size,
            num_gpus=num_gpus,
            limit=limit,
            log_samples=log_samples,
            apply_quarot=apply_quarot,
            attn_implementation=attn_implementation,
            accelerate_model_parallel=accelerate_model_parallel,
        )
    elif backend in ("vllm", "hf"):
        return _evaluate_model_vllm(
            model_path=model_path,
            tasks=tasks,
            output_path=output_path,
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            device=device,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            include_path=include_path,
            log_samples=log_samples,
            limit=limit,
        )
    else:
        raise ValueError(
            f"不支持的 model_type: '{model_type}'。"
            "请使用 'vllm', 'accelerate', 或 'hf'"
        )
