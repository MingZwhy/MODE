"""ILP solver for modality-aware expert bit-width allocation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .bit_config import ExpertBitConfig, parse_layer_idx, save_expert_bit_config


@dataclass(frozen=True)
class ILPResult:
    expert_bits: ExpertBitConfig
    objective: float
    target_avg_bits: float
    candidate_bits: Tuple[int, ...]
    solver: str


def _normalize(values: Sequence[float]) -> List[float]:
    total = float(sum(values))
    if total <= 0:
        n = len(values)
        return [1.0 / n for _ in values] if n else []
    return [float(v) / total for v in values]


def _unwrap_sensitivity(data: Mapping[str, Any]) -> Mapping[str, Any]:
    if "sensitivities" in data and isinstance(data["sensitivities"], Mapping):
        return data["sensitivities"]
    if "layers" in data and isinstance(data["layers"], Mapping):
        return data["layers"]
    return data


def _lookup_layer_sensitivity(
    sensitivity: Mapping[str, Any],
    layer_key: str,
    layer_idx: int,
) -> Mapping[str, Any]:
    for key in (layer_key, str(layer_idx), f"layers.{layer_idx}"):
        val = sensitivity.get(key)
        if isinstance(val, Mapping):
            return val
    # Slow but convenient fallback for full module paths.
    for key, val in sensitivity.items():
        if parse_layer_idx(key) == layer_idx and isinstance(val, Mapping):
            return val
    return {}


def _lookup_loss(
    layer_sens: Mapping[str, Any],
    expert_idx: int,
    bit: int,
    modality: str,
    missing_penalty: float,
) -> float:
    e = layer_sens.get(str(expert_idx), layer_sens.get(expert_idx, {}))
    if not isinstance(e, Mapping):
        return missing_penalty
    b = e.get(str(bit), e.get(bit, {}))
    if isinstance(b, Mapping):
        val = b.get(modality)
        if val is None and modality == "vision":
            val = b.get("key_vision")
        if val is None and modality == "text":
            val = b.get("T")
        if val is None and modality == "vision":
            val = b.get("V")
        if val is not None:
            return float(val)
    return missing_penalty


def _solve_layer_dp(cost: np.ndarray, candidate_bits: Sequence[int], budget: int) -> Tuple[List[int], float]:
    """Exact dynamic-programming fallback for one layer."""
    n, m = cost.shape
    inf = float("inf")
    dp: List[Dict[int, Tuple[float, List[int]]]] = [{0: (0.0, [])}]
    for expert_idx in range(n):
        cur: Dict[int, Tuple[float, List[int]]] = {}
        for used, (prev_cost, prev_bits) in dp[-1].items():
            for j, bit in enumerate(candidate_bits):
                new_used = used + int(bit)
                new_cost = prev_cost + float(cost[expert_idx, j])
                old = cur.get(new_used)
                if old is None or new_cost < old[0]:
                    cur[new_used] = (new_cost, prev_bits + [int(bit)])
        dp.append(cur)
    if budget in dp[-1]:
        best_cost, bits = dp[-1][budget]
        return bits, best_cost
    # If an exact equality is impossible, use the nearest feasible budget.
    feasible = sorted(dp[-1].items(), key=lambda kv: (abs(kv[0] - budget), kv[1][0]))
    best_cost, bits = feasible[0][1]
    return bits, best_cost


def _solve_layer_milp(cost: np.ndarray, candidate_bits: Sequence[int], budget: int) -> Tuple[List[int], float]:
    """Solve one layer with scipy.optimize.milp."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    n, m = cost.shape
    c = cost.reshape(-1)
    integrality = np.ones(n * m, dtype=np.int8)
    bounds = Bounds(0, 1)

    constraints = lil_matrix((n + 1, n * m), dtype=float)
    lb = np.ones(n + 1, dtype=float)
    ub = np.ones(n + 1, dtype=float)
    lb[0] = ub[0] = float(budget)

    for expert_idx in range(n):
        for j, bit in enumerate(candidate_bits):
            col = expert_idx * m + j
            constraints[0, col] = float(bit)
            constraints[expert_idx + 1, col] = 1.0

    res = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(constraints.tocsr(), lb, ub),
        options={"disp": False},
    )
    if not res.success:
        raise RuntimeError(res.message)

    x = np.asarray(res.x).reshape(n, m)
    selected = x.argmax(axis=1)
    bits = [int(candidate_bits[j]) for j in selected]
    return bits, float(res.fun)


def build_costs_for_layer(
    layer_key: str,
    layer_freq: Mapping[str, Any],
    layer_sens: Mapping[str, Any],
    candidate_bits: Sequence[int],
    *,
    vision_key: str = "dominant_image",
    missing_penalty: float = 1e6,
) -> np.ndarray:
    """Build the paper objective coefficients for one MoE layer."""
    text_freq = list(map(float, layer_freq.get("text", [])))
    vision_freq = list(map(float, layer_freq.get(vision_key, layer_freq.get("image", []))))
    if not text_freq and not vision_freq:
        raise ValueError(f"No text/{vision_key} frequency for {layer_key}")
    num_experts = max(len(text_freq), len(vision_freq))
    if len(text_freq) < num_experts:
        text_freq += [0.0] * (num_experts - len(text_freq))
    if len(vision_freq) < num_experts:
        vision_freq += [0.0] * (num_experts - len(vision_freq))

    ft = _normalize(text_freq)
    fv = _normalize(vision_freq)
    costs = np.zeros((num_experts, len(candidate_bits)), dtype=np.float64)
    for expert_idx in range(num_experts):
        for j, bit in enumerate(candidate_bits):
            dt = _lookup_loss(layer_sens, expert_idx, int(bit), "text", missing_penalty)
            dv = _lookup_loss(layer_sens, expert_idx, int(bit), "vision", missing_penalty)
            costs[expert_idx, j] = ft[expert_idx] * dt + fv[expert_idx] * dv
    return costs


def solve_expert_bit_allocation(
    frequency_data: Mapping[str, Any],
    sensitivity_data: Mapping[str, Any],
    *,
    target_avg_bits: float,
    candidate_bits: Iterable[int] = (2, 3, 4),
    vision_key: str = "dominant_image",
    solver: str = "auto",
    missing_penalty: float = 1e6,
) -> ILPResult:
    """Solve expert bit-width allocation independently for each MoE layer."""
    candidate_bits = tuple(int(b) for b in candidate_bits)
    if not candidate_bits:
        raise ValueError("candidate_bits must not be empty")
    sensitivity = _unwrap_sensitivity(sensitivity_data)
    expert_bits: ExpertBitConfig = {}
    total_obj = 0.0
    solver_used = "scipy-milp"

    for layer_key, layer_freq in frequency_data.items():
        if not isinstance(layer_freq, Mapping):
            continue
        layer_idx = parse_layer_idx(layer_key)
        if layer_idx is None:
            continue
        layer_sens = _lookup_layer_sensitivity(sensitivity, str(layer_key), layer_idx)
        costs = build_costs_for_layer(
            str(layer_key),
            layer_freq,
            layer_sens,
            candidate_bits,
            vision_key=vision_key,
            missing_penalty=missing_penalty,
        )
        num_experts = costs.shape[0]
        budget_float = num_experts * float(target_avg_bits)
        budget = int(round(budget_float))
        if not math.isclose(budget, budget_float, rel_tol=0, abs_tol=1e-6):
            raise ValueError(
                f"Layer {layer_idx}: num_experts * target_avg_bits must be integer "
                f"for equality budget, got {budget_float}"
            )

        try:
            if solver == "dp":
                raise RuntimeError("DP requested")
            bits, obj = _solve_layer_milp(costs, candidate_bits, budget)
        except Exception:
            bits, obj = _solve_layer_dp(costs, candidate_bits, budget)
            solver_used = "dp" if solver_used == "scipy-milp" else solver_used

        expert_bits[layer_idx] = {expert_idx: bit for expert_idx, bit in enumerate(bits)}
        total_obj += obj

    return ILPResult(
        expert_bits=expert_bits,
        objective=total_obj,
        target_avg_bits=float(target_avg_bits),
        candidate_bits=candidate_bits,
        solver=solver_used,
    )


def solve_from_files(
    frequency_json: str | Path,
    sensitivity_json: str | Path,
    output_json: str | Path,
    *,
    target_avg_bits: float,
    candidate_bits: Iterable[int] = (2, 3, 4),
    vision_key: str = "dominant_image",
    solver: str = "auto",
    missing_penalty: float = 1e6,
) -> ILPResult:
    with open(frequency_json, "r", encoding="utf-8") as f:
        frequency_data = json.load(f)
    with open(sensitivity_json, "r", encoding="utf-8") as f:
        sensitivity_data = json.load(f)

    result = solve_expert_bit_allocation(
        frequency_data,
        sensitivity_data,
        target_avg_bits=target_avg_bits,
        candidate_bits=candidate_bits,
        vision_key=vision_key,
        solver=solver,
        missing_penalty=missing_penalty,
    )
    save_expert_bit_config(
        output_json,
        result.expert_bits,
        metadata={
            "frequency_json": str(frequency_json),
            "sensitivity_json": str(sensitivity_json),
            "target_avg_bits": result.target_avg_bits,
            "candidate_bits": list(result.candidate_bits),
            "objective": result.objective,
            "solver": result.solver,
            "vision_key": vision_key,
        },
    )
    return result
