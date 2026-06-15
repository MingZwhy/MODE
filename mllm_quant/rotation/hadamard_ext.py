"""
Extended Hadamard helpers for dimensions not in QuaRot's built-in table (e.g. Kimi-VL FFN: 44 * 2^k).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_HAD44_F32: Optional[torch.Tensor] = None


def _is_pow2(n: int) -> bool:
    return (n & (n - 1) == 0) and n > 0


def load_had44_from_txt(path: Optional[Path] = None) -> torch.Tensor:
    """
    Load Sloane-style +/- text (one character per row entry, 44 rows).
    Returns float32 tensor [44, 44] with values ±1.
    """
    global _HAD44_F32
    if _HAD44_F32 is not None and path is None:
        return _HAD44_F32
    p = path or Path(__file__).resolve().parent / "had44.txt"
    if not p.is_file():
        raise FileNotFoundError(f"had44 matrix file not found: {p}")
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            row = [1.0 if c == "+" else -1.0 for c in s if c in "+-"]
            if row:
                rows.append(row)
    H = torch.tensor(rows, dtype=torch.float64)
    if H.shape != (44, 44):
        raise ValueError(f"Expected 44x44 Hadamard, got {tuple(H.shape)}")
    H32 = H.to(torch.float32)
    if path is None:
        _HAD44_F32 = H32
    return H32


def verify_had44(path: Optional[Path] = None, atol: float = 1e-4) -> dict:
    """
    Check unnormalized ±1 Hadamard: H @ H.T == 44 * I.
    """
    H = load_had44_from_txt(path).to(torch.float64)
    n = H.shape[0]
    prod = H @ H.T
    target = torch.eye(n, dtype=torch.float64) * n
    max_err = (prod - target).abs().max().item()
    ok = max_err < atol
    return {
        "ok": ok,
        "shape": tuple(H.shape),
        "max_orthogonality_error": max_err,
        "path": str(path or Path(__file__).resolve().parent / "had44.txt"),
    }


def get_hadk_kimi(n: int, transpose: bool = False) -> Tuple[torch.Tensor, int]:
    """
    Kronecker-style small factor K=44 for n = 44 * 2^m (Kimi dense 11264, MoE 1408, shared 2816).

    Returns (hadK, K) compatible with QuaRot ``matmul_hadU_cuda(X, hadK, K)``.
    """
    if n % 44 != 0 or not _is_pow2(n // 44):
        raise ValueError(
            f"n={n} is not supported by Hadamard-44 factorization (need n = 44 * 2^m). "
            f"remainder_mod_44={n % 44}, n//44={n//44}"
        )
    H = load_had44_from_txt()
    if transpose:
        H = H.T.contiguous()
    return H, 44


def ensure_had44_verified() -> None:
    r = verify_had44()
    if not r["ok"]:
        logger.warning(
            "Hadamard-44 orthogonality check failed (max_err=%.6g). Proceeding anyway.",
            r["max_orthogonality_error"],
        )
    else:
        logger.info(
            "Hadamard-44 OK: max |H@H.T - 44*I| = %.3e",
            r["max_orthogonality_error"],
        )
