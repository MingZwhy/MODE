#!/usr/bin/env python3
"""Solve modality-aware expert bit allocation from frequency + sensitivity JSON."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mllm_quant.expert_mixed_precision.ilp import solve_from_files


def _parse_bits(text: str):
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def main():
    parser = argparse.ArgumentParser(description="Solve expert-level mixed precision bit allocation")
    parser.add_argument("--freq_json", required=True, help="Frequency JSON from record_freq.py")
    parser.add_argument("--sensitivity_json", required=True, help="Sensitivity JSON from collect_expert_sensitivity.py")
    parser.add_argument("--output_json", required=True, help="Path to save expert bit assignment JSON")
    parser.add_argument("--target_avg_bits", type=float, required=True, help="Target average bit-width over routed experts")
    parser.add_argument("--candidate_bits", default="2,3,4", help="Comma-separated candidate bit-widths")
    parser.add_argument(
        "--vision_key",
        default="dominant_image",
        help="Frequency key for key vision tokens (default: dominant_image)",
    )
    parser.add_argument("--solver", choices=["auto", "scipy", "dp"], default="auto")
    parser.add_argument("--missing_penalty", type=float, default=1e6)
    args = parser.parse_args()

    result = solve_from_files(
        args.freq_json,
        args.sensitivity_json,
        args.output_json,
        target_avg_bits=args.target_avg_bits,
        candidate_bits=_parse_bits(args.candidate_bits),
        vision_key=args.vision_key,
        solver="dp" if args.solver == "dp" else "auto",
        missing_penalty=args.missing_penalty,
    )
    print(f"Saved expert bit allocation to {args.output_json}")
    print(f"solver={result.solver} objective={result.objective:.6g} layers={len(result.expert_bits)}")


if __name__ == "__main__":
    main()
