#!/usr/bin/env python3
"""Collect modality-wise expert quantization sensitivity on COCO calibration data."""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mllm_quant.expert_mixed_precision.sensitivity import collect_sensitivity


def _parse_bits(text: str):
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def main():
    parser = argparse.ArgumentParser(description="Collect expert-level modality-wise sensitivity")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_type", required=True, choices=["qwen3_vl_moe", "kimi_vl", "internvl"])
    parser.add_argument("--calib_data", default="data/calibration/calib_coco_512.json")
    parser.add_argument("--calib_img", default="data")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--candidate_bits", default="2,3,4")
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--dominant_ratio", type=float, default=0.2)
    parser.add_argument("--include_assistant", action="store_true", help="Keep assistant answer in calibration prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metric",
        choices=["logit_kl", "moe_mse", "expert_mse"],
        default="logit_kl",
        help=(
            "Sensitivity metric: full-model logits KL, current-layer MoE output "
            "hidden-state MSE, or routed expert output MSE"
        ),
    )
    parser.add_argument("--kl_positions", choices=["last", "all"], default="last")
    parser.add_argument("--layer_limit", type=int, default=None, help="Debug: only process first N MoE layers")
    parser.add_argument("--expert_limit", type=int, default=None, help="Debug: only process first N experts per layer")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    collect_sensitivity(
        model_path=args.model_path,
        model_type=args.model_type,
        calib_data=args.calib_data,
        calib_img=args.calib_img,
        output_path=args.output_path,
        n_samples=args.n_samples,
        candidate_bits=_parse_bits(args.candidate_bits),
        group_size=args.group_size,
        max_seq_length=args.max_seq_length,
        dominant_ratio=args.dominant_ratio,
        human_only=not args.include_assistant,
        seed=args.seed,
        metric=args.metric,
        kl_positions=args.kl_positions,
        layer_limit=args.layer_limit,
        expert_limit=args.expert_limit,
        device=args.device,
    )
    print(f"Saved sensitivity JSON to {args.output_path}")


if __name__ == "__main__":
    main()
