#!/usr/bin/env python3
"""
查看和可视化校准数据集内容。
支持 ShareGPT4V 格式的 JSON/JSONL，包含图片和对话文本。

用法:
    python -m mllm_quant.calibration.view_calib \
        --calib_data_path /path/to/calib_coco_256.json \
        --calib_image_folder /path/to/data \
        --n_samples 512 \
        --output_dir view
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional

# 添加项目根目录
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def load_calib_json(data_path: str) -> List[Dict[str, Any]]:
    """加载校准 JSON/JSONL 文件"""
    if data_path.endswith(".jsonl"):
        dataset = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    dataset.append(json.loads(line))
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    return dataset


def resolve_image_path(image_folder: str, image_path: str) -> Optional[str]:
    """
    解析图片完整路径。支持多种路径格式。
    - image_path 可能是 "coco/train2017/xxx.jpg" 或 "xxx.jpg"
    - image_folder 可能是 data 目录或 data/coco/train2017
    """
    candidates = [
        os.path.join(image_folder, image_path),
        os.path.join(image_folder, os.path.basename(image_path)),
    ]
    # 若 image_path 为 coco/train2017/xxx.jpg，且 image_folder 为 train2017，
    # 则需用 data 目录（train2017 的上两级）
    folder_norm = image_folder.rstrip("/")
    if "coco" in image_path and "train2017" in image_path:
        if folder_norm.endswith("train2017"):
            data_dir = os.path.dirname(os.path.dirname(image_folder))
            candidates.insert(1, os.path.join(data_dir, image_path))
        else:
            parent = os.path.dirname(image_folder)
            candidates.append(os.path.join(parent, image_path))
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def extract_text_info(conversations: List[Dict]) -> Dict[str, Any]:
    """从 conversations 提取 human/gpt 文本"""
    user_text = ""
    asst_text = ""
    for conv in conversations:
        role = conv.get("from", "")
        if role == "human":
            user_text = conv.get("value", "")
        elif role == "gpt":
            asst_text = conv.get("value", "")
    # 清洗 <image> 占位符
    user_clean = user_text.replace("<image>", "").replace("\n<image>", "").strip()
    return {"user": user_text, "user_clean": user_clean, "assistant": asst_text}


def print_statistics(dataset: List[Dict], image_folder: str, n_samples: int) -> None:
    """打印校准集统计信息"""
    print("\n" + "=" * 60)
    print("校准数据集统计信息")
    print("=" * 60)
    print(f"数据文件总样本数: {len(dataset)}")
    print(f"实际使用样本数 (n_samples): {min(n_samples, len(dataset))}")
    print(f"图片根目录: {image_folder}")
    print()

    # 样本结构
    sample = dataset[0] if dataset else {}
    print("单条样本结构:")
    for k, v in sample.items():
        if k == "conversations":
            print(f"  - {k}: [{len(v)} 轮对话]")
            for i, c in enumerate(v[:2]):
                val = str(c.get("value", ""))[:80] + "..." if len(str(c.get("value", ""))) > 80 else str(c.get("value", ""))
                print(f"      [{i}] from={c.get('from')} value={val}")
        else:
            print(f"  - {k}: {v}")
    print()

    # 文本统计
    user_lengths = []
    asst_lengths = []
    user_prompts = []
    missing_images = 0
    for i, item in enumerate(dataset[:n_samples]):
        convs = item.get("conversations", [])
        info = extract_text_info(convs)
        user_lengths.append(len(info["user_clean"]))
        asst_lengths.append(len(info["assistant"]))
        user_prompts.append(info["user_clean"])
        img_path = item.get("image")
        if img_path:
            full = resolve_image_path(image_folder, img_path)
            if not full:
                missing_images += 1

    print("文本长度统计 (字符数):")
    print(f"  Human 提示词: min={min(user_lengths)}, max={max(user_lengths)}, avg={sum(user_lengths)/len(user_lengths):.1f}")
    print(f"  Assistant 回复: min={min(asst_lengths)}, max={max(asst_lengths)}, avg={sum(asst_lengths)/len(asst_lengths):.1f}")
    print()
    print(f"缺失图片数: {missing_images} / {min(n_samples, len(dataset))}")
    print()

    # 常见提示词
    prompt_counter = Counter(user_prompts)
    print("出现最多的 Human 提示词 (Top 5):")
    for prompt, count in prompt_counter.most_common(5):
        p = (prompt[:60] + "..." if len(prompt) > 60 else prompt)
        print(f"  [{count:3d}x] {p}")
    print("=" * 60)


def visualize_samples(
    dataset: List[Dict],
    image_folder: str,
    n_samples: int,
    output_dir: str,
    grid_size: int = 4,
    num_grids: int = 4,
) -> None:
    """可视化样本：图片网格 + 文本卡片"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError as e:
        print(f"可视化需要 matplotlib 和 PIL: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    samples_to_show = min(n_samples, len(dataset))
    indices = list(range(samples_to_show))

    # 1. 图片网格
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    shown = 0
    for idx in indices:
        if shown >= grid_size * grid_size:
            break
        item = dataset[idx]
        img_path_raw = item.get("image")
        if not img_path_raw:
            continue
        full_path = resolve_image_path(image_folder, img_path_raw)
        if not full_path:
            continue
        try:
            img = Image.open(full_path).convert("RGB")
            ax = axes[shown]
            ax.imshow(img)
            ax.set_title(f"#{item.get('id', idx)}", fontsize=8)
            ax.axis("off")
            shown += 1
        except Exception as e:
            print(f"  Failed to load image {full_path}: {e}")

    for j in range(shown, len(axes)):
        axes[j].axis("off")
    plt.suptitle(f"Calib Image Samples (n={samples_to_show})", fontsize=12)
    plt.tight_layout()
    grid_path = os.path.join(output_dir, "calib_image_grid.png")
    plt.savefig(grid_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {grid_path}")

    # 2. 文本长度分布直方图
    user_lengths = []
    asst_lengths = []
    for idx in indices:
        info = extract_text_info(dataset[idx].get("conversations", []))
        user_lengths.append(len(info["user_clean"]))
        asst_lengths.append(len(info["assistant"]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(user_lengths, bins=30, edgecolor="black", alpha=0.7)
    axes[0].set_title("Human Prompt Length (chars)")
    axes[0].set_xlabel("Length")
    axes[1].hist(asst_lengths, bins=30, edgecolor="black", alpha=0.7, color="orange")
    axes[1].set_title("Assistant Response Length (chars)")
    axes[1].set_xlabel("Length")
    plt.tight_layout()
    hist_path = os.path.join(output_dir, "calib_text_length_hist.png")
    plt.savefig(hist_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {hist_path}")

    # 3. 若干图文样例卡片（图+文本）
    num_cards = min(6, samples_to_show)
    for card_i in range(num_cards):
        idx = indices[card_i]
        item = dataset[idx]
        img_path_raw = item.get("image")
        full_path = None
        if img_path_raw:
            full_path = resolve_image_path(image_folder, img_path_raw)
        info = extract_text_info(item.get("conversations", []))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.2]})
        if full_path and os.path.exists(full_path):
            img = Image.open(full_path).convert("RGB")
            axes[0].imshow(img)
        axes[0].set_title(f"ID: {item.get('id', idx)}")
        axes[0].axis("off")

        text = f"Human: {info['user_clean'][:300]}{'...' if len(info['user_clean'])>300 else ''}\n\n"
        text += f"Assistant: {info['assistant'][:400]}{'...' if len(info['assistant'])>400 else ''}"
        axes[1].text(0.05, 0.95, text, transform=axes[1].transAxes, fontsize=9,
                     verticalalignment="top", wrap=True, family="sans-serif")
        axes[1].set_title("Conversation")
        axes[1].axis("off")
        plt.tight_layout()
        card_path = os.path.join(output_dir, f"calib_sample_card_{card_i}.png")
        plt.savefig(card_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {card_path}")

    # 4. 保存前几条样本的原始 JSON 片段
    sample_json_path = os.path.join(output_dir, "calib_samples_preview.json")
    preview = []
    for idx in indices[:10]:
        item = dataset[idx]
        preview.append({
            "id": item.get("id"),
            "image": item.get("image"),
            "human": extract_text_info(item.get("conversations", []))["user_clean"][:200],
            "assistant_len": len(extract_text_info(item.get("conversations", []))["assistant"]),
        })
    with open(sample_json_path, "w", encoding="utf-8") as f:
        json.dump(preview, f, indent=2, ensure_ascii=False)
    print(f"  已保存: {sample_json_path}")


def main():
    parser = argparse.ArgumentParser(description="查看与可视化校准数据集")
    parser.add_argument("--calib_data_path", type=str,
                        default="data/calibration/calib_coco_256.json",
                        help="校准数据 JSON/JSONL 路径")
    parser.add_argument("--calib_image_folder", type=str,
                        default="data/coco/train2017",
                        help="图片根目录 (若路径含 coco/train2017，将尝试其父目录)")
    parser.add_argument("--n_samples", type=int, default=512,
                        help="使用的样本数量 (与 quantize.py 的 --n_samples 一致)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="可视化输出目录，默认: mllm_quant/calibration/view")
    parser.add_argument("--no_viz", action="store_true",
                        help="仅打印统计，不生成可视化")
    args = parser.parse_args()

    # 解析图片路径：JSON 中为 "coco/train2017/xxx.jpg"，需以 data 目录为根
    image_folder = os.path.abspath(args.calib_image_folder)
    if not os.path.isdir(image_folder):
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        if os.path.isdir(data_dir):
            image_folder = os.path.abspath(data_dir)
            print(f"注意: 原 image_folder 不存在，使用 data 目录: {image_folder}")
    # 若 image_folder 为 train2017，需改为 data 目录才能解析 coco/train2017/xxx.jpg
    _test_img_path = "coco/train2017/000000242828.jpg"
    if not os.path.exists(os.path.join(image_folder, _test_img_path)):
        if image_folder.rstrip("/").endswith("train2017"):
            data_dir = os.path.dirname(os.path.dirname(image_folder))
            if os.path.exists(os.path.join(data_dir, _test_img_path)):
                image_folder = data_dir
                print(f"自动调整图片目录为 data: {image_folder}")

    if not os.path.exists(args.calib_data_path):
        print(f"错误: 数据文件不存在: {args.calib_data_path}")
        sys.exit(1)

    dataset = load_calib_json(args.calib_data_path)
    if not dataset:
        print("错误: 校准集为空")
        sys.exit(1)

    print_statistics(dataset, image_folder, args.n_samples)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "view")
    output_dir = os.path.abspath(output_dir)

    if not args.no_viz:
        print("\n生成可视化...")
        visualize_samples(
            dataset=dataset,
            image_folder=image_folder,
            n_samples=args.n_samples,
            output_dir=output_dir,
        )
        print(f"\n所有输出已保存到: {output_dir}")


if __name__ == "__main__":
    main()
