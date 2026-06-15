#!/bin/bash
# 校准数据准备脚本
# 按照ShareGPT4V格式准备多模态校准数据

set -e

# 配置数据目录
DATA_ROOT="${DATA_ROOT:-./data}"
mkdir -p "$DATA_ROOT"

echo "========================================"
echo "校准数据准备脚本"
echo "数据根目录: $DATA_ROOT"
echo "========================================"

# 函数: 下载并解压
download_and_extract() {
    local url=$1
    local output_dir=$2
    local filename=$(basename "$url")
    
    mkdir -p "$output_dir"
    
    if [ -f "$output_dir/$filename" ]; then
        echo "文件已存在: $output_dir/$filename"
    else
        echo "下载中: $url"
        wget -q --show-progress -P "$output_dir" "$url"
    fi
    
    # 解压 (如果是zip文件)
    if [[ "$filename" == *.zip ]]; then
        echo "解压中: $filename"
        unzip -q -o "$output_dir/$filename" -d "$output_dir"
    fi
}

# ========================================
# 1. 下载COCO train2017图像 (推荐用于校准)
# ========================================
echo ""
echo "[1/4] 下载COCO train2017图像..."
COCO_DIR="$DATA_ROOT/coco"
if [ ! -d "$COCO_DIR/train2017" ]; then
    download_and_extract "http://images.cocodataset.org/zips/train2017.zip" "$COCO_DIR"
else
    echo "COCO train2017 已存在"
fi

# ========================================
# 2. 下载ShareGPT4V JSON标注文件
# ========================================
echo ""
echo "[2/4] 下载ShareGPT4V标注文件..."
SHAREGPT4V_DIR="$DATA_ROOT/sharegpt4v"
mkdir -p "$SHAREGPT4V_DIR"

# 下载100K高质量标注 (推荐用于校准，较小)
ANNO_FILE="sharegpt4v_instruct_gpt4-vision_cap100k.json"
if [ ! -f "$SHAREGPT4V_DIR/$ANNO_FILE" ]; then
    echo "下载 $ANNO_FILE ..."
    wget -q --show-progress -O "$SHAREGPT4V_DIR/$ANNO_FILE" \
        "https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/$ANNO_FILE"
else
    echo "$ANNO_FILE 已存在"
fi

# ========================================
# 3. 下载LLaVA预训练图像 (可选)
# ========================================
echo ""
echo "[3/4] 下载LLaVA预训练图像 (可选)..."
LLAVA_DIR="$DATA_ROOT/llava/llava_pretrain"
if [ ! -d "$LLAVA_DIR/images" ]; then
    echo "下载 LLaVA pretrain images..."
    mkdir -p "$LLAVA_DIR"
    download_and_extract "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip" "$LLAVA_DIR"
else
    echo "LLaVA pretrain images 已存在"
fi

# ========================================
# 4. 创建校准数据子集
# ========================================
echo ""
echo "[4/4] 创建校准数据子集..."
python3 << 'EOF'
import os
import json
import random

DATA_ROOT = os.environ.get("DATA_ROOT", "./data")
SHAREGPT4V_DIR = os.path.join(DATA_ROOT, "sharegpt4v")
CALIB_DIR = os.path.join(DATA_ROOT, "calibration")
os.makedirs(CALIB_DIR, exist_ok=True)

# 读取完整标注
anno_file = os.path.join(SHAREGPT4V_DIR, "sharegpt4v_instruct_gpt4-vision_cap100k.json")
if os.path.exists(anno_file):
    with open(anno_file, "r") as f:
        data = json.load(f)
    
    # 筛选有COCO图像的样本
    coco_samples = [item for item in data if "coco" in item.get("image", "").lower()]
    
    if len(coco_samples) < 256:
        # 如果COCO样本不够，使用所有样本
        coco_samples = data
    
    # 随机采样256个用于校准
    random.seed(42)
    calib_samples = random.sample(coco_samples, min(256, len(coco_samples)))
    
    # 保存校准子集
    output_file = os.path.join(CALIB_DIR, "calib_coco_256.json")
    with open(output_file, "w") as f:
        json.dump(calib_samples, f, ensure_ascii=False, indent=2)
    
    print(f"校准数据已保存: {output_file}")
    print(f"样本数量: {len(calib_samples)}")
else:
    print(f"标注文件不存在: {anno_file}")
    print("请先下载ShareGPT4V标注文件")
EOF

echo ""
echo "========================================"
echo "校准数据准备完成！"
echo ""
echo "数据目录结构:"
echo "$DATA_ROOT/"
echo "├── coco/"
echo "│   └── train2017/          # COCO图像"
echo "├── sharegpt4v/"
echo "│   └── *.json              # 完整标注"
echo "├── calibration/"
echo "│   └── calib_coco_256.json # 校准子集"
echo "└── llava/ (可选)"
echo ""
echo "使用示例:"
echo "  python scripts/quantize.py \\"
echo "    --model qwen3_vl \\"
echo "    --model_path Qwen/Qwen3-VL \\"
echo "    --calib_data $DATA_ROOT/calibration/calib_coco_256.json \\"
echo "    --image_folder $DATA_ROOT"
echo "========================================"
