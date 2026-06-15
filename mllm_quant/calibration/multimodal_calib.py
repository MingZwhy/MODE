"""
Multimodal calibration dataset for GPTQ quantization.
支持 ShareGPT4V 格式的校准数据
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm


def load_image(image_path: str) -> Image.Image:
    """Load image from path and convert to RGB."""
    return Image.open(image_path).convert('RGB')


def get_multimodal_calib_dataset(
    data_path: str,
    image_folder: str,
    processor: Any,
    tokenizer: Any = None,
    n_samples: int = 128,
    shuffle: bool = True,
    seed: int = 42,
    max_seq_length: int = 2048,
    model_type: str = "qwen",
) -> List[Dict[str, torch.Tensor]]:
    """
    Load multimodal calibration dataset containing both text and images.
    
    Args:
        data_path: Path to the dataset file (json or jsonl)
        image_folder: Root folder for images
        processor: Model processor (for processing images and text)
        tokenizer: Model tokenizer (optional, will use processor if None)
        n_samples: Number of calibration samples
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        max_seq_length: Maximum sequence length
        model_type: Model type ("qwen", "internvl", etc.)
        
    Returns:
        List of processed data dictionaries
    """
    # 如果没有提供 tokenizer，尝试从 processor 获取
    if tokenizer is None:
        tokenizer = getattr(processor, 'tokenizer', None)
    
    # Load dataset
    if data_path.endswith(".jsonl"):
        dataset = []
        with open(data_path, "r", encoding="utf-8") as json_file:
            for line in json_file:
                dataset.append(json.loads(line.strip()))
    elif data_path.endswith(".json"):
        with open(data_path, "r", encoding="utf-8") as json_file:
            dataset = json.load(json_file)
    else:
        raise ValueError(f"Unsupported file type: {data_path}")
    
    print(f"Loaded {len(dataset)} samples from {data_path}")
    
    if shuffle:
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(dataset)
    
    data_list = []
    skipped = 0
    
    for i in tqdm(range(n_samples), desc="Loading calibration data"):
        idx = i % len(dataset)
        data_item = dataset[idx]
        
        # Load images
        images = None
        if 'image' in data_item and data_item['image']:
            image_path = data_item['image']
            if isinstance(image_path, list):
                images = []
                for img_p in image_path:
                    full_path = os.path.join(image_folder, img_p)
                    if os.path.exists(full_path):
                        images.append(load_image(full_path))
                    else:
                        print(f"Warning: Image not found: {full_path}")
            else:
                full_path = os.path.join(image_folder, image_path)
                if os.path.exists(full_path):
                    images = [load_image(full_path)]
                else:
                    print(f"Warning: Image not found: {full_path}")
        
        # Process data
        try:
            data_dict = process_calibration_item(
                images=images,
                data_item=data_item,
                processor=processor,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                model_type=model_type,
            )
            
            if data_dict is not None:
                data_list.append(data_dict)
            else:
                skipped += 1
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            skipped += 1
    
    print(f"Successfully loaded {len(data_list)} samples, skipped {skipped}")
    return data_list


def process_calibration_item(
    images: Optional[List[Image.Image]],
    data_item: Dict[str, Any],
    processor: Any,
    tokenizer: Any = None,
    max_seq_length: int = 2048,
    model_type: str = "qwen",
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Process a single calibration item.
    
    Args:
        images: List of PIL images
        data_item: Data item containing conversations
        processor: Model processor
        tokenizer: Model tokenizer
        max_seq_length: Maximum sequence length
        model_type: Model type for specific handling
        
    Returns:
        Processed data dictionary or None if processing fails
    """
    try:
        # Extract conversation
        conversations = data_item.get("conversations", [])
        user_text = ""
        asst_text = ""
        
        for conv in conversations:
            role = conv.get("from", "")
            if role == "human":
                user_text = conv.get("value", "")
            elif role == "gpt":
                asst_text = conv.get("value", "")
        
        # Clean up text - remove image placeholders
        user_text = user_text.replace("<image>", "").replace("\n<image>", "").strip()
        
        if not user_text:
            return None
        
        # Build messages based on model type
        if model_type in ["qwen", "qwen3_vl", "qwen3_vl_moe"]:
            return _process_qwen_item(images, user_text, asst_text, processor, max_seq_length)
        elif model_type in ["kimi_vl"]:
            return _process_kimi_item(images, user_text, asst_text, processor, max_seq_length)
        elif model_type in ["internvl", "internvl3"]:
            return _process_internvl_item(images, user_text, asst_text, processor, tokenizer, max_seq_length)
        else:
            return _process_generic_item(images, user_text, asst_text, processor, tokenizer, max_seq_length)
            
    except Exception as e:
        print(f"Error processing calibration item: {e}")
        return None


def _process_qwen_item(
    images: Optional[List[Image.Image]],
    user_text: str,
    asst_text: str,
    processor: Any,
    max_seq_length: int,
) -> Optional[Dict[str, torch.Tensor]]:
    """Process item for Qwen-VL models."""
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        print("Warning: qwen_vl_utils not installed, using fallback processing")
        process_vision_info = None
    
    # Build message format for Qwen
    if images:
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": user_text})
        
        messages = [
            {"role": "user", "content": content},
        ]
        if asst_text:
            messages.append({"role": "assistant", "content": asst_text})
    else:
        messages = [
            {"role": "user", "content": user_text},
        ]
        if asst_text:
            messages.append({"role": "assistant", "content": asst_text})
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False if asst_text else True,
    )
    
    # Get text-only token count (before image token expansion)
    text_only_ids = processor.tokenizer.encode(text)
    text_token_count = len(text_only_ids)

    # Process with Qwen processor
    if images and process_vision_info:
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
    elif images:
        inputs = processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True,
        )
    else:
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        )

    # Token statistics
    original_len = inputs['input_ids'].shape[-1]
    image_token_count = max(0, original_len - text_token_count)

    # Truncate to max_seq_length if needed (right-truncation, preserves image tokens at front)
    truncated_len = original_len
    if original_len > max_seq_length:
        truncated_len = max_seq_length
        if 'input_ids' in inputs and isinstance(inputs['input_ids'], torch.Tensor):
            inputs['input_ids'] = inputs['input_ids'][..., :max_seq_length]
        if 'attention_mask' in inputs and isinstance(inputs['attention_mask'], torch.Tensor):
            inputs['attention_mask'] = inputs['attention_mask'][..., :max_seq_length]

    print(f"  [Qwen] text_tokens={text_token_count}, image_tokens={image_token_count}, "
          f"total={original_len} -> {truncated_len}"
          + (" (truncated)" if truncated_len < original_len else ""))

    # Remove batch dimension
    data_dict = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if v.dim() > 0:
                data_dict[k] = v.squeeze(0) if v.shape[0] == 1 else v
            else:
                data_dict[k] = v
        elif isinstance(v, list) and len(v) > 0:
            data_dict[k] = v[0] if len(v) == 1 else v
        else:
            data_dict[k] = v

    return data_dict


def _process_kimi_item(
    images: Optional[List[Image.Image]],
    user_text: str,
    asst_text: str,
    processor: Any,
    max_seq_length: int,
) -> Optional[Dict[str, torch.Tensor]]:
    """Process item for Kimi-VL models.

    Kimi-VL uses image_grid_hws (N, 2) instead of Qwen's image_grid_thw (N, 3).
    We pass images directly to the Kimi-VL processor without qwen_vl_utils.
    """
    # Build message format (same OpenAI-style as Qwen)
    if images:
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": user_text})

        messages = [
            {"role": "user", "content": content},
        ]
        if asst_text:
            messages.append({"role": "assistant", "content": asst_text})
    else:
        messages = [
            {"role": "user", "content": user_text},
        ]
        if asst_text:
            messages.append({"role": "assistant", "content": asst_text})

    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False if asst_text else True,
    )

    # Get text-only token count (before image token expansion)
    text_only_ids = processor.tokenizer.encode(text)
    text_token_count = len(text_only_ids)

    # Process with Kimi-VL processor (pass images directly, no qwen_vl_utils)
    if images:
        inputs = processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True,
        )
    else:
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        )

    # Token statistics
    original_len = inputs['input_ids'].shape[-1]
    image_token_count = max(0, original_len - text_token_count)

    # Truncate to max_seq_length if needed (right-truncation, preserves image tokens at front)
    truncated_len = original_len
    if original_len > max_seq_length:
        truncated_len = max_seq_length
        if 'input_ids' in inputs and isinstance(inputs['input_ids'], torch.Tensor):
            inputs['input_ids'] = inputs['input_ids'][..., :max_seq_length]
        if 'attention_mask' in inputs and isinstance(inputs['attention_mask'], torch.Tensor):
            inputs['attention_mask'] = inputs['attention_mask'][..., :max_seq_length]

    print(f"  [KimiVL] text_tokens={text_token_count}, image_tokens={image_token_count}, "
          f"total={original_len} -> {truncated_len}"
          + (" (truncated)" if truncated_len < original_len else ""))

    # Remove batch dimension
    data_dict = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if v.dim() > 0:
                data_dict[k] = v.squeeze(0) if v.shape[0] == 1 else v
            else:
                data_dict[k] = v
        elif isinstance(v, list) and len(v) > 0:
            data_dict[k] = v[0] if len(v) == 1 else v
        else:
            data_dict[k] = v

    return data_dict


def _process_internvl_item(
    images: Optional[List[Image.Image]],
    user_text: str,
    asst_text: str,
    processor: Any,
    tokenizer: Any,
    max_seq_length: int,
) -> Optional[Dict[str, torch.Tensor]]:
    """Process item for InternVL models (HF format).

    Uses the unified processor(text=..., images=...) call so that:
    1. Dynamic resolution tiling is applied (multiple patches per image)
    2. pixel_values keeps the correct 4-D shape (num_tiles, C, H, W)
    3. Image placeholder tokens in input_ids match pixel_values
    """
    tok = tokenizer if tokenizer else getattr(processor, 'tokenizer', processor)

    if images:
        # Build message in HF InternVL chat format
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": user_text})
        messages = [{"role": "user", "content": content}]
        if asst_text:
            messages.append({"role": "assistant", "content": asst_text})

        # Use processor.apply_chat_template + processor() for proper image handling
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False if asst_text else True,
        )

        # Get text-only token count (image placeholders not expanded)
        text_token_count = len(tok.encode(text))

        # Unified processor call — handles dynamic tiling + correct pixel_values shape
        inputs = processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True,
        )

        original_len = inputs['input_ids'].shape[-1]
        image_token_count = max(0, original_len - text_token_count)
        pixel_shape = inputs['pixel_values'].shape if 'pixel_values' in inputs else None

        # Truncate to max_seq_length if needed
        truncated_len = original_len
        if original_len > max_seq_length:
            truncated_len = max_seq_length
            if 'input_ids' in inputs and isinstance(inputs['input_ids'], torch.Tensor):
                inputs['input_ids'] = inputs['input_ids'][..., :max_seq_length]
            if 'attention_mask' in inputs and isinstance(inputs['attention_mask'], torch.Tensor):
                inputs['attention_mask'] = inputs['attention_mask'][..., :max_seq_length]

        print(f"  [InternVL] text_tokens={text_token_count}, image_tokens={image_token_count}, "
              f"total={original_len} -> {truncated_len}"
              + (" (truncated)" if truncated_len < original_len else "")
              + f", pixel_values={pixel_shape}")

        # Remove batch dimension (keep pixel_values as-is, it's (num_tiles, C, H, W))
        data_dict = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if k == 'pixel_values':
                    data_dict[k] = v  # keep 4-D: (num_tiles, C, H, W)
                elif v.dim() > 0 and v.shape[0] == 1:
                    data_dict[k] = v.squeeze(0)
                else:
                    data_dict[k] = v
            elif isinstance(v, list) and len(v) > 0:
                data_dict[k] = v[0] if len(v) == 1 else v
            else:
                data_dict[k] = v
    else:
        full_text = user_text + (f"\n{asst_text}" if asst_text else "")
        original_text_len = len(tok.encode(full_text))
        text_inputs = tok(
            full_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        )
        truncated_text_len = text_inputs["input_ids"].shape[-1]

        print(f"  [InternVL] text_tokens={original_text_len} -> {truncated_text_len}"
              + (" (truncated)" if truncated_text_len < original_text_len else "")
              + ", image_tokens=0")

        data_dict = {
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
        }

    return data_dict


def _process_generic_item(
    images: Optional[List[Image.Image]],
    user_text: str,
    asst_text: str,
    processor: Any,
    tokenizer: Any,
    max_seq_length: int,
) -> Optional[Dict[str, torch.Tensor]]:
    """Generic processing for other models."""
    full_text = user_text
    if asst_text:
        full_text = f"{user_text}\n{asst_text}"

    tok = tokenizer if tokenizer else processor
    if hasattr(tok, 'encode'):
        original_text_len = len(tok.encode(full_text))
    else:
        original_text_len = -1

    if images:
        inputs = processor(
            text=full_text,
            images=images[0],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        )
    else:
        inputs = tok(
            full_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        )

    truncated_len = inputs['input_ids'].shape[-1] if 'input_ids' in inputs else -1
    has_image = "yes" if images else "no"

    print(f"  [Generic] text_tokens={original_text_len} -> {truncated_len}"
          + (" (truncated)" if 0 < truncated_len < original_text_len else "")
          + f", has_image={has_image}")

    data_dict = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            data_dict[k] = v.squeeze(0)
        else:
            data_dict[k] = v

    return data_dict


def collate_calibration_data(
    data_list: List[Dict[str, Any]],
    tokenizer: Any = None,
    pad_token_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Collate calibration data into batches.
    
    Args:
        data_list: List of processed data dictionaries
        tokenizer: Model tokenizer (for pad token id)
        pad_token_id: Padding token ID (used if tokenizer not provided)
        
    Returns:
        Collated batch dictionary
    """
    if not data_list:
        raise ValueError("Empty data list")
    
    if tokenizer is not None:
        pad_id = tokenizer.pad_token_id or pad_token_id
    else:
        pad_id = pad_token_id
    
    batch = {}
    batch_size = len(data_list)
    
    # 收集所有 key
    all_keys = set()
    for item in data_list:
        all_keys.update(item.keys())
    
    for key in all_keys:
        values = [item.get(key) for item in data_list if key in item]
        if not values:
            continue
        
        first_val = values[0]
        
        if key == "input_ids":
            # 左填充 input_ids
            max_len = max(v.shape[-1] if isinstance(v, torch.Tensor) else len(v) for v in values)
            padded = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
            for i, v in enumerate(values):
                if isinstance(v, torch.Tensor):
                    seq_len = v.shape[-1]
                    padded[i, -seq_len:] = v.flatten()[-seq_len:]
                else:
                    seq_len = len(v)
                    padded[i, -seq_len:] = torch.tensor(v)
            batch[key] = padded
            
        elif key == "attention_mask":
            max_len = max(v.shape[-1] if isinstance(v, torch.Tensor) else len(v) for v in values)
            padded = torch.zeros((batch_size, max_len), dtype=torch.long)
            for i, v in enumerate(values):
                if isinstance(v, torch.Tensor):
                    seq_len = v.shape[-1]
                    padded[i, -seq_len:] = v.flatten()[-seq_len:]
                else:
                    seq_len = len(v)
                    padded[i, -seq_len:] = torch.tensor(v)
            batch[key] = padded
            
        elif key == "pixel_values":
            # pixel_values 直接 stack 或 cat
            if isinstance(first_val, torch.Tensor):
                if all(v.shape == first_val.shape for v in values):
                    batch[key] = torch.stack(values, dim=0)
                else:
                    # 不同大小的 pixel_values，cat 在一起
                    batch[key] = torch.cat(values, dim=0)
            else:
                batch[key] = values
                
        elif key == "image_grid_thw":
            # Qwen 模型特有的 grid 信息
            if isinstance(first_val, torch.Tensor):
                batch[key] = torch.cat(values, dim=0)
            else:
                batch[key] = values
                
        elif isinstance(first_val, torch.Tensor):
            # 其他 tensor，尝试 stack
            try:
                if all(v.shape == first_val.shape for v in values):
                    batch[key] = torch.stack(values, dim=0)
                else:
                    batch[key] = values
            except:
                batch[key] = values
        else:
            batch[key] = values
    
    return batch


def prepare_calibration_inputs(
    batch: Dict[str, Any],
    device: Union[str, torch.device],
) -> Dict[str, torch.Tensor]:
    """
    Prepare calibration inputs for the model by moving to device.
    
    Args:
        batch: Collated batch dictionary
        device: Target device
        
    Returns:
        Prepared inputs dictionary
    """
    inputs = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            inputs[k] = [t.to(device) for t in v]
        else:
            inputs[k] = v
    
    return inputs


def get_calib_data_for_quantization(
    data_path: str,
    image_folder: str,
    processor: Any,
    tokenizer: Any = None,
    n_samples: int = 128,
    batch_size: int = 1,
    device: str = "cuda",
    model_type: str = "qwen",
    max_seq_length: int = 2048,
) -> List[Dict[str, torch.Tensor]]:
    """
    一站式获取量化所需的校准数据。
    
    Args:
        data_path: 校准数据 JSON/JSONL 文件路径
        image_folder: 图像根目录
        processor: 模型 processor
        tokenizer: 模型 tokenizer（可选）
        n_samples: 校准样本数
        batch_size: batch 大小
        device: 设备
        model_type: 模型类型
        max_seq_length: 最大序列长度
        
    Returns:
        校准数据 batch 列表
    """
    # 加载并处理数据
    data_list = get_multimodal_calib_dataset(
        data_path=data_path,
        image_folder=image_folder,
        processor=processor,
        tokenizer=tokenizer,
        n_samples=n_samples,
        model_type=model_type,
        max_seq_length=max_seq_length,
    )
    
    # 分 batch
    batches = []
    for i in range(0, len(data_list), batch_size):
        batch_data = data_list[i:i + batch_size]
        batch = collate_calibration_data(batch_data, tokenizer)
        batch = prepare_calibration_inputs(batch, device)
        batches.append(batch)
    
    return batches
