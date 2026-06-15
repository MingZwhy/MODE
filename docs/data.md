# Data Preparation

MODE expects calibration JSON files and image folders to follow the same layout used by `mllm_quant`.

## Local Layout

```text
data/
|-- calibration/
|   |-- calib_coco_256.json
|   `-- calib_coco_512.json
|-- coco/
|   `-- train2017/
|-- sharegpt4v_instruct_gpt4-vision_cap100k.json
`-- sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json
```

`data/coco/` and the large mix JSON are local-only assets. They are ignored by git, so they can live inside the working tree without being uploaded.

## COCO

For COCO calibration, the data should be a JSON or JSONL file in a ShareGPT4V-style format. You can refer to the ShareGPT4V data preparation guide: [Data.md](https://github.com/InternLM/InternLM-XComposer/blob/main/projects/ShareGPT4V/docs/Data.md#prepare-images).

Download COCO train2017 and place images at:

```text
data/coco/train2017/
```

The calibration JSON image paths should be relative to `data/`, for example:

```json
{
  "image": "coco/train2017/000000000009.jpg"
}
```

Then use `--calib_image_folder data`.

## ShareGPT4V

The smaller `sharegpt4v_instruct_gpt4-vision_cap100k.json` file is allowed in the public repository. The much larger `sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json` file should remain local.
