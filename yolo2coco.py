#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert YOLO-format labels (txt) to COCO JSON.
This version has paths and class names defined inside the file.
"""

import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ====== USER CONFIG ======
# dataset root (relative to this script or absolute path)
DATA_ROOT = Path("data")

# input folders
TRAIN_IMAGES = DATA_ROOT / "images/train"
TRAIN_LABELS = DATA_ROOT / "labels/train"
VAL_IMAGES   = DATA_ROOT / "images/val"
VAL_LABELS   = DATA_ROOT / "labels/val"

# output folder
ANN_DIR = DATA_ROOT / "annotations"
ANN_DIR.mkdir(parents=True, exist_ok=True)

# define your classes here (single class: fire)
CLASS_NAMES = ["fire"]
# ==========================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def yolo_to_coco_bbox(xc, yc, w, h, img_w, img_h):
    x_min = (xc - w / 2.0) * img_w
    y_min = (yc - h / 2.0) * img_h
    bw = w * img_w
    bh = h * img_h
    # clip
    x_min = max(0.0, min(x_min, img_w - 1))
    y_min = max(0.0, min(y_min, img_h - 1))
    bw = max(0.0, min(bw, img_w - x_min))
    bh = max(0.0, min(bh, img_h - y_min))
    return [x_min, y_min, bw, bh]

def find_images(folder):
    return sorted([p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXTS])

def convert_split(images_dir, labels_dir, out_json, start_img_id=1, start_ann_id=1):
    imgs = find_images(images_dir)
    categories = [{"id": i+1, "name": n, "supercategory": "none"} for i, n in enumerate(CLASS_NAMES)]

    coco_images, coco_anns = [], []
    ann_id = start_ann_id
    img_id = start_img_id

    for img_path in tqdm(imgs, desc=f"Processing {images_dir.name}"):
        with Image.open(img_path) as im:
            w, h = im.size

        coco_images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h
        })

        lbl_path = labels_dir / (img_path.stem + ".txt")
        if lbl_path.is_file():
            lines = [ln.strip() for ln in lbl_path.read_text().splitlines() if ln.strip()]
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:5])
                if cls >= len(CLASS_NAMES):
                    continue
                bbox = yolo_to_coco_bbox(xc, yc, bw, bh, w, h)
                area = bbox[2] * bbox[3]
                if area <= 1:
                    continue
                coco_anns.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls+1,
                    "bbox": [round(v, 3) for v in bbox],
                    "area": round(area, 3),
                    "iscrowd": 0,
                    "segmentation": []
                })
                ann_id += 1
        img_id += 1

    coco = {
        "info": {"year": 2025, "version": "1.0", "description": "YOLO→COCO"},
        "licenses": [],
        "categories": categories,
        "images": coco_images,
        "annotations": coco_anns
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    print(f"✔ Saved {out_json} | Images={len(coco_images)} Anns={len(coco_anns)}")

def main():
    convert_split(TRAIN_IMAGES, TRAIN_LABELS, ANN_DIR/"instances_train.json")
    convert_split(VAL_IMAGES, VAL_LABELS, ANN_DIR/"instances_val.json")

if __name__ == "__main__":
    main()
