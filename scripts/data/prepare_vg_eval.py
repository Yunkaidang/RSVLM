#!/usr/bin/env python3
"""
Prepare evaluation manifests for multiple VG datasets (VRSBench, DIOR-RSVG,
RRSIS-D, RSVG). Each manifest is a JSON list of samples containing image path,
referring expression, and ground-truth bounding box in pixel coordinates.

Example:
    python scripts/data/prepare_vg_eval.py \
        --output-dir /data0/yunkai/VG_eval_manifests \
        --vrsbench-root /data0/yunkai/VRSBench \
        --dior-rsvg-root /data0/yunkai/DIOR-RSVG \
        --rrsis-root /data0/yunkai/RRSIS-D \
        --rsvg-root /data0/yunkai/RSVG/rsvg
"""

from __future__ import annotations

import argparse
import json
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Sequence


def save_manifest(samples: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)


def convert_vrsbench(root: Path) -> List[Dict]:
    ann_dir = root / "annotations" / "Annotations_val"
    img_dir = root / "images" / "Images_val"
    samples: List[Dict] = []
    for ann_path in sorted(ann_dir.glob("*.json")):
        with ann_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        image_name = f"{ann_path.stem}.png"
        image_path = str((img_dir / image_name).resolve())
        width = 512
        height = 512
        for obj in data.get("objects", []):
            sentence = (obj.get("referring_sentence") or "").strip()
            coords = obj.get("obj_coord") or []
            if not sentence or len(coords) != 4:
                continue
            x1 = int(round(float(coords[0]) * width))
            y1 = int(round(float(coords[1]) * height))
            x2 = int(round(float(coords[2]) * width))
            y2 = int(round(float(coords[3]) * height))
            samples.append(
                {
                    "dataset": "VRSBench-VG",
                    "image_path": image_path,
                    "sentence": sentence,
                    "bbox": [x1, y1, x2, y2],
                    "image_width": width,
                    "image_height": height,
                }
            )
    return samples


def convert_dior_rsvg(root: Path) -> List[Dict]:
    ann_dir = root / "Annotations"
    img_dir = root / "JPEGImages"
    test_ids = [line.strip() for line in (root / "test.txt").read_text().splitlines() if line.strip()]
    samples: List[Dict] = []
    for image_id in test_ids:
        xml_path = ann_dir / f"{image_id}.xml"
        if not xml_path.exists():
            continue
        tree = ET.parse(str(xml_path))
        root_xml = tree.getroot()
        filename = root_xml.findtext("filename") or f"{image_id}.jpg"
        width = int(root_xml.findtext("size/width"))
        height = int(root_xml.findtext("size/height"))
        image_path = str((img_dir / filename).resolve())
        for obj in root_xml.findall("object"):
            desc = (obj.findtext("description") or obj.findtext("name") or "").strip()
            bbox = obj.find("bndbox")
            if not desc or bbox is None:
                continue
            try:
                x1 = int(round(float(bbox.findtext("xmin"))))
                y1 = int(round(float(bbox.findtext("ymin"))))
                x2 = int(round(float(bbox.findtext("xmax"))))
                y2 = int(round(float(bbox.findtext("ymax"))))
            except (TypeError, ValueError):
                continue
            samples.append(
                {
                    "dataset": "DIOR-RSVG",
                    "image_path": image_path,
                    "sentence": desc,
                    "bbox": [x1, y1, x2, y2],
                    "image_width": width,
                    "image_height": height,
                }
            )
    return samples


def convert_rrsis(rrsis_root: Path) -> List[Dict]:
    refs_path = rrsis_root / "rrsisd" / "refs(unc).p"
    inst_path = rrsis_root / "rrsisd" / "instances.json"
    img_dir = rrsis_root / "images" / "rrsisd" / "JPEGImages"
    with refs_path.open("rb") as f:
        refs = pickle.load(f)
    with inst_path.open("r", encoding="utf-8") as f:
        instances = json.load(f)
    ann_map = {ann["id"]: ann for ann in instances["annotations"]}
    img_map = {img["id"]: img for img in instances["images"]}
    samples: List[Dict] = []
    for ref in refs:
        if ref.get("split") != "test":
            continue
        ann = ann_map.get(ref["ann_id"])
        img_meta = img_map.get(ref["image_id"])
        if not ann or not img_meta:
            continue
        x, y, w, h = ann["bbox"]
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))
        sentences = [
            (sent.get("raw") or sent.get("sent") or "").strip()
            for sent in ref.get("sentences", [])
        ]
        sentences = [s for s in sentences if s]
        if not sentences:
            continue
        for sent in sentences:
            samples.append(
                {
                    "dataset": "RRSIS-D",
                    "image_path": str((img_dir / ref["file_name"]).resolve()),
                    "sentence": sent,
                    "bbox": [x1, y1, x2, y2],
                    "image_width": int(img_meta["width"]),
                    "image_height": int(img_meta["height"]),
                }
            )
    return samples


def convert_rsvg(root: Path) -> List[Dict]:
    import torch  # lazy import

    img_dir = root / "images"
    test_path = root / "rsvg_test.pth"
    data = torch.load(str(test_path))
    samples: List[Dict] = []
    for image_name, bbox, sentence, *_ in data:
        sentence = (sentence or "").strip()
        if not sentence or not bbox:
            continue
        bbox_vals = [int(round(float(v))) for v in bbox]
        samples.append(
            {
                "dataset": "RSVG",
                "image_path": str((img_dir / image_name).resolve()),
                "sentence": sentence,
                "bbox": bbox_vals,
                "image_width": 512,
                "image_height": 512,
            }
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vrsbench-root", type=Path)
    parser.add_argument("--dior-rsvg-root", type=Path)
    parser.add_argument("--rrsis-root", type=Path)
    parser.add_argument("--rsvg-root", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, int] = {}

    if args.vrsbench_root:
        samples = convert_vrsbench(args.vrsbench_root.resolve())
        save_manifest(samples, args.output_dir / "vrsbench_vg_val.json")
        summary["VRSBench-VG"] = len(samples)

    if args.dior_rsvg_root:
        samples = convert_dior_rsvg(args.dior_rsvg_root.resolve())
        save_manifest(samples, args.output_dir / "dior_rsvg_test.json")
        summary["DIOR-RSVG"] = len(samples)

    if args.rrsis_root:
        samples = convert_rrsis(args.rrsis_root.resolve())
        save_manifest(samples, args.output_dir / "rrsis_d_test.json")
        summary["RRSIS-D"] = len(samples)

    if args.rsvg_root:
        samples = convert_rsvg(args.rsvg_root.resolve())
        save_manifest(samples, args.output_dir / "rsvg_test.json")
        summary["RSVG"] = len(samples)

    for name, count in summary.items():
        print(f"{name}: {count} samples")
    if not summary:
        print("No datasets were processed. Please pass at least one root path.")


if __name__ == "__main__":
    main()
