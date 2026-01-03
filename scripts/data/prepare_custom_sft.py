#!/usr/bin/env python3
"""
Prepare an MF-RSVLM-style supervised fine-tuning (SFT) list file from a set of
remote-sensing datasets (VRSBench, DIOR-RSVG, RRSIS-D, RSVG).

The script converts each dataset into the conversation format expected by
MF-RSVLM's LazySupervisedDatasetForRS and also creates a lightweight directory
structure under the specified output root (via symbolic links) so that images
can be resolved by ``data_path / source_dataset / image``.

Example
-------
python scripts/data/prepare_custom_sft.py \
    --vrsbench-root /data0/yunkai/VRSBench \
    --dior-rsvg-root /data0/yunkai/DIOR-RSVG \
    --rrsis-root /data0/yunkai/RRSIS-D \
    --rsvg-root /data0/yunkai/RSVG/rsvg \
    --output-root /data0/yunkai/MF-RSVLM_custom_sft \
    --output-list /data0/yunkai/MF-RSVLM_custom_sft/list_custom_4sets.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
    import torch
except ImportError:  # pragma: no cover - torch is always available in training env
    torch = None


def _ensure_symlink(src: Path, dst: Path) -> None:
    """Create a symbolic link from ``dst`` to ``src`` if needed."""
    src = src.resolve()
    dst = dst.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        if dst.is_symlink():
            if dst.resolve() == src:
                return
            dst.unlink()
        else:
            raise FileExistsError(
                f"{dst} already exists and is not a symbolic link. "
                "Please remove it manually if you want to recreate the mapping."
            )
    os.symlink(src, dst)


def _bbox_to_string(coords: Sequence[int]) -> str:
    return "[[{},{},{},{}]]".format(*coords)


def _clamp(value: float, *, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(value, low), high)


def convert_vrsbench(vrs_root: Path, output_root: Path) -> List[Dict]:
    images_src = vrs_root / "images" / "Images_train"
    ann_root = vrs_root / "annotations" / "Annotations_train"
    if not ann_root.exists():
        raise FileNotFoundError(f"Cannot locate VRSBench annotations under {ann_root}")

    link_dst = output_root / "VRSBench" / "train" / "images" / "Images_train"
    _ensure_symlink(images_src, link_dst)
    dataset_key = "VRSBench/train/images/Images_train"

    entries: List[Dict] = []
    for ann_path in sorted(ann_root.glob("*.json")):
        data = json.loads(ann_path.read_text())
        image_name = f"{ann_path.stem}.png"

        caption = (data.get("caption") or "").strip()
        if caption:
            entries.append(
                {
                    "image": image_name,
                    "source_dataset": dataset_key,
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\n{IT} Describe the remote sensing image in detail.",
                        },
                        {"from": "gpt", "value": caption},
                    ],
                }
            )

        for obj in data.get("objects", []):
            sentence = (obj.get("referring_sentence") or "").strip()
            coords = obj.get("obj_coord") or []
            if not sentence or len(coords) != 4:
                continue
            bbox = [
                int(round(_clamp(float(coords[0])) * 1000)),
                int(round(_clamp(float(coords[1])) * 1000)),
                int(round(_clamp(float(coords[2])) * 1000)),
                int(round(_clamp(float(coords[3])) * 1000)),
            ]
            entries.append(
                {
                    "image": image_name,
                    "source_dataset": dataset_key,
                    "conversations": [
                        {
                            "from": "human",
                            "value": (
                                "<image>\n{VG} "
                                f"{sentence} Answer with bounding box coordinates in the "
                                "format [x_min, y_min, x_max, y_max] scaled to 0-1000."
                            ),
                        },
                        {"from": "gpt", "value": _bbox_to_string(bbox)},
                    ],
                }
            )

        for qa in data.get("qa_pairs", []):
            question = (qa.get("question") or "").strip()
            answer = (qa.get("answer") or "").strip()
            if not question or not answer:
                continue
            entries.append(
                {
                    "image": image_name,
                    "source_dataset": dataset_key,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>\n{{VQA}} {question} Answer briefly.",
                        },
                        {"from": "gpt", "value": answer},
                    ],
                }
            )
    return entries


def convert_dior_rsvg(dior_root: Path, output_root: Path) -> List[Dict]:
    images_src = dior_root / "JPEGImages"
    ann_root = dior_root / "Annotations"
    train_txt = dior_root / "train.txt"
    if not train_txt.exists():
        raise FileNotFoundError(f"Missing DIOR-RSVG train split file: {train_txt}")

    link_dst = output_root / "DOIR-RSVG"
    _ensure_symlink(images_src, link_dst)
    dataset_key = "DOIR-RSVG"

    id_list = [line.strip() for line in train_txt.read_text().splitlines() if line.strip()]
    entries: List[Dict] = []
    for image_id in id_list:
        xml_path = ann_root / f"{image_id}.xml"
        if not xml_path.exists():
            continue
        root = ET.parse(xml_path).getroot()
        filename = root.findtext("filename") or f"{image_id}.jpg"
        for obj in root.findall("object"):
            desc = (obj.findtext("description") or obj.findtext("name") or "").strip()
            if not desc:
                continue
            bbox_node = obj.find("bndbox")
            if bbox_node is None:
                continue
            try:
                xmin = int(round(float(bbox_node.findtext("xmin"))))
                ymin = int(round(float(bbox_node.findtext("ymin"))))
                xmax = int(round(float(bbox_node.findtext("xmax"))))
                ymax = int(round(float(bbox_node.findtext("ymax"))))
            except (TypeError, ValueError):
                continue
            bbox = [xmin, ymin, xmax, ymax]
            entries.append(
                {
                    "image": filename,
                    "source_dataset": dataset_key,
                    "conversations": [
                        {
                            "from": "human",
                            "value": (
                                "<image>\n{VG} "
                                f"{desc} Provide the bounding box as [x_min, y_min, x_max, y_max] "
                                "in image pixels."
                            ),
                        },
                        {"from": "gpt", "value": _bbox_to_string(bbox)},
                    ],
                }
            )
    return entries


def convert_rrsis(rrsis_root: Path, output_root: Path) -> List[Dict]:
    images_src = rrsis_root / "images" / "rrsisd" / "JPEGImages"
    refs_path = rrsis_root / "rrsisd" / "refs(unc).p"
    inst_path = rrsis_root / "rrsisd" / "instances.json"
    if not refs_path.exists() or not inst_path.exists():
        raise FileNotFoundError(
            f"RRSIS-D resources not found under {rrsis_root}. Expected refs(unc).p and instances.json."
        )

    link_dst = output_root / "RRSIS-D"
    _ensure_symlink(images_src, link_dst)
    dataset_key = "RRSIS-D"

    with inst_path.open("r", encoding="utf-8") as f:
        inst_data = json.load(f)
    ann_map = {ann["id"]: ann for ann in inst_data["annotations"]}

    with refs_path.open("rb") as f:
        refs = pickle.load(f)

    entries: List[Dict] = []
    for ref in refs:
        if ref.get("split") != "train":
            continue
        ann = ann_map.get(ref["ann_id"])
        if not ann:
            continue
        x, y, w, h = ann["bbox"]
        bbox = [
            int(round(x)),
            int(round(y)),
            int(round(x + w)),
            int(round(y + h)),
        ]
        sentences = [
            (sent.get("raw") or sent.get("sent") or "").strip()
            for sent in ref.get("sentences", [])
        ]
        sentences = [s for s in sentences if s]
        if not sentences:
            continue
        for sentence in sentences:
            entries.append(
                {
                    "image": ref["file_name"],
                    "source_dataset": dataset_key,
                    "conversations": [
                        {
                            "from": "human",
                            "value": (
                                "<image>\n{VG} "
                                f"{sentence} Return the bounding box as [x_min, y_min, x_max, y_max] "
                                "in pixels."
                            ),
                        },
                        {"from": "gpt", "value": _bbox_to_string(bbox)},
                    ],
                }
            )
    return entries


def convert_rsvg(rsvg_root: Path, output_root: Path) -> List[Dict]:
    if torch is None:
        raise RuntimeError("PyTorch is required to load the RSVG .pth annotations.")
    images_src = rsvg_root / "images"
    train_pth = rsvg_root / "rsvg_train.pth"
    if not train_pth.exists():
        raise FileNotFoundError(f"Cannot locate rsvg_train.pth under {rsvg_root}")

    link_dst = output_root / "RSVG" / "images"
    _ensure_symlink(images_src, link_dst)
    dataset_key = "RSVG/images"

    data = torch.load(train_pth)
    entries: List[Dict] = []
    for image_name, bbox, sentence, *_ in data:
        sentence = (sentence or "").strip()
        if not sentence or not bbox:
            continue
        bbox_vals = [int(round(float(v))) for v in bbox]
        entries.append(
            {
                "image": image_name,
                "source_dataset": dataset_key,
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            "<image>\n{VG} "
                            f"{sentence} Provide [x_min, y_min, x_max, y_max] in pixels."
                        ),
                    },
                    {"from": "gpt", "value": _bbox_to_string(bbox_vals)},
                ],
            }
        )
    return entries


def build(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    builders = []
    if args.vrsbench_root:
        builders.append(("VRSBench", convert_vrsbench, Path(args.vrsbench_root)))
    if args.dior_rsvg_root:
        builders.append(("DOIR-RSVG", convert_dior_rsvg, Path(args.dior_rsvg_root)))
    if args.rrsis_root:
        builders.append(("RRSIS-D", convert_rrsis, Path(args.rrsis_root)))
    if args.rsvg_root:
        builders.append(("RSVG", convert_rsvg, Path(args.rsvg_root)))

    if not builders:
        raise ValueError("At least one dataset root must be specified.")

    all_entries: List[Dict] = []
    stats: Dict[str, int] = {}
    for name, fn, root in builders:
        if not root.exists():
            raise FileNotFoundError(f"{name} root not found: {root}")
        records = fn(root, output_root)
        stats[name] = len(records)
        all_entries.extend(records)

    all_entries.sort(
        key=lambda item: (
            item["source_dataset"],
            item["image"],
            item["conversations"][0]["value"],
        )
    )

    out_path = Path(args.output_list).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False)
    print(f"Wrote {len(all_entries)} samples to {out_path}")
    for name, count in stats.items():
        print(f"  - {name}: {count} samples")
    print(
        "Use --data_path {} --list_file {} when launching fine-tuning.".format(
            output_root, out_path
        )
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vrsbench-root", type=Path, help="Path to the VRSBench root.")
    parser.add_argument("--dior-rsvg-root", type=Path, help="Path to the DIOR-RSVG root.")
    parser.add_argument("--rrsis-root", type=Path, help="Path to the RRSIS-D root.")
    parser.add_argument("--rsvg-root", type=Path, help="Path to the RSVG root (folder that contains images/ and rsvg_train.pth).")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory where symlinks and images will be organized.")
    parser.add_argument("--output-list", type=Path, required=True, help="Path to the resulting JSON list file.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    build(parse_args(sys.argv[1:]))
