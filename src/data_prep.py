"""
data_prep.py
------------
Sanity-checks a YOLO-format detection dataset (image/label pairing,
label formatting) and reports per-split class distribution.

Usage:
    python src/data_prep.py --data configs/agriverse_synthetic.yaml
    python src/data_prep.py --data configs/real_target.yaml
"""

import argparse
import collections
from pathlib import Path

import pandas as pd
import yaml


def load_dataset_yaml(yaml_path: Path) -> dict:
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    # `path` is resolved relative to the yaml file's own directory
    cfg["_root"] = (yaml_path.parent / cfg["path"]).resolve()
    return cfg


def check_split(root: Path, images_rel: str, class_names: list[str]) -> dict:
    img_dir = root / images_rel
    lbl_dir = root / images_rel.replace("images", "labels")

    if not img_dir.exists():
        return {"images": 0, "labels": 0, "missing_labels": [], "class_counts": {}}

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    class_counts = collections.Counter()
    missing_labels = []
    empty_labels = []

    for img_path in images:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            missing_labels.append(img_path.name)
            continue
        lines = [l for l in lbl_path.read_text().splitlines() if l.strip()]
        if not lines:
            empty_labels.append(img_path.name)
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Malformed label line in {lbl_path}: {line!r}")
            cls_id = int(parts[0])
            if not (0 <= cls_id < len(class_names)):
                raise ValueError(f"Class id {cls_id} out of range in {lbl_path}")
            class_counts[class_names[cls_id]] += 1

    return {
        "images": len(images),
        "labels": len(images) - len(missing_labels),
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "class_counts": dict(class_counts),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to a dataset yaml (e.g. configs/agriverse_synthetic.yaml)")
    args = ap.parse_args()

    yaml_path = Path(args.data)
    cfg = load_dataset_yaml(yaml_path)
    root = cfg["_root"]
    names = cfg["names"]

    print(f"Dataset: {yaml_path}")
    print(f"Root:    {root}")
    print(f"Classes: {names}\n")

    rows = []
    for split_key in ("train", "val", "test"):
        if split_key not in cfg:
            continue
        stats = check_split(root, cfg[split_key], names)
        row = {"split": split_key, "images": stats["images"], "labeled": stats["labels"]}
        for name in names:
            row[name] = stats["class_counts"].get(name, 0)
        rows.append(row)

        if stats["missing_labels"]:
            print(f"[WARN] {split_key}: {len(stats['missing_labels'])} images with no label file")
        if stats.get("empty_labels"):
            print(f"[WARN] {split_key}: {len(stats['empty_labels'])} images with an empty label file (background-only)")

    df = pd.DataFrame(rows).set_index("split")
    print("\nPer-split instance counts:")
    print(df.to_string())

    total = df[names].sum()
    ratio = total.max() / max(total.min(), 1)
    print(f"\nOverall class balance: {total.to_dict()}")
    if ratio >= 3:
        print(
            f"[NOTE] Class imbalance ~{ratio:.1f}:1 between most/least frequent class. "
            "Consider class-weighted loss, focal loss, or reporting per-class mAP "
            "rather than only an aggregate mAP."
        )


if __name__ == "__main__":
    main()