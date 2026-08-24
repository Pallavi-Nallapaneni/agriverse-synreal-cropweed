"""
visualize_dataset.py
---------------------
Draws YOLO-format bounding boxes on a handful of sample images so you can
visually sanity-check the annotations before training.

Usage:
    python src/visualize_dataset.py --data configs/agriverse_synthetic.yaml \
        --split train --n 6 --out results/figures/dataset_preview.png
"""

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import yaml

# Distinct BGR colors per class (extend if you add classes)
CLASS_COLORS = [(60, 200, 60), (40, 40, 230)]  # maize=green, weed=red


def load_dataset_yaml(yaml_path: Path) -> dict:
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = (yaml_path.parent / cfg["path"]).resolve()
    return cfg


def draw_boxes(img_path: Path, lbl_path: Path, class_names: list[str]):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    if lbl_path.exists():
        for line in lbl_path.read_text().splitlines():
            if not line.strip():
                continue
            cls_id, xc, yc, bw, bh = line.split()
            cls_id = int(cls_id)
            xc, yc, bw, bh = (float(v) for v in (xc, yc, bw, bh))
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, class_names[cls_id], (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=6, help="Number of sample images to show")
    ap.add_argument("--out", default="results/figures/dataset_preview.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_dataset_yaml(Path(args.data))
    root = cfg["_root"]
    names = list(cfg["names"].values())

    img_dir = root / cfg[args.split]
    lbl_dir = root / cfg[args.split].replace("images", "labels")

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    random.seed(args.seed)
    sample = random.sample(images, min(args.n, len(images)))

    cols = 3
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten() if len(sample) > 1 else [axes]

    for ax, img_path in zip(axes, sample):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        vis = draw_boxes(img_path, lbl_path, names)
        ax.imshow(vis)
        ax.set_title(img_path.name, fontsize=8)
        ax.axis("off")

    for ax in axes[len(sample):]:
        ax.axis("off")

    fig.suptitle(f"{args.split} split — sample annotations", fontsize=14)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()