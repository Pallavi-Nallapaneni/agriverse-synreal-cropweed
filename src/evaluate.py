"""
evaluate.py
-----------
Runs a trained YOLO checkpoint against a dataset split and reports
mAP50, mAP50-95, precision, recall, and F1 (overall + per-class),
saving the results as a CSV row so multiple runs can be compared later.

Usage:
    # Evaluate on the synthetic val/test split
    python src/evaluate.py --weights runs/detect/agriverse_synthetic/weights/best.pt \
        --data configs/agriverse_synthetic.yaml --split test --tag synthetic_test

    # Evaluate the SAME weights on the real target set (domain-gap check)
    python src/evaluate.py --weights runs/detect/agriverse_synthetic/weights/best.pt \
        --data configs/real_target.yaml --split test --tag real_test
"""

import argparse
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to trained .pt checkpoint")
    ap.add_argument("--data", required=True, help="Dataset yaml to evaluate on")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--tag", required=True, help="Short label for this run, e.g. 'synthetic_test' or 'real_test'")
    ap.add_argument("--out", default="results/metrics/eval_results.csv")
    args = ap.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split=args.split, imgsz=args.imgsz, plots=True)

    names = metrics.names  # {id: class_name}
    p, r, map50, map5095 = metrics.box.mp, metrics.box.mr, metrics.box.map50, metrics.box.map
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    row = {
        "tag": args.tag,
        "weights": args.weights,
        "data": args.data,
        "split": args.split,
        "precision": p,
        "recall": r,
        "f1": f1,
        "mAP50": map50,
        "mAP50-95": map5095,
    }

    # Per-class AP50 (metrics.box.ap50 is indexed in the same order as metrics.box.ap_class_index)
    for idx, cls_id in enumerate(metrics.box.ap_class_index):
        cls_name = names[int(cls_id)]
        row[f"AP50_{cls_name}"] = metrics.box.ap50[idx]

    print("\n=== Evaluation summary ===")
    for k, v in row.items():
        print(f"{k:>20}: {v}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if out_path.exists():
        df_existing = pd.read_csv(out_path)
        df_new = pd.concat([df_existing, df_new], ignore_index=True)
    df_new.to_csv(out_path, index=False)
    print(f"\nAppended results to {out_path}")


if __name__ == "__main__":
    main()