"""
train.py
--------
Trains a YOLO detector on the synthetic AgriVerse crop/weed dataset.

Usage:
    python src/train.py --data configs/agriverse_synthetic.yaml \
        --model yolov8n.pt --epochs 100 --imgsz 1024 --batch 8 \
        --name agriverse_synthetic_yolov8n

Notes:
- Start from a small pretrained checkpoint (yolov8n/s) rather than random
  init — the sample is only 120 training images, so transfer learning
  matters a lot here. Once you have the full 6,500-image set, a larger
  backbone (yolov8m/l) becomes worth trying.
- `imgsz=1024` matches the native render resolution; drop to 640 for faster
  iteration during development.
"""

import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dataset yaml, e.g. configs/agriverse_synthetic.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="Starting checkpoint or model yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=20, help="Early-stopping patience")
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="agriverse_synthetic")
    ap.add_argument("--device", default=None, help="e.g. 0, 0,1, or cpu — default lets Ultralytics choose")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=args.project,
        name=args.name,
        device=args.device,
        plots=True,
    )


if __name__ == "__main__":
    main()
