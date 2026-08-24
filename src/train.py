
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data",
        required=True,
        help="Dataset YAML file",
    )

    ap.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Starting checkpoint or model",
    )

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=20)

    ap.add_argument(
        "--project",
        default="runs/detect",
    )

    ap.add_argument(
        "--name",
        default="agriverse_synthetic",
    )

    ap.add_argument(
        "--device",
        default=None,
        help="0 for GPU or cpu for CPU",
    )

    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the supplied checkpoint",
    )

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
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

