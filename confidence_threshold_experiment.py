from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path(
    "runs/detect/runs/detect/agriverse_aug_yolov8n-5/weights/best.pt"
)

IMAGE_DIR = Path(
    "data/agriverse/images/test"
)

LABEL_DIR = Path(
    "data/agriverse/labels/test"
)

IMG_SIZE = 1024
IOU_THRESHOLD = 0.50

THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
]


def yolo_to_xyxy(xc, yc, w, h):
    return [
        xc - w / 2,
        yc - h / 2,
        xc + w / 2,
        yc + h / 2,
    ]


def iou(a, b):

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)

    intersection = iw * ih

    area_a = (
        max(0, a[2] - a[0])
        * max(0, a[3] - a[1])
    )

    area_b = (
        max(0, b[2] - b[0])
        * max(0, b[3] - b[1])
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def read_gt(label_path):

    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path) as f:

        for line_number, line in enumerate(f):

            parts = line.strip().split()

            if len(parts) < 5:
                continue

            cls = int(parts[0])

            # Exclude the one known malformed annotation.
            if (
                label_path.name
                == "_home_iman_Documents_big_dataset_rgb_3531.txt"
                and line_number == 0
            ):
                continue

            # Weed class only.
            if cls != 1:
                continue

            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            # Exclude zero-area boxes.
            if w <= 0 or h <= 0:
                continue

            boxes.append(
                yolo_to_xyxy(
                    xc,
                    yc,
                    w,
                    h,
                )
            )

    return boxes


def evaluate_threshold(model, threshold):

    tp = 0
    fp = 0
    fn = 0

    for image_path in sorted(IMAGE_DIR.glob("*")):

        if image_path.suffix.lower() not in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
        ]:
            continue

        label_path = (
            LABEL_DIR
            / f"{image_path.stem}.txt"
        )

        gt_boxes = read_gt(label_path)

        results = model.predict(
            source=str(image_path),
            imgsz=IMG_SIZE,
            conf=threshold,
            verbose=False,
            device="cpu",
        )

        result = results[0]

        predictions = []

        if result.boxes is not None:

            for box, cls, conf in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
            ):

                # Weed = class 1.
                if int(cls) != 1:
                    continue

                if conf < threshold:
                    continue

                x1, y1, x2, y2 = box

                # Convert prediction from pixel coordinates
                # to normalized 0-1 coordinates.
                predictions.append([
                    x1 / result.orig_shape[1],
                    y1 / result.orig_shape[0],
                    x2 / result.orig_shape[1],
                    y2 / result.orig_shape[0],
                ])

        matches = []

        for pi, pred_box in enumerate(predictions):

            for gi, gt_box in enumerate(gt_boxes):

                score = iou(
                    pred_box,
                    gt_box,
                )

                if score >= IOU_THRESHOLD:

                    matches.append(
                        (score, pi, gi)
                    )

        matches.sort(
            reverse=True
        )

        matched_predictions = set()
        matched_gt = set()

        for score, pi, gi in matches:

            if pi in matched_predictions:
                continue

            if gi in matched_gt:
                continue

            matched_predictions.add(pi)
            matched_gt.add(gi)

            tp += 1

        fp += (
            len(predictions)
            - len(matched_predictions)
        )

        fn += (
            len(gt_boxes)
            - len(matched_gt)
        )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
    }


def main():

    print()
    print("=" * 70)
    print("WEED CONFIDENCE-THRESHOLD EXPERIMENT")
    print("=" * 70)

    print()
    print("Model:", MODEL_PATH)
    print("Image size:", IMG_SIZE)
    print("IoU threshold:", IOU_THRESHOLD)
    print("Excluded annotation:")
    print("  3531.txt, line 0")
    print()

    model = YOLO(
        str(MODEL_PATH)
    )

    results = []

    for threshold in THRESHOLDS:

        print(
            f"Running confidence threshold "
            f"{threshold:.2f}..."
        )

        result = evaluate_threshold(
            model,
            threshold,
        )

        results.append(result)

        print(
            f"  TP={result['TP']} "
            f"FP={result['FP']} "
            f"FN={result['FN']} "
            f"Precision={result['precision']:.4f} "
            f"Recall={result['recall']:.4f} "
            f"F1={result['F1']:.4f}"
        )

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"{'Conf':>8}"
        f"{'TP':>8}"
        f"{'FP':>8}"
        f"{'FN':>8}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
    )

    print("-" * 70)

    for result in results:

        print(
            f"{result['threshold']:>8.2f}"
            f"{result['TP']:>8}"
            f"{result['FP']:>8}"
            f"{result['FN']:>8}"
            f"{result['precision']:>12.4f}"
            f"{result['recall']:>12.4f}"
            f"{result['F1']:>12.4f}"
        )


if __name__ == "__main__":
    main()
