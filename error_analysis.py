from pathlib import Path
from collections import defaultdict
import csv

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
GT_DIR = Path("data/agriverse/labels/test")
PRED_DIR = Path(
    "runs/detect/runs/detect/error_analysis_raw/labels"
)

# 0 = maize, 1 = weed
CLASS_NAMES = {
    0: "maize",
    1: "weed",
}

IOU_THRESHOLD = 0.50


# ---------------------------------------------------------
# Convert YOLO box to xyxy
# ---------------------------------------------------------
def yolo_to_xyxy(xc, yc, w, h):
    return [
        xc - w / 2,
        yc - h / 2,
        xc + w / 2,
        yc + h / 2,
    ]


# ---------------------------------------------------------
# IoU
# ---------------------------------------------------------
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_w = max(0.0, x2 - x1)
    intersection_h = max(0.0, y2 - y1)
    intersection = intersection_w * intersection_h

    area1 = max(0.0, box1[2] - box1[0]) * max(
        0.0, box1[3] - box1[1]
    )
    area2 = max(0.0, box2[2] - box2[0]) * max(
        0.0, box2[3] - box2[1]
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ---------------------------------------------------------
# Read YOLO annotation/prediction
# ---------------------------------------------------------
def read_boxes(path, predictions=False):
    boxes = []

    if not path.exists():
        return boxes

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if not parts:
                continue

            cls = int(parts[0])

            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            box = yolo_to_xyxy(xc, yc, w, h)

            confidence = float(parts[5]) if predictions else None

            boxes.append({
                "class": cls,
                "box": box,
                "confidence": confidence,
            })

    return boxes


# ---------------------------------------------------------
# Global summary
# ---------------------------------------------------------
summary = {
    0: {"tp": 0, "fp": 0, "fn": 0},
    1: {"tp": 0, "fp": 0, "fn": 0},
}

image_results = []
missed_objects = []


# ---------------------------------------------------------
# Process each test image
# ---------------------------------------------------------
for gt_file in sorted(GT_DIR.glob("*.txt")):

    image_name = gt_file.stem
    pred_file = PRED_DIR / gt_file.name

    gt_boxes = read_boxes(gt_file, predictions=False)
    pred_boxes = read_boxes(pred_file, predictions=True)

    # Per-image counters MUST be initialized before matching
    image_tp = defaultdict(int)
    image_fp = defaultdict(int)
    image_fn = defaultdict(int)

    matched_gt = set()
    matched_pred = set()

    # -----------------------------------------------------
    # Build possible class-correct matches
    # -----------------------------------------------------
    possible_matches = []

    for pi, pred in enumerate(pred_boxes):

        for gi, gt in enumerate(gt_boxes):

            if pred["class"] != gt["class"]:
                continue

            iou = calculate_iou(
                pred["box"],
                gt["box"]
            )

            if iou >= IOU_THRESHOLD:
                possible_matches.append(
                    (iou, pi, gi)
                )

    # Highest IoU matches first
    possible_matches.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # -----------------------------------------------------
    # One-to-one matching
    # -----------------------------------------------------
    for iou, pi, gi in possible_matches:

        if pi in matched_pred:
            continue

        if gi in matched_gt:
            continue

        matched_pred.add(pi)
        matched_gt.add(gi)

        cls = gt_boxes[gi]["class"]

        summary[cls]["tp"] += 1
        image_tp[cls] += 1

    # -----------------------------------------------------
    # False negatives
    # -----------------------------------------------------
    for gi, gt in enumerate(gt_boxes):

        if gi not in matched_gt:

            cls = gt["class"]

            summary[cls]["fn"] += 1
            image_fn[cls] += 1

            missed_objects.append({
                "image": image_name,
                "class": CLASS_NAMES[cls],
                "gt_index": gi,
            })

    # -----------------------------------------------------
    # False positives
    # -----------------------------------------------------
    for pi, pred in enumerate(pred_boxes):

        if pi not in matched_pred:

            cls = pred["class"]

            summary[cls]["fp"] += 1
            image_fp[cls] += 1

    # -----------------------------------------------------
    # Store per-image results
    # -----------------------------------------------------
    image_results.append({
        "image": image_name,

        "maize_gt": sum(
            1 for x in gt_boxes
            if x["class"] == 0
        ),
        "maize_tp": image_tp[0],
        "maize_fn": image_fn[0],
        "maize_fp": image_fp[0],

        "weed_gt": sum(
            1 for x in gt_boxes
            if x["class"] == 1
        ),
        "weed_tp": image_tp[1],
        "weed_fn": image_fn[1],
        "weed_fp": image_fp[1],
    })


# ---------------------------------------------------------
# Print global metrics
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("YOLO ERROR ANALYSIS")
print("=" * 60)

for cls in [0, 1]:

    name = CLASS_NAMES[cls]

    tp = summary[cls]["tp"]
    fp = summary[cls]["fp"]
    fn = summary[cls]["fn"]

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    print(f"\n{name.upper()}")
    print("-" * 40)
    print(f"TP:        {tp}")
    print(f"FP:        {fp}")
    print(f"FN:        {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")


# ---------------------------------------------------------
# Missed weeds
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("MISSED WEEDS")
print("=" * 60)

missed_weeds = [
    x for x in missed_objects
    if x["class"] == "weed"
]

for item in missed_weeds:
    print(
        f"{item['image']}  "
        f"GT weed #{item['gt_index']}"
    )

print(
    f"\nTotal missed weeds: "
    f"{len(missed_weeds)}"
)


# ---------------------------------------------------------
# Save per-image CSV
# ---------------------------------------------------------
output_csv = Path(
    "error_analysis_per_image.csv"
)

with open(
    output_csv,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=image_results[0].keys()
    )

    writer.writeheader()
    writer.writerows(image_results)

print(
    f"\nPer-image results saved to: "
    f"{output_csv}"
)
