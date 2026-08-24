
import argparse
import random
import shutil
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import yaml


def load_dataset_yaml(yaml_path: Path) -> dict:
    """Load dataset YAML and resolve its dataset path."""

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["path"])

    if not root.is_absolute():
        root = yaml_path.parent / root

    cfg["_root"] = root.resolve()

    return cfg


def build_transform(strength="medium"):
    """Photometric-only augmentation; bounding boxes remain unchanged."""

    if strength == "light":
        p = 0.5
    elif strength == "heavy":
        p = 0.9
    else:
        p = 0.7

    return A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=p,
            ),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=25,
                val_shift_limit=15,
                p=p,
            ),
            A.OneOf(
                [
                    A.GaussNoise(),
                    A.ISONoise(
                        color_shift=(0.01, 0.05),
                        intensity=(0.1, 0.5),
                    ),
                ],
                p=p * 0.6,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.MotionBlur(blur_limit=(3, 7)),
                ],
                p=p * 0.4,
            ),
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=p * 0.5,
            ),
            A.RandomShadow(
                p=p * 0.3,
            ),
        ]
    )


def get_dirs(root, split):
    """Get image and label directories."""

    images_dir = root / split

    labels_dir = Path(
        str(images_dir).replace(
            "images",
            "labels",
            1,
        )
    )

    return images_dir, labels_dir


def main():
    parser = argparse.ArgumentParser(
        description="Create photometrically augmented synthetic training data."
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Dataset YAML file.",
    )

    parser.add_argument(
        "--out-suffix",
        default="_aug",
        help="Suffix for augmented directories.",
    )

    parser.add_argument(
        "--copies",
        type=int,
        default=2,
        help="Augmented copies per training image.",
    )

    parser.add_argument(
        "--strength",
        choices=["light", "medium", "heavy"],
        default="medium",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    yaml_path = args.data.resolve()
    cfg = load_dataset_yaml(yaml_path)
    root = cfg["_root"]

    # Dataset YAML normally contains:
    # train: images/train
    train_images = root / cfg["train"]
    train_labels = Path(
        str(train_images).replace("images", "labels", 1)
    )

    augmented_images = (
        root / "images" / f"train{args.out_suffix}"
    )

    augmented_labels = (
        root / "labels" / f"train{args.out_suffix}"
    )

    combined_images = (
        root / "images" / "train_combined"
    )

    combined_labels = (
        root / "labels" / "train_combined"
    )

    print("========================================")
    print("AgriVerse Synthetic Augmentation")
    print("========================================")
    print(f"Dataset root:     {root}")
    print(f"Training images:  {train_images}")
    print(f"Training labels:  {train_labels}")
    print(f"Copies/image:     {args.copies}")
    print(f"Strength:         {args.strength}")
    print()

    if not train_images.exists():
        raise FileNotFoundError(
            f"Training image directory does not exist:\n{train_images}"
        )

    if not train_labels.exists():
        raise FileNotFoundError(
            f"Training label directory does not exist:\n{train_labels}"
        )

    image_paths = sorted(
        p
        for p in train_images.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No training images found in {train_images}"
        )

    augmented_images.mkdir(parents=True, exist_ok=True)
    augmented_labels.mkdir(parents=True, exist_ok=True)
    combined_images.mkdir(parents=True, exist_ok=True)
    combined_labels.mkdir(parents=True, exist_ok=True)

    # Copy original training images/labels into combined dataset.
    original_count = 0

    for img_path in image_paths:
        label_path = train_labels / f"{img_path.stem}.txt"

        if not label_path.exists():
            print(f"WARNING: missing label: {label_path}")
            continue

        shutil.copy2(
            img_path,
            combined_images / img_path.name,
        )

        shutil.copy2(
            label_path,
            combined_labels / label_path.name,
        )

        original_count += 1

    print(
        f"Copied {original_count} original training images."
    )

    transform = build_transform(args.strength)

    augmented_count = 0

    for img_path in image_paths:
        label_path = train_labels / f"{img_path.stem}.txt"

        if not label_path.exists():
            continue

        image = cv2.imread(str(img_path))

        if image is None:
            print(f"WARNING: could not read {img_path}")
            continue

        for copy_idx in range(args.copies):
            result = transform(image=image)
            augmented = result["image"]

            filename = (
                f"{img_path.stem}_aug{copy_idx}{img_path.suffix}"
            )

            augmented_img = augmented_images / filename
            augmented_lbl = (
                augmented_labels
                / f"{img_path.stem}_aug{copy_idx}.txt"
            )

            cv2.imwrite(
                str(augmented_img),
                augmented,
            )

            shutil.copy2(
                label_path,
                augmented_lbl,
            )

            # Add augmented sample to combined training set.
            shutil.copy2(
                augmented_img,
                combined_images / filename,
            )

            shutil.copy2(
                augmented_lbl,
                combined_labels / augmented_lbl.name,
            )

            augmented_count += 1

    total = original_count + augmented_count

    print()
    print(f"Original images:   {original_count}")
    print(f"Augmented images:  {augmented_count}")
    print(f"Combined images:   {total}")

    # Create a new YAML.
    new_cfg = dict(cfg)
    new_cfg.pop("_root", None)
    new_cfg["train"] = "images/train_combined"

    output_yaml = (
        yaml_path.parent
        / f"{yaml_path.stem}{args.out_suffix}.yaml"
    )

    with open(
        output_yaml,
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            new_cfg,
            f,
            sort_keys=False,
        )

    print()
    print(f"Created: {output_yaml}")
    print()
    print("Validation and test sets were NOT modified.")
    print()
    print("Next training command:")
    print(
        "python src/train.py "
        f"--data {output_yaml} "
        "--model yolov8n.pt "
        "--epochs 100 "
        "--imgsz 1024 "
        "--batch 8 "
        "--name agriverse_aug_yolov8n"
    )


if __name__ == "__main__":
    main()