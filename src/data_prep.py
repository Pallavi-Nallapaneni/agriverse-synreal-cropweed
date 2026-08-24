import argparse
import collections
from pathlib import Path

import pandas as pd
import yaml


def load_dataset_yaml(yaml_path: Path) -> dict:
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["_root"] = (yaml_path.parent / cfg["path"]).resolve()

    return cfg


def check_split(root: Path, images_rel: str, class_names: list[str]) -> dict:
    img_dir = root / images_rel
    lbl_dir = root / images_rel.replace("images", "labels")

    if not img_dir.exists():
        return {
            "images": 0,
            "labels": 0,
            "missing_labels": [],
            "empty_labels": [],
            "class_counts": {}
        }

    images = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    class_counts = collections.Counter()
    missing_labels = []
    empty_labels = []

    for img_path in images:
        lbl_path = lbl_dir / (img_path.stem + ".txt")

        if not lbl_path.exists():
            missing_labels.append(img_path.name)
            continue

        lines = [
            l for l in lbl_path.read_text().splitlines()
            if l.strip()
        ]

        if not lines:
            empty_labels.append(img_path.name)

        for line in lines:
            parts = line.split()

            if len(parts) != 5:
                raise ValueError(
                    f"Malformed label line in {lbl_path}: {line}"
                )

            cls_id = int(parts[0])

            if cls_id < 0 or cls_id >= len(class_names):
                raise ValueError(
                    f"Class id {cls_id} out of range"
                )

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

    ap.add_argument(
        "--data",
        required=True,
        help="Path to dataset yaml"
    )

    args = ap.parse_args()

    yaml_path = Path(args.data)

    cfg = load_dataset_yaml(yaml_path)

    root = cfg["_root"]

    names = (
        list(cfg["names"].values())
        if isinstance(cfg["names"], dict)
        else cfg["names"]
    )

    print(f"Dataset: {yaml_path}")
    print(f"Root:    {root}")
    print(f"Classes: {names}\n")


    rows = []

    for split_key in ("train", "val", "test"):

        stats = check_split(
            root,
            cfg[split_key],
            names
        )

        row = {
            "split": split_key,
            "images": stats["images"],
            "labeled": stats["labels"]
        }

        for name in names:
            row[name] = stats["class_counts"].get(name,0)

        rows.append(row)


    df = pd.DataFrame(rows).set_index("split")


    print("Per-split instance counts:")
    print(df.to_string())


    total = df[names].sum()

    print("\nOverall class balance:")
    print(total.to_dict())


if __name__ == "__main__":
    main()