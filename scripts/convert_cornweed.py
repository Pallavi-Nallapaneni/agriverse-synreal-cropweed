import json
from pathlib import Path
import shutil


SRC = Path(r"C:\Users\palla\Downloads\cornweed_temp")

OUT_IMG = Path("data/real_target/images/test")
OUT_LBL = Path("data/real_target/labels/test")

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)


with open(SRC / "weedcoco.json", "r") as f:
    coco = json.load(f)


images = {img["id"]: img for img in coco["images"]}

annotations = {}

for ann in coco["annotations"]:
    annotations.setdefault(ann["image_id"], []).append(ann)


converted = 0

for img_id, img in images.items():

    filename = img["file_name"]

    src_img = SRC / "images" / filename

    if not src_img.exists():
        print("Missing:", filename)
        continue


    shutil.copy(
        src_img,
        OUT_IMG / filename
    )


    label_path = OUT_LBL / (Path(filename).stem + ".txt")

    labels = []

    for ann in annotations.get(img_id, []):

        cls = ann["category_id"]

        x, y, w, h = ann["bbox"]

        xc = (x + w / 2) / img["width"]
        yc = (y + h / 2) / img["height"]

        wn = w / img["width"]
        hn = h / img["height"]


        labels.append(
            f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"
        )


    with open(label_path, "w") as f:
        f.write("\n".join(labels))


    converted += 1


print("==========================")
print("Converted:", converted)
print("Images:", OUT_IMG)
print("Labels:", OUT_LBL)
print("==========================")