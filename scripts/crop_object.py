import os
import click
import json
from PIL import Image
from tqdm import tqdm

SUPPORTED_CLASSESS = ["pedestrian", "car", "bus", "truck"]


def process(
    dataset: str,
    subset: str,
    output: str,
    split: str,
    img_size: int,
    min_object_size: int,
):
    with open(os.path.join(dataset, f"labels/det_20/det_{split}.json")) as f:
        data = json.load(f)

    labels = []

    os.makedirs(os.path.join(output, f"{split}/images"), exist_ok=True)

    for item in tqdm(data, total=len(data), desc=f"Processing {split}"):
        if "labels" not in item:
            continue

        for box in item["labels"]:
            if (
                box["category"] not in SUPPORTED_CLASSESS
                or box["attributes"]["occluded"]
                or box["attributes"]["truncated"]
            ):
                continue
            x1, y1, x2, y2 = (
                box["box2d"]["x1"],
                box["box2d"]["y1"],
                box["box2d"]["x2"],
                box["box2d"]["y2"],
            )
            if x2 - x1 < min_object_size or y2 - y1 < min_object_size:
                continue
            img = Image.open(
                os.path.join(dataset, "images", subset, split, item["name"])
            )
            img = img.crop((x1, y1, x2, y2))
            img = img.resize((img_size, img_size), Image.Resampling.NEAREST)

            name = str(len(labels)).zfill(6) + ".jpg"

            img.save(os.path.join(output, f"{split}/images/{name}"))

            labels.append(
                {
                    "name": name,
                    "timeofday": item["attributes"]["timeofday"],
                    "weather": item["attributes"]["weather"],
                    "category": box["category"],
                }
            )

    with open(os.path.join(output, f"{split}/labels.json"), "w") as f:
        json.dump(labels, f)


@click.command()
@click.option("--dataset", type=str)
@click.option("--subset", type=str, default="100k")
@click.option("--output", type=str)
@click.option("--img_size", type=int, default=256)
@click.option("--min_object_size", type=int, default=64)
def crop_object(
    dataset: str, subset: str, output: str, img_size: int, min_object_size: int
):
    process(dataset, subset, output, "val", img_size, min_object_size)
    process(dataset, subset, output, "train", img_size, min_object_size)


if __name__ == "__main__":
    crop_object()
