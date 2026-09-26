#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

from PIL import Image

from catalog import read_catalog
from teacher import BioClipClassifier, MegaDetector, crop_box


ROOT = Path(__file__).parents[1]


def yolo_box(box, width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pseudo-label licensed iNaturalist images with MDV6 and BioCLIP 2")
    parser.add_argument("--catalog", type=Path, default=ROOT / "work" / "north-carolina-species.json")
    parser.add_argument("--source", type=Path, default=ROOT / "work" / "source-images")
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "yolo-dataset")
    parser.add_argument("--megadetector", type=Path, default=ROOT / "work" / "models" / "MDV6-yolov10-c.pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--detection-threshold", type=float, default=0.25)
    parser.add_argument("--unknown-threshold", type=float, default=0.25)
    args = parser.parse_args()

    metadata, species = read_catalog(args.catalog)
    detector = MegaDetector(args.megadetector, args.device)
    classifier = BioClipClassifier(species, args.device, ROOT / "work" / "models")
    labels = [item.label for item in species] + ["unknown"]
    unknown_index = len(species)
    rows = list(csv.DictReader((args.source / "attribution.csv").open()))
    random.Random(360).shuffle(rows)

    for row_number, row in enumerate(rows):
        source = args.source / row["file"]
        try:
            image = Image.open(source).convert("RGB")
        except Exception as error:
            print(f"skip {source}: {error}")
            continue
        boxes = detector.detect(image, args.detection_threshold)
        crops = [crop_box(image, box) for box, _ in boxes]
        predictions = classifier.classify(crops)
        split = "val" if row_number % 5 == 0 else "train"
        image_dir = args.output / "images" / split
        label_dir = args.output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        output_image = image_dir / source.name
        annotations = []
        for (box, _), probabilities in zip(boxes, predictions):
            predicted_label, confidence = max(probabilities.items(), key=lambda item: item[1])
            expected_index = int(row["class_index"])
            class_index = expected_index if predicted_label == species[expected_index].label and confidence >= args.unknown_threshold else unknown_index
            annotations.append((class_index, *yolo_box(box, image.width, image.height)))
        if not annotations:
            continue
        shutil.copy2(source, output_image)
        (label_dir / f"{source.stem}.txt").write_text(
            "".join(f"{index} {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n" for index, x, y, w, h in annotations)
        )

    yaml = [f"path: {args.output.resolve()}", "train: images/train", "val: images/val", "names:"]
    yaml.extend(f"  {index}: {json.dumps(label)}" for index, label in enumerate(labels))
    (args.output / "dataset.yaml").write_text("\n".join(yaml) + "\n")
    print(f"Wrote training dataset to {args.output}")


if __name__ == "__main__":
    main()
