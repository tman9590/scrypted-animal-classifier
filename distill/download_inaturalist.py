#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
from pathlib import Path

from catalog import USER_AGENT, api_get, read_catalog


ROOT = Path(__file__).parents[1]


def image_url(photo: dict) -> str:
    return photo["url"].replace("square", "large")


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download licensed North Carolina iNaturalist observations")
    parser.add_argument("--catalog", type=Path, default=ROOT / "work" / "north-carolina-species.json")
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "source-images")
    parser.add_argument("--per-species", type=int, default=30)
    args = parser.parse_args()
    metadata, species = read_catalog(args.catalog)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []

    for class_index, item in enumerate(species):
        class_dir = args.output / f"{class_index:03d}-{item.taxon_id}"
        class_dir.mkdir(exist_ok=True)
        payload = api_get(
            "observations",
            {
                "place_id": metadata["inaturalist_place_id"],
                "taxon_id": item.taxon_id,
                "quality_grade": "research",
                "photos": "true",
                "photo_license": "cc0,cc-by",
                "per_page": min(args.per_species * 3, 200),
                "order_by": "votes",
                "order": "desc",
            },
        )
        saved = len(list(class_dir.glob("*.jpg")))
        for observation in payload.get("results", []):
            if saved >= args.per_species:
                break
            photos = observation.get("photos") or []
            if not photos:
                continue
            photo = photos[0]
            license_code = (photo.get("license_code") or "").lower()
            if license_code not in {"cc0", "cc-by"}:
                continue
            destination = class_dir / f"{observation['id']}-{photo['id']}.jpg"
            if not destination.exists():
                try:
                    download(image_url(photo), destination)
                except Exception as error:
                    print(f"skip {image_url(photo)}: {error}")
                    continue
                time.sleep(0.1)
            rows.append(
                {
                    "file": str(destination.relative_to(args.output)),
                    "class_index": class_index,
                    "taxon_id": item.taxon_id,
                    "scientific_name": item.scientific_name,
                    "common_name": item.common_name,
                    "observation_url": f"https://www.inaturalist.org/observations/{observation['id']}",
                    "photo_url": image_url(photo),
                    "license": license_code,
                    "attribution": photo.get("attribution") or "",
                }
            )
            saved += 1
        print(f"{item.label}: {saved}/{args.per_species}")

    with (args.output / "attribution.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys() if rows else ["file"])
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "metadata.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
