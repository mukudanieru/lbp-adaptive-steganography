"""
convert.py

One-off script to convert all .tif/.tiff images in a directory to .png format.

How to run:

    python src/scripts/convert.py --input ./data/raw --output ./data
"""

import os
import argparse
from PIL import Image


def convert_tiff_to_png(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".tif", ".tiff")):
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_dir, output_filename)

            with Image.open(input_path) as img:
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                img.save(output_path, "PNG")

            print(f"Converted: {filename} → {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    convert_tiff_to_png(args.input, args.output)
