"""
convert.py

Convert images into multiple formats and organize them into folders.

Supported input formats:
    TIFF, JPG/JPEG

Behavior:
    - ALL images are converted into:
        - /tiff → .tiff
        - /png  → .png
        - /bmp  → .bmp

How to run:

    # Process all supported files in a directory
    python src/scripts/convert.py --input ./data/raw --output ./data

    # Process only JPG files
    python src/scripts/convert.py --input ./data/raw --output ./data --format jpg

    # Process only TIFF files
    python src/scripts/convert.py --input ./data/raw --output ./data --format tiff

    # Process a single file
    python src/scripts/convert.py --file ./data/raw/image.jpg --output ./data
"""

import os
import argparse
from PIL import Image

SUPPORTED_FORMATS = (".tif", ".tiff", ".jpg", ".jpeg")


def ensure_dirs(output_base: str):
    """Create required output directories."""
    paths = {
        "tiff": os.path.join(output_base, "tiff"),
        "png": os.path.join(output_base, "png"),
        "bmp": os.path.join(output_base, "bmp"),
    }

    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return paths


def prepare_image(img):
    """Ensure image is in a compatible mode."""
    if img.mode not in ("RGB", "L"):
        return img.convert("RGB")
    return img


def convert_to_all_formats(input_path: str, output_dirs: dict):
    """Convert one image into TIFF, PNG, and BMP."""
    filename = os.path.basename(input_path)
    name, _ = os.path.splitext(filename)

    try:
        with Image.open(input_path) as img:
            img = prepare_image(img)

            # TIFF
            img.save(os.path.join(output_dirs["tiff"], f"{name}.tiff"), "TIFF")

            # PNG
            img.save(os.path.join(output_dirs["png"], f"{name}.png"), "PNG")

            # BMP
            img.save(os.path.join(output_dirs["bmp"], f"{name}.bmp"), "BMP")

        print(f"Converted: {filename}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")


def convert_directory(input_dir: str, output_dir: str, format_filter: str = None):
    """Process all supported images in a directory."""
    output_dirs = ensure_dirs(output_dir)

    # Filter logic
    if format_filter:
        fmt = format_filter.lower().lstrip(".")

        if fmt in ("jpg", "jpeg"):
            extensions = (".jpg", ".jpeg")
        elif fmt in ("tif", "tiff"):
            extensions = (".tif", ".tiff")
        else:
            raise ValueError(f"Unsupported format filter: {format_filter}")
    else:
        extensions = SUPPORTED_FORMATS

    converted = 0

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)

        # skip folders
        if not os.path.isfile(input_path):
            continue

        if filename.lower().endswith(extensions):
            convert_to_all_formats(input_path, output_dirs)
            converted += 1

    print(f"\nTotal processed: {converted} file(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert images into TIFF, PNG, and BMP formats"
    )

    parser.add_argument("--input", type=str,
                        help="Input directory containing images")

    parser.add_argument("--file", type=str,
                        help="Single file to process")

    parser.add_argument("--output", required=True,
                        help="Base output directory (creates /tiff, /png, /bmp)")

    parser.add_argument("--format", type=str, default=None,
                        help="Filter by format (jpg or tiff)")

    args = parser.parse_args()

    if args.file:
        output_dirs = ensure_dirs(args.output)
        convert_to_all_formats(args.file, output_dirs)

    elif args.input:
        convert_directory(args.input, args.output, args.format)

    else:
        parser.error("Either --input or --file is required")
