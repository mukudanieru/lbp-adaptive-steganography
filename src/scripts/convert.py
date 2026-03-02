"""
convert.py

Convert images to PNG format. Supports TIFF, JPG/JPEG, BMP, GIF, WEBP.

How to run:

    # Convert all supported formats in a directory
    python src/scripts/convert.py --input ./data/raw --output ./data

    # Convert only JPG files
    python src/scripts/convert.py --input ./data/raw --output ./data --format jpg

    # Convert a single file
    python src/scripts/convert.py --file image.jpg --output ./data
"""

import os
import argparse
from PIL import Image

# Supported input formats
SUPPORTED_FORMATS = (".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")


def convert_to_png(input_path: str, output_dir: str) -> str:
    """Convert a single image to PNG format."""
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.basename(input_path)
    output_filename = os.path.splitext(filename)[0] + ".png"
    output_path = os.path.join(output_dir, output_filename)

    with Image.open(input_path) as img:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(output_path, "PNG")

    print(f"Converted: {filename} → {output_filename}")
    return output_path


def convert_directory(input_dir: str, output_dir: str, format_filter: str = None) -> None:
    """Convert all supported images in a directory to PNG format."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which extensions to look for
    if format_filter:
        extensions = tuple(f".{format_filter.lower().lstrip('.')}")
        if format_filter.lower() in ("jpg", "jpeg"):
            extensions = (".jpg", ".jpeg")
        elif format_filter.lower() in ("tif", "tiff"):
            extensions = (".tif", ".tiff")
    else:
        extensions = SUPPORTED_FORMATS

    converted = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(extensions):
            input_path = os.path.join(input_dir, filename)
            convert_to_png(input_path, output_dir)
            converted += 1

    print(f"\nTotal converted: {converted} file(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert images to PNG format"
    )
    parser.add_argument("--input", type=str,
                        help="Input directory containing images")
    parser.add_argument("--file", type=str,
                        help="Single file to convert")
    parser.add_argument("--output", required=True,
                        help="Output directory for PNG files")
    parser.add_argument("--format", type=str, default=None,
                        help="Filter by format (e.g., jpg, tiff)")
    args = parser.parse_args()

    if args.file:
        convert_to_png(args.file, args.output)
    elif args.input:
        convert_directory(args.input, args.output, args.format)
    else:
        parser.error("Either --input or --file is required")
