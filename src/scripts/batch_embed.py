"""
batch_embed.py

Embed a secret message in all images in a directory.

Usage:
    python src/scripts/batch_embed.py --input ./data --output ./stego --message "Your secret" --password mypass
    python src/scripts/batch_embed.py --input ./data --output ./stego --bpp 0.1 --password mypass
"""

import os
import sys
import argparse
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.preprocessing import load_img, img_to_grayscale
from src.core.lbp import compute_lbp_classification
from src.core.embedding import embed_message
from src.core.pseudorandom import password_to_seed, generate_pixel_coordinates

SUPPORTED_FORMATS = (".png", ".bmp", ".tif", ".tiff")

# Base Lorem Ipsum text (~3285 chars)
LOREM_BASE = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Phasellus commodo, quam in tincidunt semper, odio nulla lacinia massa, in lobortis est enim dapibus purus. Donec at semper risus. Mauris ullamcorper luctus velit eu faucibus. Phasellus fermentum convallis turpis, et aliquet odio tincidunt quis. Proin id consectetur magna. Donec tempus mauris leo, nec lacinia ligula congue nec. Pellentesque sollicitudin odio vel tellus congue semper. Maecenas porta consectetur erat non dapibus. Donec eu elit sit amet magna finibus porta. Nullam imperdiet eros sit amet diam vulputate, nec scelerisque nibh ornare. Quisque ut velit mi. Nulla condimentum, ex nec gravida rutrum, libero nibh venenatis nunc, eget fermentum urna nunc sit amet neque. Donec tempus hendrerit vehicula. Morbi id eleifend quam. Mauris sagittis turpis in nibh ullamcorper congue. Nunc et neque ligula. Nam tristique consequat arcu, eu placerat diam. Suspendisse id volutpat enim. Curabitur ac turpis eget turpis vehicula blandit non nec metus. Maecenas vel nulla faucibus, dapibus nunc sed, ullamcorper ipsum. Pellentesque in rhoncus erat. Sed venenatis magna urna, non volutpat ante porttitor nec. Donec at imperdiet diam. Aliquam ultrices mattis condimentum. Maecenas hendrerit tempus tincidunt. Curabitur velit turpis, pharetra a neque vel, rutrum cursus neque. Mauris egestas efficitur tortor, vitae eleifend nulla sagittis id. Mauris sem nulla, maximus at aliquet vitae, viverra vitae nisi. Vestibulum fermentum quam ac egestas interdum. Quisque finibus eros vel tempor interdum. In id leo et est posuere volutpat ut quis enim. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Nullam molestie velit mi, eget sagittis nisi feugiat id. Integer posuere nisl at eros convallis accumsan. Ut id urna sed dui dictum congue. Duis ullamcorper sem mauris, vel vestibulum urna tincidunt non. Etiam porta metus a pharetra feugiat. Nullam ut diam non ligula mattis laoreet. Nam luctus, sapien eget maximus lobortis, sapien nisi finibus nulla, id condimentum ex eros nec nisl. Praesent nec risus sollicitudin, euismod diam nec, accumsan urna. Ut arcu odio, placerat id purus a, ultricies aliquam libero. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Pellentesque sodales suscipit sollicitudin. Ut facilisis tincidunt aliquet. Donec tempus faucibus augue, eget lacinia risus finibus quis. Sed vel turpis a risus imperdiet dignissim nec id risus. Sed leo enim, sodales id sapien bibendum, consectetur auctor risus. Maecenas placerat bibendum quam, vitae dignissim velit eleifend vestibulum. Sed tempus ex sit amet aliquet euismod. In congue vulputate lacus sed aliquet. Proin ut facilisis massa. Curabitur egestas facilisis metus et imperdiet. Aliquam pretium fermentum sapien vitae congue. Curabitur sagittis odio est, quis porttitor leo tempor eget. Quisque nisi lacus, mattis nec dui ac, varius commodo felis. Nullam placerat mauris id elit laoreet, eget consequat leo fringilla. Cras vel neque bibendum, sagittis sem eget, maximus quam. Proin id risus justo. Donec sodales, sem at accumsan auctor, ligula felis placerat est, sit amet sollicitudin libero nunc ac leo. Suspendisse euismod magna orci, at egestas odio faucibus ac. Proin ut nam."


def generate_message_for_bpp(bpp: float, height: int, width: int) -> str:
    """Generate a message of the exact length needed for target BPP."""
    total_pixels = height * width
    target_bits = int(bpp * total_pixels)
    target_chars = target_bits // 8
    
    # Repeat Lorem Ipsum as needed
    repeats = (target_chars // len(LOREM_BASE)) + 1
    full_text = (LOREM_BASE + " ") * repeats
    
    return full_text[:target_chars]


def embed_single_image(input_path: str, output_path: str, message: str, password: str) -> dict:
    """Embed message in a single image and return stats."""
    # Load and prepare image
    rgb_img = load_img(input_path)
    height, width, _ = rgb_img.shape
    
    # Compute LBP
    grayscale = img_to_grayscale(rgb_img)
    classification_map = compute_lbp_classification(grayscale)
    
    # Generate pixel coordinates
    seed = password_to_seed(password)
    pixel_coords = generate_pixel_coordinates(height, width, seed)
    
    # Embed
    stego_img = embed_message(
        rgb_img=rgb_img,
        secret_message=message,
        password=password,
        classification_map=classification_map,
        pixel_coords=pixel_coords,
    )
    
    # Save
    cv2.imwrite(output_path, stego_img)
    
    # Calculate stats
    bits_embedded = len(message) * 8
    bpp = bits_embedded / (height * width)
    
    return {
        "filename": os.path.basename(input_path),
        "size": f"{width}x{height}",
        "chars": len(message),
        "bits": bits_embedded,
        "bpp": round(bpp, 4),
    }


def batch_embed(input_dir: str, output_dir: str, message: str = None, 
                bpp: float = None, password: str = "secret") -> None:
    """Embed message in all images in input_dir."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all supported images
    images = [f for f in os.listdir(input_dir) 
              if f.lower().endswith(SUPPORTED_FORMATS)]
    
    if not images:
        print(f"No supported images found in {input_dir}")
        print(f"Supported formats: {SUPPORTED_FORMATS}")
        return
    
    print(f"{'='*60}")
    print(f"  Batch Embedding: {len(images)} images")
    print(f"  Password: {password}")
    if bpp:
        print(f"  Target BPP: {bpp}")
    else:
        print(f"  Message length: {len(message)} chars")
    print(f"{'='*60}\n")
    
    results = []
    for i, filename in enumerate(images, 1):
        input_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + "_stego.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # If BPP mode, generate message for this image's size
        if bpp:
            img = load_img(input_path)
            h, w, _ = img.shape
            msg = generate_message_for_bpp(bpp, h, w)
        else:
            msg = message
        
        print(f"[{i}/{len(images)}] {filename}...", end=" ")
        
        try:
            stats = embed_single_image(input_path, output_path, msg, password)
            results.append(stats)
            print(f"✓ ({stats['bits']:,} bits, BPP={stats['bpp']})")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print(f"\n{'='*60}")
    print(f"  Completed: {len(results)}/{len(images)} images")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch embed secret message in all images"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory containing images")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for stego images")
    parser.add_argument("--message", type=str, default=None,
                        help="Secret message to embed")
    parser.add_argument("--bpp", type=float, default=None,
                        help="Target BPP (auto-generates message)")
    parser.add_argument("--password", type=str, default="secure_password_123",
                        help="Password for pseudorandom embedding")
    args = parser.parse_args()
    
    if args.bpp is None and args.message is None:
        parser.error("Either --message or --bpp is required")
    
    batch_embed(args.input, args.output, args.message, args.bpp, args.password)


if __name__ == "__main__":
    main()
