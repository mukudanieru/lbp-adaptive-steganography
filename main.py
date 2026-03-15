"""
Main entry point for Adaptive Texture-Based LSB Steganography
(Hardcoded configuration version)
"""

import cv2
import numpy as np
from pathlib import Path

from src.core.preprocessing import load_img
from src.core.lbp import compute_lbp_classification
from src.core.embedding import embed_message
from src.core.extraction import extract_message
from src.core.pseudorandom import password_to_seed, generate_pixel_coordinates


# HARD-CODED CONFIGURATION
SECRET_MESSAGE = "Hello from adaptive LSB steganography!"
PASSWORD = "Jonathan"

cover_path = Path("./data/cover")
stego_path = Path("./data/stego")

stego_path.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 50)
    print("ADAPTIVE TEXTURE-BASED LSB STEGANOGRAPHY")
    print("=" * 50)

    while True:
        print("\n[1] - EMBED MESSAGE")
        print("[2] - EXTRACT MESSAGE")
        print("[3] - CHECK IMAGES")
        print("[4] - EXIT")

        try:
            prompt = int(input("\n>> ").strip())
        except ValueError:
            print("\n[!] Invalid input. Please enter a number.")
            continue

        if prompt == 1:
            embed_workflow()

        elif prompt == 2:
            extract_workflow()

        elif prompt == 3:
            check_images()

        elif prompt == 4:
            print("\n[*] Exiting...")
            break

        else:
            print("\n[!] Option not recognized.")


def get_image_paths(directory: Path) -> list[Path]:
    """Get all image file paths from a directory."""
    return [img for img in directory.iterdir() if img.is_file()]


def select_image(image_paths: list[Path], title: str) -> Path | None:
    """Display image selection menu and return selected image path."""
    print(f"\n{title}")

    for i in range(len(image_paths)):
        print(f"[{i + 1}] - {image_paths[i].name}")
    print(f"[{len(image_paths) + 1}] - Back")

    while True:
        try:
            choice = int(input("\nSelect >> ").strip())
        except ValueError:
            print("\n[!] Invalid input. Please enter a number.")
            continue

        if choice == len(image_paths) + 1:
            return None
        elif 1 <= choice <= len(image_paths):
            return image_paths[choice - 1]
        else:
            print("\n[!] Option not recognized.")


def embed_workflow():
    """Handle the embedding workflow."""
    cover_images = get_image_paths(cover_path)

    if not cover_images:
        print("\n[!] No cover images found in ./data/cover/")
        return

    # Select cover image
    selected_image = select_image(cover_images, "COVER IMAGE SELECTION TO EMBED:")
    if selected_image is None:
        return

    # Load and process image
    rgb_img: np.ndarray = load_img(selected_image)
    print(f"\n[+] Loading {selected_image.name}...")

    height, width, _ = rgb_img.shape
    print(f"[+] Image size: {height}x{width}")

    # Get secret message
    message = input("\nSecret message: ").strip()
    if not message:
        print("\n[!] Message cannot be empty.")
        return

    # Get password
    password = input("Password: ").strip()
    if not password:
        print("\n[!] Password cannot be empty.")
        return

    print("[+] Computing LBP classification map...")
    classification_map: np.ndarray = compute_lbp_classification(rgb_img)

    print("[+] Generating pixel coordinates from password...")
    seed: int = password_to_seed(password)
    pixel_coords = generate_pixel_coordinates(height, width, seed)

    print("[+] Embedding message...")
    stego_img: np.ndarray = embed_message(
        rgb_img=rgb_img,
        secret_message=message,
        classification_map=classification_map,
        pixel_coords=pixel_coords,
    )

    # Save stego image
    output_img_path = stego_path / selected_image.name
    print(f"\n[+] Saving stego image to {output_img_path}...")
    success = cv2.imwrite(str(output_img_path), stego_img)

    if not success:
        raise IOError("Failed to save output image.")

    print("\n[✓] Message successfully embedded!")
    print(f"[✓] Output saved to: {output_img_path}")
    print(f"[✓] Message length: {len(message)} characters")


def extract_workflow():
    """Handle the extraction workflow."""
    stego_images = get_image_paths(stego_path)

    if not stego_images:
        print("\n[!] No stego images found in ./data/stego/")
        return

    # Select stego image
    selected_image = select_image(stego_images, "STEGO IMAGE SELECTION TO EXTRACT:")
    if selected_image is None:
        return

    output_img_path = stego_path / selected_image.name
    print(output_img_path)

    # Load image
    print(f"\n[+] Loading {selected_image.name}...")
    stego_img: np.ndarray = load_img(str(output_img_path))

    height, width, _ = stego_img.shape
    print(f"[+] Image size: {height}x{width}")

    # Get password
    password = input("\nPassword: ").strip()
    if not password:
        print("\n[!] Password cannot be empty.")
        return

    print("[+] Computing LBP classification map...")
    classification_map: np.ndarray = compute_lbp_classification(stego_img)

    print("[+] Generating pixel coordinates from password...")
    seed: int = password_to_seed(password)
    pixel_coords = generate_pixel_coordinates(height, width, seed)

    print("[+] Extracting message from stego image...")
    try:
        extracted_message: str = extract_message(
            stego_image=stego_img,
            classification_map=classification_map,
            pixel_coords=pixel_coords,
        )

        print("\n[✓] Extraction successful!")
        print(f"[✓] Extracted message: {extracted_message}")
        print(f"[✓] Message length: {len(extracted_message)} characters")

    except Exception as e:
        print(f"\n[!] Extraction failed: {str(e)}")


def check_images():
    """Display available cover and stego images."""
    cover_images = get_image_paths(cover_path)
    stego_images = get_image_paths(stego_path)

    print("\n" + "=" * 50)
    print("COVER IMAGES (./data/cover/):")
    print("=" * 50)
    if cover_images:
        for i, img in enumerate(cover_images, 1):
            print(f"  [{i}] {img.name}")
    else:
        print("  (No cover images found)")

    print("\n" + "=" * 50)
    print("STEGO IMAGES (./data/stego/):")
    print("=" * 50)
    if stego_images:
        for i, img in enumerate(stego_images, 1):
            print(f"  [{i}] {img.name}")
    else:
        print("  (No stego images found)")
    print("=" * 50)


if __name__ == "__main__":
    main()
