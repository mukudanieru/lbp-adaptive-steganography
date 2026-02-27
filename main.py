"""
Main entry point for Adaptive Texture-Based LSB Steganography
(Hardcoded configuration version)
"""

import cv2
import numpy as np

from src.core.preprocessing import load_img, img_to_grayscale
from src.core.lbp import compute_lbp_classification
from src.core.embedding import embed_message
from src.core.extraction import extract_message
from src.core.pseudorandom import password_to_seed, generate_pixel_coordinates


# HARD-CODED CONFIGURATION
INPUT_IMAGE_PATH = "./data/peppers.png"
OUTPUT_IMAGE_PATH = "stego.png"
SECRET_MESSAGE = "Hello from adaptive LSB steganography!" * 50
PASSWORD = "secure_password_123"


def main():
    print("[+] Loading image...")
    rgb_img: np.ndarray = load_img(INPUT_IMAGE_PATH)

    height, width, _ = rgb_img.shape
    print(f"[+] Image size: {height}x{width}")

    print("[+] Converting to grayscale...")
    grayscale_img: np.ndarray = img_to_grayscale(rgb_img)

    print("[+] Computing LBP classification map...")
    classification_map: np.ndarray = compute_lbp_classification(
        grayscale_img
    )

    print("[+] Generating password-based pixel order...")
    seed: int = password_to_seed(PASSWORD)
    pixel_coords = generate_pixel_coordinates(height, width, seed)

    print("[+] Embedding message...")
    stego_img: np.ndarray = embed_message(
        rgb_img=rgb_img,
        secret_message=SECRET_MESSAGE,
        password=PASSWORD,
        classification_map=classification_map,
        pixel_coords=pixel_coords,
    )

    print("[+] Saving stego image...")
    success = cv2.imwrite(OUTPUT_IMAGE_PATH, stego_img)
    if not success:
        raise IOError("Failed to save output image.")

    print("[✓] Message successfully embedded!")
    print(f"[✓] Output saved to: {OUTPUT_IMAGE_PATH}")

    # EXTRACTION PIPELINE
    print("[+] Extracting message from stego image...")
    extracted_message: str = extract_message(
        stego_image=stego_img,
        password=PASSWORD,
        classification_map=classification_map,
        pixel_coords=pixel_coords,
    )

    print("[✓] Extracted message:")
    print(extracted_message)


if __name__ == "__main__":
    main()
