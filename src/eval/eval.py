"""
=============================================================================
  Steganography Evaluation using Standard Libraries
  Validates custom implementations against scikit-image metrics
  Metrics: MSE, PSNR, SSIM, BPP
=============================================================================

USAGE
-----
Single pair evaluation:
    python eval_library.py --cover cover.png --stego stego.png --bits 104832

Dependencies:
    pip install scikit-image numpy pillow
"""

import os
import csv
import argparse
import numpy as np
from PIL import Image

# ─── scikit-image metrics (industry standard) ────────────────────────────────
from skimage.metrics import (
    mean_squared_error as skimage_mse,
    peak_signal_noise_ratio as skimage_psnr,
    structural_similarity as skimage_ssim,
)


# =============================================================================
#  IMAGE I/O
# =============================================================================

def load_image_rgb(path: str) -> np.ndarray:
    """Load an image as a uint8 RGB numpy array (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


# =============================================================================
#  METRICS USING SCIKIT-IMAGE
# =============================================================================

def compute_mse(cover: np.ndarray, stego: np.ndarray) -> float:
    """
    MSE using scikit-image's mean_squared_error.
    Computed per channel then averaged for RGB.
    """
    if cover.ndim == 3:
        # Average MSE across RGB channels
        mse_vals = [skimage_mse(cover[:, :, ch], stego[:, :, ch]) for ch in range(3)]
        return float(np.mean(mse_vals))
    return float(skimage_mse(cover, stego))


def compute_psnr(cover: np.ndarray, stego: np.ndarray) -> float:
    """
    PSNR using scikit-image's peak_signal_noise_ratio.
    """
    return float(skimage_psnr(cover, stego, data_range=255))


def compute_ssim(cover: np.ndarray, stego: np.ndarray) -> float:
    """
    SSIM using scikit-image's structural_similarity.
    Uses channel_axis for RGB images.
    """
    if cover.ndim == 3:
        return float(skimage_ssim(cover, stego, channel_axis=2, data_range=255))
    return float(skimage_ssim(cover, stego, data_range=255))


def compute_bpp(total_bits: int, image_shape: tuple) -> float:
    """
    BPP = Total Embedded Bits / (M x N)
    """
    M, N = image_shape[0], image_shape[1]
    return total_bits / (M * N)


# =============================================================================
#  EVALUATION PIPELINE
# =============================================================================

def evaluate_pair(
        cover_path: str,
        stego_path: str,
        total_bits_embedded: int,
        label: str = None
) -> dict:
    """
    Run all metrics on a cover/stego pair using scikit-image.
    """
    print(f"\n{'═'*60}")
    print(f"  Evaluating (Library): {label or os.path.basename(stego_path)}")
    print(f"{'═'*60}")
    print(f"  Using: scikit-image metrics (industry standard)")

    cover = load_image_rgb(cover_path)
    stego = load_image_rgb(stego_path)

    if cover.shape != stego.shape:
        raise ValueError(
            f"Shape mismatch: cover {cover.shape} vs stego {stego.shape}."
        )

    H, W = cover.shape[:2]
    print(f"  Image size   : {W}x{H} pixels ({W*H:,} total pixels)")
    print(f"  Bits embedded: {total_bits_embedded:,}")

    # ── MSE (scikit-image) ──────────────────────────────────────────────────
    mse = compute_mse(cover, stego)
    print(f"\n  [MSE]  {mse:.6f}  (scikit-image)")
    print(f"         (0 = identical; lower is better)")

    # ── PSNR (scikit-image) ─────────────────────────────────────────────────
    psnr = compute_psnr(cover, stego)
    quality_label = (
        "★ Very good (>40 dB)" if psnr > 40 else
        "✓ Acceptable (30-40 dB)" if psnr >= 30 else
        "✗ Not acceptable (<30 dB)"
    )
    print(f"\n  [PSNR] {psnr:.4f} dB  →  {quality_label}  (scikit-image)")

    # ── SSIM (scikit-image) ─────────────────────────────────────────────────
    ssim = compute_ssim(cover, stego)
    ssim_label = (
        "★ Nearly imperceptible (>0.98)" if ssim > 0.98 else
        "✓ Acceptable (0.90-0.98)" if ssim >= 0.90 else
        "✗ Noticeable distortion (<0.90)"
    )
    print(f"\n  [SSIM] {ssim:.6f}  →  {ssim_label}  (scikit-image)")

    # ── BPP ─────────────────────────────────────────────────────────────────
    bpp = compute_bpp(total_bits_embedded, cover.shape)
    print(f"\n  [BPP]  {bpp:.6f} bits per pixel")
    print(f"         ({total_bits_embedded:,} bits / {W*H:,} pixels)")

    print(f"{'─'*60}")

    return {
        "label": label or os.path.basename(stego_path),
        "cover_path": cover_path,
        "stego_path": stego_path,
        "image_size": f"{W}x{H}",
        "total_pixels": W * H,
        "bits_embedded": total_bits_embedded,
        "mse": round(mse, 6),
        "psnr_db": round(psnr, 4),
        "ssim": round(ssim, 6),
        "bpp": round(bpp, 6),
        "library": "scikit-image",
    }


# =============================================================================
#  CSV EXPORT
# =============================================================================

def save_csv(results: list, output_path: str = "results_library.csv") -> None:
    """Save evaluation results to CSV."""
    if not results:
        print("No results to save.")
        return

    fieldnames = list(results[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  ✔  Results saved to: {output_path}")


# =============================================================================
#  CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Steganography Evaluation using scikit-image (validation)"
    )
    parser.add_argument("--cover", type=str, required=True,
                        help="Path to cover image")
    parser.add_argument("--stego", type=str, required=True,
                        help="Path to stego image")
    parser.add_argument("--bits", type=int, required=True,
                        help="Number of bits embedded")
    parser.add_argument("--label", type=str, default=None,
                        help="Optional label")
    parser.add_argument("--output", type=str, default="results_library.csv",
                        help="Output CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    result = evaluate_pair(args.cover, args.stego, args.bits, args.label)
    save_csv([result], args.output)


if __name__ == "__main__":
    main()
