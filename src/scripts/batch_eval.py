"""
batch_eval.py

Evaluate all stego images against their cover images.

Usage:
    python src/scripts/batch_eval.py --cover ./data --stego ./stego --bits 26208
    python src/scripts/batch_eval.py --cover ./data --stego ./stego --bpp 0.1
    python src/scripts/batch_eval.py --cover ./data --stego ./stego --bpp 0.1 --output results.csv
"""

import os
import sys
import csv
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.eval.eval import (
    load_image_rgb,
    compute_mse,
    compute_psnr,
    compute_ssim,
    compute_bpp,
)

SUPPORTED_FORMATS = (".png", ".bmp", ".tif", ".tiff")


def find_cover_image(stego_filename: str, cover_dir: str) -> str:
    """Find the matching cover image for a stego image."""
    # Remove common stego suffixes
    base_name = stego_filename.replace("_stego", "").replace("_lsb", "")
    
    for ext in SUPPORTED_FORMATS:
        # Try exact match
        cover_path = os.path.join(cover_dir, base_name)
        if os.path.exists(cover_path):
            return cover_path
        
        # Try with different extensions
        name_no_ext = os.path.splitext(base_name)[0]
        cover_path = os.path.join(cover_dir, name_no_ext + ext)
        if os.path.exists(cover_path):
            return cover_path
    
    return None


def evaluate_single(cover_path: str, stego_path: str, bits_embedded: int) -> dict:
    """Evaluate a single cover/stego pair."""
    cover = load_image_rgb(cover_path)
    stego = load_image_rgb(stego_path)
    
    if cover.shape != stego.shape:
        raise ValueError(f"Shape mismatch: {cover.shape} vs {stego.shape}")
    
    H, W = cover.shape[:2]
    
    mse = compute_mse(cover, stego)
    psnr = compute_psnr(cover, stego)
    ssim = compute_ssim(cover, stego)
    bpp = compute_bpp(bits_embedded, cover.shape)
    
    return {
        "cover": os.path.basename(cover_path),
        "stego": os.path.basename(stego_path),
        "size": f"{W}x{H}",
        "bits": bits_embedded,
        "bpp": round(bpp, 4),
        "mse": round(mse, 6),
        "psnr": round(psnr, 4),
        "ssim": round(ssim, 6),
    }


def batch_evaluate(cover_dir: str, stego_dir: str, 
                   bits: int = None, bpp: float = None,
                   output_csv: str = None) -> list:
    """Evaluate all stego images in a directory."""
    
    # Get all stego images
    stego_files = [f for f in os.listdir(stego_dir) 
                   if f.lower().endswith(SUPPORTED_FORMATS)]
    
    if not stego_files:
        print(f"No stego images found in {stego_dir}")
        return []
    
    print(f"\n{'═'*70}")
    print(f"  Batch Evaluation: {len(stego_files)} images")
    print(f"  Cover directory: {cover_dir}")
    print(f"  Stego directory: {stego_dir}")
    print(f"{'═'*70}\n")
    
    results = []
    
    # Print header
    print(f"{'Image':<20} {'PSNR (dB)':<12} {'SSIM':<10} {'MSE':<12} {'BPP':<8}")
    print(f"{'-'*20} {'-'*12} {'-'*10} {'-'*12} {'-'*8}")
    
    for stego_filename in sorted(stego_files):
        stego_path = os.path.join(stego_dir, stego_filename)
        cover_path = find_cover_image(stego_filename, cover_dir)
        
        if cover_path is None:
            print(f"{stego_filename:<20} ✗ No matching cover found")
            continue
        
        # Calculate bits from BPP if needed
        if bpp is not None:
            cover = load_image_rgb(cover_path)
            H, W = cover.shape[:2]
            bits_embedded = int(bpp * H * W)
        else:
            bits_embedded = bits
        
        try:
            stats = evaluate_single(cover_path, stego_path, bits_embedded)
            results.append(stats)
            
            # Determine quality indicators
            psnr_icon = "★" if stats["psnr"] > 40 else "✓" if stats["psnr"] >= 30 else "✗"
            ssim_icon = "★" if stats["ssim"] > 0.98 else "✓" if stats["ssim"] >= 0.90 else "✗"
            
            print(f"{stats['cover']:<20} {stats['psnr']:>8.2f} {psnr_icon}   "
                  f"{stats['ssim']:.6f} {ssim_icon} {stats['mse']:<12.6f} {stats['bpp']:<8.4f}")
            
        except Exception as e:
            print(f"{stego_filename:<20} ✗ Error: {e}")
    
    # Summary statistics
    if results:
        avg_psnr = sum(r["psnr"] for r in results) / len(results)
        avg_ssim = sum(r["ssim"] for r in results) / len(results)
        avg_mse = sum(r["mse"] for r in results) / len(results)
        
        print(f"\n{'-'*70}")
        print(f"{'AVERAGE':<20} {avg_psnr:>8.2f}     {avg_ssim:.6f}   {avg_mse:<12.6f}")
        print(f"{'='*70}")
        
        # Quality summary
        print(f"\n  Summary:")
        print(f"    Images evaluated: {len(results)}")
        print(f"    Average PSNR: {avg_psnr:.2f} dB", end="")
        if avg_psnr > 40:
            print(" → ★ Very good (>40 dB)")
        elif avg_psnr >= 30:
            print(" → ✓ Acceptable (30-40 dB)")
        else:
            print(" → ✗ Not acceptable (<30 dB)")
        
        print(f"    Average SSIM: {avg_ssim:.6f}", end="")
        if avg_ssim > 0.98:
            print(" → ★ Nearly imperceptible (>0.98)")
        elif avg_ssim >= 0.90:
            print(" → ✓ Acceptable (0.90-0.98)")
        else:
            print(" → ✗ Noticeable distortion (<0.90)")
    
    # Save CSV if requested
    if output_csv and results:
        save_csv(results, output_csv)
    
    return results


def save_csv(results: list, output_path: str) -> None:
    """Save results to CSV."""
    fieldnames = list(results[0].keys())
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n  ✔ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluate stego images against cover images"
    )
    parser.add_argument("--cover", type=str, required=True,
                        help="Directory containing cover images")
    parser.add_argument("--stego", type=str, required=True,
                        help="Directory containing stego images")
    parser.add_argument("--bits", type=int, default=None,
                        help="Number of bits embedded per image")
    parser.add_argument("--bpp", type=float, default=None,
                        help="BPP used for embedding (calculates bits automatically)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path")
    args = parser.parse_args()
    
    if args.bits is None and args.bpp is None:
        parser.error("Either --bits or --bpp is required")
    
    batch_evaluate(args.cover, args.stego, args.bits, args.bpp, args.output)


if __name__ == "__main__":
    main()
