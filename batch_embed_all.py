"""
Batch embed all PNG images at multiple BPP values and evaluate.
Creates separate folders per BPP and compiles all results into a single CSV.
"""
import os
import csv
import numpy as np
from PIL import Image
from src.core.preprocessing import img_to_grayscale
from src.core.lbp import compute_lbp_classification
from src.core.embedding import embed_message, calculate_capacity
from src.core.pseudorandom import password_to_seed, generate_pixel_coordinates
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

# Config
INPUT_DIR = 'data'
BPP_VALUES = [0.1, 0.2, 0.3, 0.4]
PASSWORD = 'mypass'
OUTPUT_CSV = 'batch_eval_all.csv'

LOREM = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. ' * 200


def embed_and_eval(cover_path, output_path, bpp, password):
    """Embed at target BPP and return evaluation metrics."""
    # Load image
    img = Image.open(cover_path).convert('RGB')
    rgb_img = np.array(img)
    height, width = rgb_img.shape[:2]
    total_pixels = height * width
    
    # Compute LBP classification
    gray = img_to_grayscale(rgb_img)
    classification_map = compute_lbp_classification(gray)
    
    # Calculate max capacity
    max_capacity = calculate_capacity(classification_map, 3)
    
    # Generate message for target BPP
    target_bits = int(bpp * total_pixels)
    target_chars = target_bits // 8
    message = (LOREM * ((target_chars // len(LOREM)) + 1))[:target_chars]
    actual_bits = len(message) * 8
    
    # Embed
    seed = password_to_seed(password)
    pixel_coords = generate_pixel_coordinates(height, width, seed)
    stego_img = embed_message(rgb_img, message, password, classification_map, pixel_coords)
    
    # Save
    Image.fromarray(stego_img).save(output_path)
    
    # Evaluate
    actual_bpp = actual_bits / total_pixels
    mse = float(np.mean([mean_squared_error(rgb_img[:,:,c], stego_img[:,:,c]) for c in range(3)]))
    psnr = float(peak_signal_noise_ratio(rgb_img, stego_img, data_range=255))
    ssim = float(structural_similarity(rgb_img, stego_img, channel_axis=2, data_range=255))
    
    return {
        'image': os.path.basename(cover_path),
        'size': f'{width}x{height}',
        'total_pixels': total_pixels,
        'max_capacity_bits': max_capacity,
        'target_bpp': bpp,
        'actual_bpp': round(actual_bpp, 4),
        'bits_embedded': actual_bits,
        'chars_embedded': len(message),
        'mse': round(mse, 6),
        'psnr_db': round(psnr, 4),
        'ssim': round(ssim, 6),
        'stego_path': output_path
    }


def main():
    # Get all PNG files
    png_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.png')]
    png_files.sort()
    
    print(f'Found {len(png_files)} PNG files')
    print(f'BPP values: {BPP_VALUES}')
    print(f'Total embeddings: {len(png_files) * len(BPP_VALUES)}')
    print('=' * 60)
    
    results = []
    
    for bpp in BPP_VALUES:
        # Create output folder
        out_dir = f'stego_bpp_{bpp}'
        os.makedirs(out_dir, exist_ok=True)
        print(f'\n[BPP {bpp}] Output: {out_dir}/')
        
        for img_file in png_files:
            cover_path = os.path.join(INPUT_DIR, img_file)
            output_path = os.path.join(out_dir, img_file)
            
            print(f'  {img_file}...', end=' ')
            result = embed_and_eval(cover_path, output_path, bpp, PASSWORD)
            results.append(result)
            print(f'PSNR: {result["psnr_db"]:.2f} dB, SSIM: {result["ssim"]:.4f}')
    
    # Save all results to CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print('\n' + '=' * 60)
    print(f'Results saved to: {OUTPUT_CSV}')
    print(f'Total rows: {len(results)}')
    
    # Summary table
    print('\n| Image | BPP | PSNR (dB) | SSIM |')
    print('|-------|-----|-----------|------|')
    for r in results:
        print(f'| {r["image"]} | {r["target_bpp"]} | {r["psnr_db"]:.2f} | {r["ssim"]:.4f} |')


if __name__ == '__main__':
    main()
