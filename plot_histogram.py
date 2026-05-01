from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

COVER_DIR = Path("data/png")
STEGO_BASE_DIR = Path("data/stego/png")
OUTPUT_DIR = Path("histograms/png")
BPP_VALUES = [0.1, 0.2, 0.3, 0.4]


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def plot_histogram(cover: np.ndarray, stego: np.ndarray, title: str, output_path: Path) -> None:
    colors = ["red", "green", "blue"]
    stego_color = "orange"
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)

    for i, ax in enumerate(axes):
        ax.hist(
            cover[:, :, i].ravel(),
            bins=256,
            range=(0, 255),
            color=colors[i],
            alpha=0.45,
            label="Cover",
        )
        ax.hist(
            stego[:, :, i].ravel(),
            bins=256,
            range=(0, 255),
            color=stego_color,
            alpha=0.45,
            label="Stego",
        )
        ax.set_title(f"{colors[i].title()} channel")
        ax.set_xlabel("Pixel intensity")
        ax.set_ylabel("Count")
        ax.legend()

    fig.suptitle(title, fontsize=12)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cover_paths = sorted(COVER_DIR.glob("*.png"))
    if not cover_paths:
        raise FileNotFoundError(f"No PNG files found in {COVER_DIR}")

    for cover_path in cover_paths:
        cover = load_rgb(cover_path)
        image_name = cover_path.stem

        for bpp in BPP_VALUES:
            bpp_label = f"{bpp:.1f}"
            stego_dir = STEGO_BASE_DIR / f"png_stego_bpp_{bpp_label}"
            stego_path = stego_dir / cover_path.name

            if not stego_path.exists():
                print(f"Skipping missing stego: {stego_path}")
                continue

            stego = load_rgb(stego_path)
            output_path = OUTPUT_DIR / f"histogram_{image_name}_bpp_{bpp_label}.png"
            title = f"Histogram Comparison: {cover_path.name} at {bpp_label} BPP"
            plot_histogram(cover, stego, title, output_path)
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
