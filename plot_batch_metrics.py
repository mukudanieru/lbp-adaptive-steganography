import csv
import math
from collections import defaultdict
import matplotlib.pyplot as plt

# Add all CSVs you want to average together (png/bmp/tiff steganalysis outputs).
INPUT_CSVS = [
    "batch_eval_all12.csv"
    # "png_batch_eval_all.csv",
    # "bmp_batch_eval_all.csv",
    # "tiff_batch_eval_all.csv",
]

OUTPUT_PNG = "batch_metrics_line_graph.png"

METRICS = {
    "mse": "MSE",
    "psnr_db": "PSNR (dB)",
    "ssim": "SSIM",
    "RS analysis": "RS analysis",
}


def parse_float(value: str):
    try:
        val = float(value)
        if math.isnan(val):
            return None
        return val
    except (TypeError, ValueError):
        return None


def main():
    # Accumulate values by target_bpp
    grouped = {metric: defaultdict(list) for metric in METRICS}

    for path in INPUT_CSVS:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                bpp = parse_float(row.get("target_bpp", ""))
                if bpp is None:
                    continue

                for metric in METRICS:
                    val = parse_float(row.get(metric, ""))
                    if val is not None:
                        grouped[metric][bpp].append(val)

    # Compute averages by BPP
    bpps = sorted({bpp for metric in grouped for bpp in grouped[metric]})
    if not bpps:
        raise ValueError("No valid data found. Check INPUT_CSVS and column names.")

    averages = {metric: [] for metric in METRICS}
    for bpp in bpps:
        for metric in METRICS:
            values = grouped[metric].get(bpp, [])
            if not values:
                averages[metric].append(None)
            else:
                averages[metric].append(sum(values) / len(values))

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes = axes.flatten()

    for idx, (metric, label) in enumerate(METRICS.items()):
        ax = axes[idx]
        y = averages[metric]
        ax.plot(bpps, y, marker="o")
        ax.set_title(label)
        ax.set_xlabel("Target BPP")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Average Stego Quality vs Payload", fontsize=12)
    fig.savefig(OUTPUT_PNG, dpi=300)
    print(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
