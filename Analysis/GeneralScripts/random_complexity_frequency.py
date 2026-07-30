import argparse
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt

from Core.ComplexityMeasures.complexity import ZlibComplexity, RLEComplexity, PatchComplexity2D


def generate_random_images(num_samples, shape, rng):
    H, W = shape
    return rng.integers(0, 2, size=(num_samples, H, W), dtype=np.uint8)


def analyze_distribution(images, metric):
    # Count identical images by their bytes
    counter = Counter()
    sample_map = {}
    for img in images:
        h = img.tobytes()
        counter[h] += 1
        # keep one example image for each hash
        if h not in sample_map:
            sample_map[h] = img

    unique_hashes = list(counter.keys())
    freqs = np.array([counter[h] for h in unique_hashes], dtype=np.int64)

    # Compute complexity for each unique image
    complexities = []
    for h in unique_hashes:
        img = sample_map[h]
        complexities.append(metric.calculate(img))
    complexities = np.array(complexities, dtype=float)

    return complexities, freqs


import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, pearsonr

def plot_complexity_vs_logfreq(complexities, freqs, metric_name, out_path=None):
    log_freq = np.log10(freqs)

    # Compute correlations
    pearson_corr, _ = pearsonr(complexities, log_freq)
    spearman_corr, _ = spearmanr(complexities, log_freq)

    # Create plot
    plt.figure(figsize=(8, 6))
    plt.scatter(
        complexities,
        log_freq,
        alpha=0.6,
        label=(
            f"Pearson: {pearson_corr:.3f}\n"
            f"Spearman: {spearman_corr:.3f}"
        )
    )

    plt.xlabel(f"Complexity ({metric_name})")
    plt.ylabel("log10(Frequency)")
    plt.title("Complexity vs log-frequency of randomly sampled binary images")

    plt.legend()
    plt.grid(alpha=0.3)

    if out_path:
        plt.savefig(out_path, bbox_inches='tight')

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze complexity vs log-frequency for random binary images")
    parser.add_argument("--samples", type=int, default=5000000)
    parser.add_argument("--height", type=int, default=4)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--metric", choices=["zlib", "rle", "patch"], default="zlib")
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out", type=str, default=None, help="Path to save plot (optional)")

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    images = generate_random_images(args.samples, (args.height, args.width), rng)

    if args.metric == "zlib":
        metric = ZlibComplexity()
    elif args.metric == "rle":
        metric = RLEComplexity()
    else:
        metric = PatchComplexity2D(patch_size=args.patch_size)

    complexities, freqs = analyze_distribution(images, metric)

    # Sort for nicer plotting
    order = np.argsort(complexities)
    complexities = complexities[order]
    freqs = freqs[order]

    plot_complexity_vs_logfreq(complexities, freqs, metric.name(), out_path=args.out)


if __name__ == "__main__":
    main()
