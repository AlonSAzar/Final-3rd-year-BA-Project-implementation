import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from Core.ComplexityMeasures.complexity import ZlibComplexity, RLEComplexity, PatchComplexity2D

def generate_random_images(num_samples, shape, rng):
    """Generates random binary images (0 or 1)."""
    H, W = shape
    return rng.integers(0, 2, size=(num_samples, H, W), dtype=np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Generate complexity frequency histogram for random binary images (L x T).")
    parser.add_argument("--L", type=int, default=64, help="Width of the image")
    parser.add_argument("--T", type=int, default=64, help="Height of the image (Time steps in CA context)")
    parser.add_argument("--samples", type=int, default=1000, help="Number of random images to sample")
    parser.add_argument("--metric", choices=["zlib", "rle", "patch"], default="zlib", help="Complexity metric to use")
    parser.add_argument("--patch-size", type=int, default=4, help="Patch size for PatchComplexity2D")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save", type=str, default="Analysis/Saved Figures/random_complexity_histogram.png", help="Path to save the histogram")

    args = parser.parse_args()

    # Initialize RNG
    rng = np.random.default_rng(args.seed)

    # Initialize Metric
    if args.metric == "zlib":
        metric = ZlibComplexity()
    elif args.metric == "rle":
        metric = RLEComplexity()
    elif args.metric == "patch":
        metric = PatchComplexity2D(patch_size=args.patch_size)
    else:
        raise ValueError(f"Unknown metric: {args.metric}")

    print(f"Generating {args.samples} random images of size {args.L}x{args.T}...")
    images = generate_random_images(args.samples, (args.T, args.L), rng)

    print(f"Calculating complexity using {metric.name()}...")
    complexities = []
    for i in range(args.samples):
        c = metric.calculate(images[i])
        complexities.append(c)
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{args.samples}...")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(complexities, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f"Complexity Frequency Histogram (Random Images {args.L}x{args.T})\nMetric: {metric.name()}")
    plt.xlabel("Complexity Score")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.3)

    # Ensure save directory exists
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    plt.savefig(args.save)
    print(f"Histogram saved to {args.save}")
    plt.show()

if __name__ == "__main__":
    main()
