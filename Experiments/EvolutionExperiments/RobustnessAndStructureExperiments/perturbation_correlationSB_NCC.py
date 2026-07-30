import os
import zlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from tqdm import tqdm


def get_lambda(rule_int):
    """Calculate Langton's lambda parameter for an ECA rule."""
    binary = [int(x) for x in bin(rule_int)[2:].zfill(8)]
    return sum(binary) / 8.0


def run_eca(rule_int, seed, steps):
    """Run Elementary Cellular Automata for T steps."""
    L = len(seed)
    grid = np.zeros((steps, L), dtype=np.int8)
    grid[0] = seed

    # Binary lookup array for rule transitions
    rule_bin = np.array([(rule_int >> i) & 1 for i in range(8)], dtype=np.int8)

    for t in range(1, steps):
        left = np.roll(grid[t - 1], 1)
        center = grid[t - 1]
        right = np.roll(grid[t - 1], -1)
        neighborhood = (left << 2) | (center << 1) | right
        grid[t] = rule_bin[neighborhood]

    return grid


def get_zlib_complexity(grid):
    """Compute base phenotype complexity K(x) using Zlib compression."""
    compressed = zlib.compress(grid.tobytes())
    return len(compressed)


def compute_hamming_distance(img1, img2):
    """Normalized Hamming distance between two 2D space-time patterns."""
    return np.mean(img1 != img2)


def compute_shift_invariant_ncc_distance(img1, img2, max_shift=4):
    """
    Computes 1 - Max Normalized Cross-Correlation over a small (dx, dt) search window.
    Distance = 0 implies identical structure up to spatial/temporal translation.
    """
    # Convert binary to zero-mean float [-1, 1]
    f1 = img1.astype(float) * 2 - 1
    f2 = img2.astype(float) * 2 - 1

    norm_factor = np.sqrt(np.sum(f1**2) * np.sum(f2**2))
    if norm_factor == 0:
        return 0.0

    max_corr = -1.0
    # Search over small spatial and temporal shift lags
    for dt in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            f2_shifted = np.roll(f2, shift=(dt, dx), axis=(0, 1))
            corr = np.sum(f1 * f2_shifted) / norm_factor
            if corr > max_corr:
                max_corr = corr

    # Return distance metric (0 = perfectly matchable via shift, 1 = uncorrelatable)
    return max(0.0, 1.0 - max_corr)


def run_experiment(
    L=32,
    T=32,
    num_seeds=20,
    max_flip_bits=6,
    max_shift=4,
    random_seed=42,
):
    rng = np.random.default_rng(random_seed)
    rules = range(256)
    results = []

    print(f"Running metric evaluation across 256 ECA rules (L={L}, T={T})...")

    for rule in tqdm(rules, desc="Evaluating Rules"):
        rule_lambda = get_lambda(rule)
        lambda_dist = abs(rule_lambda - 0.5)

        rule_hamming_list = []
        rule_sincc_list = []
        rule_complexity_list = []

        for _ in range(num_seeds):
            seed = rng.integers(0, 2, size=L, dtype=np.int8)
            base_img = run_eca(rule, seed, T)
            base_complexity = get_zlib_complexity(base_img[1:])

            # Perturb seed by flipping k bits
            for k in range(1, max_flip_bits + 1):
                mutated_seed = seed.copy()
                flip_indices = rng.choice(L, size=k, replace=False)
                mutated_seed[flip_indices] = 1 - mutated_seed[flip_indices]

                mutant_img = run_eca(rule, mutated_seed, T)

                # Skip initial row (t=0)
                p1 = base_img[1:]
                p2 = mutant_img[1:]

                h_dist = compute_hamming_distance(p1, p2)
                s_dist = compute_shift_invariant_ncc_distance(p1, p2, max_shift=max_shift)

                rule_hamming_list.append(h_dist)
                rule_sincc_list.append(s_dist)
                rule_complexity_list.append(base_complexity)

        results.append(
            {
                "rule": rule,
                "lambda": rule_lambda,
                "lambda_distance": lambda_dist,
                "base_complexity": np.mean(rule_complexity_list),
                "mean_hamming_dist": np.mean(rule_hamming_list),
                "mean_sincc_dist": np.mean(rule_sincc_list),
            }
        )

    return pd.DataFrame(results)


def plot_metric_comparison(df, save_path="hamming_vs_sincc_comparison.png"):
    """Plot Hamming vs SI-NCC Distance against Base Complexity K(x)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

    c = df["lambda_distance"]
    x = df["base_complexity"]

    # 1. Hamming Distance Plot
    sc1 = axes[0].scatter(
        x,
        df["mean_hamming_dist"],
        c=c,
        cmap="viridis_r",
        s=45,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.3,
    )
    axes[0].set_title("Hamming Distance vs Base Complexity K(x)")
    axes[0].set_xlabel("Base Phenotype Complexity K(x)")
    axes[0].set_ylabel("Mean Hamming Distance")
    axes[0].grid(True, alpha=0.25)

    # 2. Shift-Invariant NCC Distance Plot
    sc2 = axes[1].scatter(
        x,
        df["mean_sincc_dist"],
        c=c,
        cmap="viridis_r",
        s=45,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.3,
    )
    axes[1].set_title("Shift-Invariant NCC Distance vs Base Complexity K(x)")
    axes[1].set_xlabel("Base Phenotype Complexity K(x)")
    axes[1].set_ylabel("Mean SI-NCC Distance (1 - max_corr)")
    axes[1].grid(True, alpha=0.25)

    # Shared colorbar for |lambda - 0.5|
    cbar = fig.colorbar(sc2, ax=axes.ravel().tolist(), pad=0.02)
    cbar.set_label("|lambda - 0.5|", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    df = run_experiment(L=32, T=32, num_seeds=15, max_flip_bits=6, max_shift=4)
    plot_metric_comparison(df)