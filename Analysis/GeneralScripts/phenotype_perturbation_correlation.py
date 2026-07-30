import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Core.ComplexityMeasures.complexity import ZlibComplexity
from Core.ComplexityMeasures.conditional_complexity import ZlibConditionalComplexity
from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA


def get_lambda(rule_int):
    binary = [int(x) for x in bin(rule_int)[2:].zfill(8)]
    return sum(binary) / 8.0


def flip_seed_bits(seed, bit_count, rng):
    """Return a copy of seed with exactly bit_count positions flipped."""
    if bit_count < 1:
        raise ValueError("bit_count must be at least 1")

    if bit_count > len(seed):
        raise ValueError(f"bit_count={bit_count} exceeds seed length {len(seed)}")

    mutated = seed.copy()
    indices = rng.choice(len(seed), size=bit_count, replace=False)
    mutated[indices] = 1 - mutated[indices]
    return mutated


def compute_hamming_distance(x, y, normalize=True):
    """Compute (normalized) Hamming distance between two binary grids."""
    if normalize:
        return np.mean(x != y)
    return np.sum(x != y)


def compute_shift_invariant_ncc(x, y):
    """
    Compute Shift-Invariant Normalized Cross-Correlation (NCC) between two 2D CA grids.
    Performs circular spatial shifts along the grid width (axis 1) to find the peak correlation.
    """
    x_float = x.astype(float)
    y_float = y.astype(float)

    x_mean = np.mean(x_float)
    y_mean = np.mean(y_float)
    x_std = np.std(x_float)
    y_std = np.std(y_float)

    # Handle zero-variance edge cases (e.g., constant fields)
    if x_std == 0 or y_std == 0:
        return 1.0 if np.array_equal(x, y) else 0.0

    x_norm = (x_float - x_mean) / (x_std * np.sqrt(x_float.size))
    y_norm = (y_float - y_mean) / (y_std * np.sqrt(y_float.size))

    max_ncc = -1.0
    L = x.shape[1] if x.ndim > 1 else x.shape[0]
    spatial_axis = 1 if x.ndim > 1 else 0

    # Maximize NCC over periodic spatial shifts
    for shift in range(L):
        y_shifted = np.roll(y_norm, shift, axis=spatial_axis)
        ncc = np.sum(x_norm * y_shifted)
        if ncc > max_ncc:
            max_ncc = ncc

    return float(np.clip(max_ncc, -1.0, 1.0))


def collect_perturbation_data(
    L=32,
    T=32,
    rules=range(256),
    num_seeds_per_rule=40,
    max_flip_bits=8,
    samples_per_size=12,
    random_seed=1234,
):
    """
    For each rule and seed, measure how conditional complexity, Hamming distance,
    and shift-invariant NCC change as the number of flipped seed bits increases.
    """
    engine = ElementaryCA(L=L, T=T)
    complexity_metric = ZlibComplexity()
    conditional_metric = ZlibConditionalComplexity()
    rng = np.random.default_rng(random_seed)

    rows = []

    print(
        f"Running phenotype perturbation correlation experiment: L={L}, T={T}, "
        f"seeds/rule={num_seeds_per_rule}, max_flip_bits={max_flip_bits}, samples/size={samples_per_size}"
    )

    for rule in tqdm(list(rules), desc="Rules"):
        rule_lambda = get_lambda(rule)
        lambda_distance = abs(rule_lambda - 0.5)

        for _ in range(num_seeds_per_rule):
            seed = engine.generate_seed(seed_type="random")
            base_img = engine.run(rule, seed)
            base_phenotype = base_img[1:]
            base_complexity = complexity_metric.calculate(base_phenotype)

            for flip_bits in range(1, min(max_flip_bits, len(seed)) + 1):
                for _ in range(samples_per_size):
                    mutated_seed = flip_seed_bits(seed, flip_bits, rng)
                    mutant_img = engine.run(rule, mutated_seed)
                    mutant_phenotype = mutant_img[1:]

                    cond_complexity = conditional_metric.calculate(base_phenotype, mutant_phenotype)
                    hamming_dist = compute_hamming_distance(base_phenotype, mutant_phenotype)
                    shift_ncc = compute_shift_invariant_ncc(base_phenotype, mutant_phenotype)
                    phenotype_delta = complexity_metric.calculate(mutant_phenotype) - base_complexity

                    rows.append(
                        {
                            "rule": rule,
                            "lambda": rule_lambda,
                            "lambda_distance": lambda_distance,
                            "base_complexity": base_complexity,
                            "flip_bits": flip_bits,
                            "conditional_complexity": cond_complexity,
                            "hamming_distance": hamming_dist,
                            "shift_invariant_ncc": shift_ncc,
                            "phenotype_delta_complexity": phenotype_delta,
                        }
                    )

    return pd.DataFrame(rows)


def summarize_by_rule(df):
    """Aggregate sample-level data into summary statistics and correlations per rule."""
    summary_rows = []

    for rule, group in df.groupby("rule"):
        if group["flip_bits"].nunique() < 2:
            continue

        # Correlations vs flip_bits for each metric
        rho_cond, p_cond_spearman = spearmanr(group["flip_bits"], group["conditional_complexity"])
        r_cond, p_cond_pearson = pearsonr(group["flip_bits"], group["conditional_complexity"])

        rho_hamm, p_hamm_spearman = spearmanr(group["flip_bits"], group["hamming_distance"])
        r_hamm, p_hamm_pearson = pearsonr(group["flip_bits"], group["hamming_distance"])

        rho_ncc, p_ncc_spearman = spearmanr(group["flip_bits"], group["shift_invariant_ncc"])
        r_ncc, p_ncc_pearson = pearsonr(group["flip_bits"], group["shift_invariant_ncc"])

        summary_rows.append(
            {
                "rule": rule,
                "lambda": group["lambda"].iloc[0],
                "lambda_distance": group["lambda_distance"].iloc[0],
                "base_complexity": group["base_complexity"].mean(),
                # Conditional Complexity Stats
                "spearman_rho_cond": rho_cond,
                "pearson_r_cond": r_cond,
                "mean_conditional_complexity": group["conditional_complexity"].mean(),
                # Hamming Distance Stats
                "spearman_rho_hamming": rho_hamm,
                "pearson_r_hamming": r_hamm,
                "mean_hamming_distance": group["hamming_distance"].mean(),
                # Shift-Invariant NCC Stats
                "spearman_rho_ncc": rho_ncc,
                "pearson_r_ncc": r_ncc,
                "mean_shift_invariant_ncc": group["shift_invariant_ncc"].mean(),
                "mean_phenotype_delta_complexity": group["phenotype_delta_complexity"].mean(),
            }
        )

    return pd.DataFrame(summary_rows)


def plot_metrics_vs_correlation(summary_df, corr_type="spearman", save_path=None):
    """Plot phenotype complexity vs perturbation correlation side-by-side for all 3 metrics."""
    if summary_df.empty:
        raise ValueError("No data available to plot.")

    if save_path is None:
        save_path = os.path.join(
            "Analysis",
            "GeneralScripts",
            "Saved Figures",
            f"phenotype_complexity_vs_{corr_type}_correlations.png",
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    corr_suffix = "rho" if corr_type == "spearman" else "r"
    metrics_info = [
        ("spearman_rho_cond" if corr_type == "spearman" else "pearson_r_cond", "K(y|x) (Conditional Complexity)"),
        ("spearman_rho_hamming" if corr_type == "spearman" else "pearson_r_hamming", "Hamming Distance"),
        ("spearman_rho_ncc" if corr_type == "spearman" else "pearson_r_ncc", "Shift-Invariant NCC"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True)
    x = summary_df["base_complexity"]
    c = summary_df["lambda_distance"]

    for ax, (col, name) in zip(axes, metrics_info):
        y = summary_df[col]
        scatter = ax.scatter(
            x,
            y,
            c=c,
            cmap="viridis_r",
            s=35,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.3,
        )

        if len(summary_df) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 200)
            ax.plot(xs, slope * xs + intercept, color="crimson", linewidth=2, label=f"Trend (slope={slope:.3g})")
            ax.legend(loc="best")

        ax.set_title(f"{name}\nvs Base Complexity K(x)")
        ax.set_xlabel("Base phenotype complexity K(x)")
        ax.set_ylabel(f"Correlation ({corr_type.capitalize()} {corr_suffix})")
        ax.grid(True, alpha=0.25)

    fig.colorbar(scatter, ax=axes.ravel().tolist(), label="|lambda - 0.5|", pad=0.02)
    plt.suptitle(f"Perturbation-Size Correlation Comparison ({corr_type.capitalize()})", fontsize=14, y=1.02)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.show()


def plot_average_response_by_flip_size(df, save_path=None):
    """Plot how Conditional Complexity, Hamming Distance, and Shift-Invariant NCC scale with perturbation size."""
    if save_path is None:
        save_path = os.path.join(
            "Analysis",
            "GeneralScripts",
            "Saved Figures",
            "phenotype_response_vs_flip_size.png",
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    metrics = [
        ("conditional_complexity", "Mean K(y|x) (Conditional Complexity)", "teal"),
        ("hamming_distance", "Mean Hamming Distance", "chocolate"),
        ("shift_invariant_ncc", "Mean Shift-Invariant NCC", "darkslateblue"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    for ax, (metric_col, ylabel, color) in zip(axes, metrics):
        grouped = df.groupby("flip_bits")[metric_col].agg(["mean", "std"]).reset_index()

        ax.errorbar(
            grouped["flip_bits"],
            grouped["mean"],
            yerr=grouped["std"],
            fmt="o-",
            capsize=4,
            color=color,
            ecolor="gray",
            linewidth=2,
            markersize=6,
        )
        ax.set_title(f"{ylabel} vs Perturbation Size")
        ax.set_xlabel("Number of flipped seed bits")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    plt.suptitle("Average Response Across All Rules by Seed Flip Size", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.show()


def main():
    L = 32
    T = 32
    NUM_SEEDS_PER_RULE = 20
    MAX_FLIP_BITS = 8
    SAMPLES_PER_SIZE = 10
    RULES = [30]

    output_dir = os.path.join("Analysis", "GeneralScripts", "Saved Figures")
    os.makedirs(output_dir, exist_ok=True)

    raw_df = collect_perturbation_data(
        L=L,
        T=T,
        rules=RULES,
        num_seeds_per_rule=NUM_SEEDS_PER_RULE,
        max_flip_bits=MAX_FLIP_BITS,
        samples_per_size=SAMPLES_PER_SIZE,
    )

    csv_path = os.path.join(output_dir, "phenotype_perturbation_correlation_raw.csv")
    raw_df.to_csv(csv_path, index=False)
    print(f"Saved raw data to {csv_path}")

    summary_df = summarize_by_rule(raw_df)
    summary_csv_path = os.path.join(output_dir, "phenotype_perturbation_correlation_by_rule.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved rule summary to {summary_csv_path}")

    # Plot comparative correlations for Spearman and Pearson
    plot_metrics_vs_correlation(
        summary_df,
        corr_type="spearman",
        save_path=os.path.join(output_dir, "phenotype_complexity_vs_spearman_rho.png"),
    )
    plot_metrics_vs_correlation(
        summary_df,
        corr_type="pearson",
        save_path=os.path.join(output_dir, "phenotype_complexity_vs_pearson_r.png"),
    )

    # Plot comparative responses over seed perturbation size
    plot_average_response_by_flip_size(
        raw_df,
        save_path=os.path.join(output_dir, "phenotype_response_vs_flip_size.png"),
    )

    if not summary_df.empty:
        print("\n--- Summary Statistics (Spearman Rho across rules) ---")
        print(f"Conditional Complexity mean: {summary_df['spearman_rho_cond'].mean():.4f}")
        print(f"Hamming Distance mean:       {summary_df['spearman_rho_hamming'].mean():.4f}")
        print(f"Shift-Invariant NCC mean:    {summary_df['spearman_rho_ncc'].mean():.4f}")


if __name__ == "__main__":
    main()