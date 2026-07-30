import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from scipy.stats import linregress, spearmanr
from collections import Counter

# Add the project root to sys.path to resolve imports properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.ComplexityMeasures.complexity import ZlibComplexity


def analyze_rule_simplicity_bias(rule, L=128, T=2048, piece_height=16):
    """
    Runs a rule from a random seed, divides the evolution into pieces,
    and computes the relationship between complexity and log-frequency.
    """
    engine = ElementaryCA(L=L, T=T)
    metric = ZlibComplexity()

    # Random seed
    seed = engine.generate_seed()
    full_history = engine.run(rule, seed)[1:]  # Ignore seed row

    # Divide into pieces
    num_pieces = T // piece_height
    pieces = [
        full_history[i * piece_height:(i + 1) * piece_height]
        for i in range(num_pieces)
    ]

    # Count frequencies
    piece_hashes = [p.tobytes() for p in pieces]
    counts = Counter(piece_hashes)

    unique_hashes = list(counts.keys())

    if len(unique_hashes) < 3:
        return None

    complexities = []
    log_frequencies = []

    for h in unique_hashes:
        idx = piece_hashes.index(h)
        complexities.append(metric.calculate(pieces[idx]))
        log_frequencies.append(np.log(counts[h]))

    if np.std(complexities) == 0:
        return None

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(
        complexities,
        log_frequencies
    )

    # Spearman correlation
    spearman_corr, spearman_p = spearmanr(
        complexities,
        log_frequencies
    )
    # Treat undefined Spearman correlations as zero
    if np.isnan(spearman_corr):
        spearman_corr = 0.0

    return {
        "slope": slope,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "spearman": spearman_corr,
        "spearman_p": spearman_p,
    }


def main():

    # ============================================================
    # PARAMETERS
    # ============================================================

    L = 16
    T = 2048
    piece_height = 16

    # <<< CHANGE THIS TO RUN MORE RANDOM SEEDS PER RULE >>>
    num_seed_iterations = 50

    # ============================================================

    print("Testing Simplicity Bias across all 256 rules...")
    print(f"L = {L}")
    print(f"T = {T}")
    print(f"Piece Height = {piece_height}")
    print(f"Random seeds per rule = {num_seed_iterations}")

    significant_negative_rules = 0
    total_valid_rules = 0

    all_slopes = []
    all_spearman = []

    for rule in tqdm(range(256), desc="Rules"):

        slopes = []
        spearmans = []
        p_values = []

        # Repeat using different random seeds
        for _ in range(num_seed_iterations):

            result = analyze_rule_simplicity_bias(
                rule,
                L=L,
                T=T,
                piece_height=piece_height,
            )

            if result is None:
                continue

            slopes.append(result["slope"])
            spearmans.append(result["spearman"])
            p_values.append(result["p_value"])

        # Skip rules where every run failed
        if len(slopes) == 0:
            continue

        total_valid_rules += 1

        avg_slope = np.mean(slopes)
        avg_spearman = np.mean(spearmans)
        avg_p = np.mean(p_values)

        all_slopes.append(avg_slope)
        all_spearman.append(avg_spearman)

        if avg_slope < 0 and avg_p < 0.05:
            significant_negative_rules += 1

    print("\n--- RESULTS ---")
    print("Total Rules Analyzed: 256")
    print(f"Rules with valid data: {total_valid_rules}")
    print(
        f"Rules with significant negative relationship: "
        f"{significant_negative_rules}"
    )

    if total_valid_rules > 0:
        print(
            f"Percentage: "
            f"{100 * significant_negative_rules / total_valid_rules:.2f}%"
        )
        print(f"Average slope: {np.mean(all_slopes):.4f}")
        print(
            f"Average Spearman: "
            f"{np.mean(all_spearman):.4f} ± {np.std(all_spearman):.4f}"
        )
    os.makedirs("Saved Figures", exist_ok=True)

    # ============================================================
    # Histogram of slopes
    # ============================================================

    plt.figure(figsize=(10, 6))

    plt.hist(
        all_slopes,
        bins=30,
        color="skyblue",
        edgecolor="black",
    )

    plt.axvline(
        0,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label="Slope = 0",
    )

    plt.title(
        f"Slope Distribution\n"
        f"L={L}, T={T}, Slice Height={piece_height}, "
        f"Seeds={num_seed_iterations}"
    )

    plt.xlabel("Slope")
    plt.ylabel("Number of Rules")
    plt.legend()

    slope_path = "Saved Figures/Simplicity_Bias_Slope_Distribution.png"
    plt.savefig(slope_path)

    # ============================================================
    # Histogram of Spearman correlations
    # ============================================================

    plt.figure(figsize=(10, 6))

    plt.hist(
        all_spearman,
        bins=30,
        color="lightgreen",
        edgecolor="black",
    )

    plt.axvline(
        0,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label="ρ = 0",
    )

    plt.title(
        f"Spearman Correlation Distribution\n"
        f"L={L}, T={T}, Slice Height={piece_height}, "
        f"Seeds={num_seed_iterations}"
    )

    plt.xlabel("Spearman Correlation (ρ)")
    plt.ylabel("Number of Rules")
    plt.legend()

    spearman_path = (
        "Saved Figures/Simplicity_Bias_Spearman_Distribution.png"
    )
    plt.savefig(spearman_path)

    print(f"\nSaved slope histogram to:\n{slope_path}")
    print(f"Saved Spearman histogram to:\n{spearman_path}")

    plt.show()


if __name__ == "__main__":
    main()