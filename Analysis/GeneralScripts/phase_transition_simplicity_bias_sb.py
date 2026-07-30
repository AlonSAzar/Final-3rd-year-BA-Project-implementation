"""Something is wrong here, the correlation values are much closer to 0 than they should be"""
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm


# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Core.ComplexityMeasures.complexity import ZlibComplexity
from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA, _rule_to_lut


def get_lambda(rule_int):
    binary = [int(x) for x in bin(rule_int)[2:].zfill(8)]
    return sum(binary) / 8.0


def _make_rng(random_seed):
    if random_seed is None:
        return np.random.default_rng()
    return np.random.default_rng(random_seed)


def simulate_missed_updates(rule, seed, T, update_success_prob, rng):
    """
    Standard synchronous CA update, except each cell only updates with probability
    update_success_prob. Otherwise it keeps its previous value.
    """
    lut = _rule_to_lut(rule)
    L = len(seed)
    history = np.zeros((T, L), dtype=np.uint8)
    history[0] = seed

    for t in range(1, T):
        prev = history[t - 1]
        current = np.empty(L, dtype=np.uint8)

        for i in range(L):
            if rng.random() >= update_success_prob:
                current[i] = prev[i]
                continue

            left = prev[(i - 1) % L]
            center = prev[i]
            right = prev[(i + 1) % L]
            neighborhood = (left << 2) | (center << 1) | right
            current[i] = lut[neighborhood]

        history[t] = current

    return history


def simulate_connectivity_dropout(rule, seed, T, neighbor_connection_prob, rng):
    """
    Synchronous CA update where each neighbor link can fail independently.
    If a neighbor link fails, the missing input is replaced by the center cell.
    This is a local proxy for weakened connectivity between adjacent cells.
    """
    lut = _rule_to_lut(rule)
    L = len(seed)
    history = np.zeros((T, L), dtype=np.uint8)
    history[0] = seed

    for t in range(1, T):
        prev = history[t - 1]
        current = np.empty(L, dtype=np.uint8)

        for i in range(L):
            center = prev[i]

            if rng.random() < neighbor_connection_prob:
                left = prev[(i - 1) % L]
            else:
                left = center

            if rng.random() < neighbor_connection_prob:
                right = prev[(i + 1) % L]
            else:
                right = center

            neighborhood = (left << 2) | (center << 1) | right
            current[i] = lut[neighborhood]

        history[t] = current

    return history


def run_phase_transition_scan(
    mode,
    L=64,
    T=64,
    rules=range(256),
    num_seeds_per_rule=40,
    perturbation_values=None,
    random_seed=1234,
):
    """
    For each perturbation level, collect the complexity-frequency distribution and
    compute the Spearman correlation between complexity and log frequency.

    Returns a DataFrame with one row per perturbation level.
    """
    if perturbation_values is None:
        if mode == "connectivity":
            perturbation_values = np.linspace(1.0, 0.2, 9)
        else:
            perturbation_values = np.linspace(1.0, 0.5, 11)

    metric = ZlibComplexity()
    engine = ElementaryCA(L=L, T=T)
    rng = _make_rng(random_seed)

    rows = []
    rules = list(rules)

    print(f"Running phase-transition scan for mode={mode} (L={L}, T={T}, seeds/rule={num_seeds_per_rule})")

    for value in perturbation_values:
        phenotype_counts = Counter()
        phenotype_lookup = {}

        if mode == "asynchrony":
            update_success_prob = value
        elif mode == "missed_updates":
            update_success_prob = 1.0 - value
        elif mode == "connectivity":
            neighbor_connection_prob = value
        else:
            raise ValueError(f"Unknown mode: {mode}")

        for rule in tqdm(rules, desc=f"{mode}={value:.3f}"):
            for _ in range(num_seeds_per_rule):
                seed = engine.generate_seed(seed_type="random")

                if mode == "asynchrony":
                    phenotype = simulate_missed_updates(rule, seed, T, update_success_prob, rng)
                elif mode == "missed_updates":
                    phenotype = simulate_missed_updates(rule, seed, T, update_success_prob, rng)
                else:
                    phenotype = simulate_connectivity_dropout(rule, seed, T, neighbor_connection_prob, rng)

                phenotype = phenotype[1:]
                p_hash = phenotype.tobytes().hex()
                phenotype_counts[p_hash] += 1
                if p_hash not in phenotype_lookup:
                    phenotype_lookup[p_hash] = phenotype

        total = sum(phenotype_counts.values())
        complexities = []
        log_freqs = []

        for p_hash, count in phenotype_counts.items():
            complexities.append(metric.calculate(phenotype_lookup[p_hash]))
            log_freqs.append(np.log10(count / total))

        complexities = np.asarray(complexities, dtype=float)
        log_freqs = np.asarray(log_freqs, dtype=float)

        if len(complexities) > 1 and np.std(complexities) > 0 and np.std(log_freqs) > 0:
            rho, p_val = spearmanr(complexities, log_freqs)
        else:
            rho, p_val = np.nan, np.nan

        rows.append(
            {
                "mode": mode,
                "perturbation_value": float(value),
                "spearman_rho": float(rho) if rho is not None else np.nan,
                "p_value": float(p_val) if p_val is not None else np.nan,
                "unique_phenotypes": len(complexities),
                "total_samples": total,
            }
        )

    return pd.DataFrame(rows)


def plot_phase_transition(df, mode, save_path=None):
    if df.empty:
        raise ValueError("No results to plot.")

    if save_path is None:
        save_path = os.path.join(
            "Analysis",
            "GeneralScripts",
            "Saved Figures",
            f"phase_transition_{mode}.png",
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    mode_labels = {
        "asynchrony": "Update success probability",
        "missed_updates": "Probability a cell does not update when it should",
        "connectivity": "Neighbor connection probability",
    }

    x = df["perturbation_value"].to_numpy(dtype=float)
    y = df["spearman_rho"].to_numpy(dtype=float)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker="o", color="crimson", linewidth=2)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    plt.grid(True, alpha=0.25)
    plt.xlabel(mode_labels.get(mode, "Perturbation strength"))
    plt.ylabel("Spearman correlation between complexity and log frequency")
    plt.title(f"SB phase transition under {mode}")

    stats_text = (
        f"min ρ: {np.nanmin(y):.3f}\n"
        f"max ρ: {np.nanmax(y):.3f}\n"
        f"mean ρ: {np.nanmean(y):.3f}"
    )
    plt.text(
        0.02,
        0.05,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved plot to {save_path}")
    plt.show()


def main():
    L = 16
    T = 16
    NUM_SEEDS_PER_RULE = 100
    RULES = range(256)

    output_dir = os.path.join("Analysis", "GeneralScripts", "Saved Figures")
    os.makedirs(output_dir, exist_ok=True)

    experiments = {
        # "asynchrony": np.linspace(1.0, 0.05, 10),
        # "missed_updates": np.linspace(0.0, 0.95, 10),
        "connectivity": np.linspace(1.0, 0.05, 25),
    }

    all_results = []

    for mode, values in experiments.items():
        df = run_phase_transition_scan(
            mode=mode,
            L=L,
            T=T,
            rules=RULES,
            num_seeds_per_rule=NUM_SEEDS_PER_RULE,
            perturbation_values=values,
            random_seed=1234,
        )

        csv_path = os.path.join(output_dir, f"phase_transition_{mode}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved data to {csv_path}")

        plot_phase_transition(
            df,
            mode,
            save_path=os.path.join(output_dir, f"phase_transition_{mode}.png"),
        )

        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    combined_csv_path = os.path.join(output_dir, "phase_transition_all_modes.csv")
    combined.to_csv(combined_csv_path, index=False)
    print(f"Saved combined data to {combined_csv_path}")


if __name__ == "__main__":
    main()