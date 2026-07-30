import numpy as np
import zlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# CORE LOGIC: CA ENGINE & METRICS
# ==========================================
def generate_eca_pattern(rule: int, ic: np.ndarray, steps: int = 20) -> np.ndarray:
    rule_bin = np.array([int(x) for x in np.binary_repr(rule, 8)], dtype=np.uint8)[::-1]
    pattern = np.zeros((steps, len(ic)), dtype=np.uint8)
    pattern[0] = ic
    for t in range(1, steps):
        left, center, right = np.roll(pattern[t - 1], 1), pattern[t - 1], np.roll(pattern[t - 1], -1)
        idx = 4 * left + 2 * center + right
        pattern[t] = rule_bin[idx]
    return pattern


def compute_complexity(pattern: np.ndarray) -> int:
    return len(zlib.compress(pattern.tobytes()))


def calculate_landscape_metrics(pattern1: np.ndarray, pattern2: np.ndarray):
    c1, c2 = compute_complexity(pattern1), compute_complexity(pattern2)
    combined = np.concatenate((pattern1, pattern2), axis=0)
    cond_complexity = compute_complexity(combined) - c1
    asymmetry = abs(c1 - c2)
    return cond_complexity, asymmetry


# ==========================================
# EXPERIMENT 4: 2D DENSITY MAPS
# ==========================================
def experiment_4_density_maps(num_pairs=200):
    print(f"Running Exp 4: Generating Density Maps with {num_pairs} random IC pairs per rule...")

    # Ground truth archetypal rules from Petak et al.
    archetypes = {
        "Shared Peaks (Plasticity)": [50, 70],
        "Tracking (Evolvability)": [54, 30],
        "Deceptive (Hurt Fitness)": [122],
        "Independent (No Overlap)": [102, 110]
    }

    results = []

    # Generate N random IC pairs
    ic_pairs = []
    for _ in range(num_pairs):
        ic_a = np.array([1] * 11 + [0] * 11, dtype=np.uint8)
        ic_b = np.array([1] * 11 + [0] * 11, dtype=np.uint8)
        np.random.shuffle(ic_a)
        np.random.shuffle(ic_b)
        ic_pairs.append((ic_a, ic_b))

    # Run the engine
    for behavior, rules in archetypes.items():
        for rule in rules:
            for ic1, ic2 in ic_pairs:
                p1, p2 = generate_eca_pattern(rule, ic1), generate_eca_pattern(rule, ic2)
                cond_c, asym = calculate_landscape_metrics(p1, p2)

                results.append({
                    "Behavior": behavior,
                    "Rule": rule,
                    "Cond_Complexity": cond_c,
                    "Asymmetry": asym
                })

    df = pd.DataFrame(results)

    # ==========================================
    # PLOTTING THE 2x2 DENSITY GRID
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), sharex=True, sharey=True)
    axes = axes.flatten()

    # Define color maps for the density gradients
    palettes = {
        "Shared Peaks (Plasticity)": "Greens",
        "Tracking (Evolvability)": "Blues",
        "Deceptive (Hurt Fitness)": "Reds",
        "Independent (No Overlap)": "Oranges"
    }

    for idx, (behavior, cmap) in enumerate(palettes.items()):
        ax = axes[idx]
        subset = df[df['Behavior'] == behavior]

        # 1. Plot KDE density contours
        sns.kdeplot(
            data=subset, x='Cond_Complexity', y='Asymmetry',
            cmap=cmap, fill=True, thresh=0.05, alpha=0.7, levels=10, ax=ax
        )

        # 2. Overlay a light scatter plot to show the raw sample distribution
        dark_color = sns.color_palette(cmap)[-2]  # Grab a dark shade from the cmap
        sns.scatterplot(
            data=subset, x='Cond_Complexity', y='Asymmetry',
            color=dark_color, s=15, alpha=0.2, edgecolor=None, ax=ax
        )

        # 3. Add the interpretative boundary lines for reference
        ax.axvline(x=20, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=15, color='gray', linestyle='--', alpha=0.5)

        # Formatting
        ax.set_title(f"{behavior}\n(Rules: {archetypes[behavior]})", fontsize=14, fontweight='bold')
        ax.set_xlabel("Conditional Complexity: C(IC2|IC1)", fontsize=12)
        ax.set_ylabel("Raw Asymmetry: |C(IC1) - C(IC2)|", fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Exp 4: Probability Density of Evolutionary Archetypes\n({num_pairs} Random Landscape Pairs Simulated per Rule)",
        fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    experiment_4_density_maps(num_pairs=1000)