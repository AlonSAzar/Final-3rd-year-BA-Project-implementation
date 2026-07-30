import numpy as np
import zlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


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
# EXPERIMENT 1: ANNOTATED PREDICTOR SPACE
# ==========================================
def experiment_1_annotated_predictor_space():
    print("Running Annotated Exp 1: 105 Landscape Pairs Analysis...")
    rules = [18, 22, 30, 50, 54, 62, 70, 90, 94, 102, 110, 122, 126, 150, 154]

    # Ground truth mappings straight from the Petak et al. PDF
    behavior_map = {
        50: "Shared Peaks (Plasticity)",
        70: "Shared Peaks (Plasticity)",
        154: "Shared Peaks (Plasticity)",
        54: "Tracking (Evolvability)",
        30: "Tracking (Evolvability)",
        122: "Deceptive (Hurt Fitness)",
        102: "Independent (No Overlap)",
        110: "Independent (No Overlap)"
    }

    ic_pairs = []
    # 1 manual pair
    ic_pairs.append((np.array([1 if i % 5 == 0 else 0 for i in range(22)], dtype=np.uint8),
                     np.array([1 if i % 3 == 0 else 0 for i in range(22)], dtype=np.uint8)))
    # 50 random pairs
    for _ in range(50):
        ic_a, ic_b = np.array([1] * 11 + [0] * 11, dtype=np.uint8), np.array([1] * 11 + [0] * 11, dtype=np.uint8)
        np.random.shuffle(ic_a);
        np.random.shuffle(ic_b)
        ic_pairs.append((ic_a, ic_b))

    results = []
    for rule in rules:
        behavior = behavior_map.get(rule, "Unclassified")
        for pair_idx, (ic1, ic2) in enumerate(ic_pairs):
            p1, p2 = generate_eca_pattern(rule, ic1), generate_eca_pattern(rule, ic2)
            cond_c, asym = calculate_landscape_metrics(p1, p2)

            results.append({
                "Rule": f"Rule {rule}",
                "Behavior": behavior,
                "IC_Pair": "Manual (Analyzed in Paper)" if pair_idx == 0 else "Random",
                "Cond_Complexity": cond_c,
                "Asymmetry": asym
            })

    df = pd.DataFrame(results)

    # Define custom palette for the known behaviors
    palette = {
        "Shared Peaks (Plasticity)": "green",
        "Tracking (Evolvability)": "blue",
        "Deceptive (Hurt Fitness)": "red",
        "Independent (No Overlap)": "darkorange",
        "Unclassified": "lightgray"
    }

    plt.figure(figsize=(12, 9))

    # Plot unclassified rules first (so they are in the background)
    sns.scatterplot(data=df[df['Behavior'] == 'Unclassified'],
                    x='Cond_Complexity', y='Asymmetry',
                    color='lightgray', style='IC_Pair', s=100, alpha=0.5, legend=False)

    # Plot classified rules with distinct colors
    sns.scatterplot(data=df[df['Behavior'] != 'Unclassified'],
                    x='Cond_Complexity', y='Asymmetry',
                    hue='Behavior', style='IC_Pair', palette=palette,
                    s=200, alpha=0.9, edgecolor='black')

    # Annotate the specific Manual IC points that the paper heavily analyzed
    manual_df = df[(df['IC_Pair'] == 'Manual (Analyzed in Paper)') & (df['Behavior'] != 'Unclassified')]
    for _, row in manual_df.iterrows():
        plt.text(row['Cond_Complexity'] + 1, row['Asymmetry'] + 1,
                 row['Rule'], fontsize=10, fontweight='bold', color=palette[row['Behavior']])

    # Overlay interpretative boundary lines
    plt.axvline(x=20, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=15, color='gray', linestyle='--', alpha=0.5)

    plt.title("Exp 1: Information Theory Predicts Evolutionary Dynamics\n(Ground Truth from Petak et al. Overlaid)",
              fontsize=14, fontweight='bold')
    plt.xlabel("Conditional Complexity: C(IC2|IC1)", fontsize=12)
    plt.ylabel("Raw Asymmetry: |C(IC1) - C(IC2)|", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Paper's Finding")
    plt.tight_layout()
    plt.show()

# ==========================================
# EXPERIMENT 2: SIMPLICITY BIAS & BASINS
# ==========================================

def sigmoid(x, a=10, c=5): return 1 / (1 + np.exp(-a * x + c))


def simulate_grn(w_int, w_comm, ic, steps=20, num_cells=22, num_genes=4):
    state = np.zeros((num_cells, num_genes))
    state[:, 0] = ic
    phenotype = np.zeros((steps, num_cells))
    phenotype[0] = ic

    for t in range(1, steps):
        rho_int = sigmoid(state @ w_int)
        left = np.roll(rho_int[:, 1], 1)
        right = np.roll(rho_int[:, 2], -1)
        state = sigmoid(np.column_stack((left, rho_int, right)) @ w_comm)
        phenotype[t] = state[:, 0]
    return phenotype


def experiment_2_fitness_transfer(rule_plastic=50, rule_deceptive=122, num_grns=2000):
    print(f"Running Exp 2: Basin Overlap...")
    ic1 = np.array([1 if i % 5 == 0 else 0 for i in range(22)], dtype=np.uint8)
    ic2 = np.array([1 if i % 3 == 0 else 0 for i in range(22)], dtype=np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)

    for ax, rule, title in zip(axes, [rule_plastic, rule_deceptive],
                               ["Shared Peaks (Rule 50)", "Deceptive Plateau (Rule 122)"]):
        t1, t2 = generate_eca_pattern(rule, ic1), generate_eca_pattern(rule, ic2)
        fit1, fit2 = [], []

        for _ in range(num_grns):
            w_int = np.random.normal(0, 1, (4, 4))
            w_comm = np.random.normal(0, 1, (6, 4))
            p1, p2 = simulate_grn(w_int, w_comm, ic1), simulate_grn(w_int, w_comm, ic2)

            fit1.append(1 - (np.sum(np.abs(p1 - t1)) / 440))
            fit2.append(1 - (np.sum(np.abs(p2 - t2)) / 440))

        corr, _ = pearsonr(fit1, fit2)

        # Plotting the density of the fitness landscape
        sns.kdeplot(x=fit1, y=fit2, cmap="Blues", fill=True, thresh=0.05, ax=ax)
        sns.scatterplot(x=fit1, y=fit2, alpha=0.3, s=15, color="darkblue", ax=ax)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)  # y=x line

        ax.set_title(f"{title}\nFitness Correlation: r = {corr:.3f}")
        ax.set_xlabel("Fitness in Environment 1")
        ax.set_ylabel("Fitness in Environment 2")
        ax.set_xlim(0.4, 1.0)  # Zooming in on realistic GRN bounds
        ax.set_ylim(0.4, 1.0)

    plt.tight_layout()
    plt.show()


# ==========================================
# EXPERIMENT 3: PREDICTIVE LANDSCAPE DESIGN
# ==========================================

def experiment_3_design_principle_search(target_cond_c_max=20, target_asym_min=20):
    print("Running Exp 3: Searching for Experimental Controls...")
    ic1 = np.array([1] * 11 + [0] * 11, dtype=np.uint8);
    np.random.shuffle(ic1)
    ic2 = np.array([1] * 11 + [0] * 11, dtype=np.uint8);
    np.random.shuffle(ic2)

    results = []
    candidates = []

    for rule in range(256):
        p1, p2 = generate_eca_pattern(rule, ic1), generate_eca_pattern(rule, ic2)

        # Filter dead rules (trivial all-white/all-black patterns)
        if np.mean(p1) < 0.05 or np.mean(p1) > 0.95: continue

        cond_c, asym = calculate_landscape_metrics(p1, p2)
        results.append({"Rule": rule, "Cond_C": cond_c, "Asym": asym})

        if cond_c <= target_cond_c_max and asym >= target_asym_min:
            candidates.append({"Rule": rule, "Cond_C": cond_c, "Asym": asym})

    df = pd.DataFrame(results)
    c_df = pd.DataFrame(candidates)

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='Cond_C', y='Asym', color='lightgray', s=50, label='All Non-Trivial Rules')

    if not c_df.empty:
        sns.scatterplot(data=c_df, x='Cond_C', y='Asym', color='darkorange', s=150, edgecolor='black',
                        label='Target Regimes (Tracking)')
        for i, row in c_df.iterrows():
            plt.text(row['Cond_C'] + 0.5, row['Asym'] + 0.5, f"R{int(row['Rule'])}", fontsize=9, fontweight='bold')

    # Highlight the target bounding box
    plt.axvspan(0, target_cond_c_max, ymin=(target_asym_min / max(df['Asym'])), ymax=1, color='orange', alpha=0.1)

    plt.title(
        f"Exp 3: Rule Space Search\nTargeting: Low Cond Complexity (≤{target_cond_c_max}), High Asymmetry (≥{target_asym_min})")
    plt.xlabel("Conditional Complexity C(IC2|IC1)")
    plt.ylabel("Raw Asymmetry |C(IC1) - C(IC2)|")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    experiment_1_annotated_predictor_space()
    experiment_2_fitness_transfer()
    experiment_3_design_principle_search()