import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import zlib
from tqdm import tqdm


# --- 1. The Asynchronous Engine ---
@njit
def simulate_async_ca(rule_lut, seed, T, sync_prob=0.95):
    """
    sync_prob: The probability that a cell updates normally.
    1.0 = Standard CA (Perfect global clock).
    0.95 = 5% chance a cell "misses" the tick and keeps its old state.
    """
    L = len(seed)
    output = np.zeros((T, L), dtype=np.uint8)
    output[0] = seed

    for t in range(1, T):
        for i in range(L):
            left = output[t - 1][(i - 1) % L]
            center = output[t - 1][i]
            right = output[t - 1][(i + 1) % L]
            neighborhood = (left << 2) | (center << 1) | right

            # Asynchronous check
            if np.random.random() < sync_prob:
                output[t][i] = rule_lut[neighborhood]
            else:
                # Timing failure: cell retains its previous state
                output[t][i] = center

    return output


def get_lut(rule):
    return np.array([(rule >> i) & 1 for i in range(7, -1, -1)], dtype=np.uint8)


# --- 2. The Experiment ---
def test_synchronization_cliff(L=128, T=128, sync_prob=0.95, num_seeds=10):
    print(f"Running Asynchronous Collapse Test (Sync Prob = {sync_prob})...")

    baseline_complexities = []
    robustness_scores = []
    rule_labels = []

    for rule in tqdm(range(256)):
        lut = get_lut(rule)

        rule_base_k = []
        rule_robustness = []

        for _ in range(num_seeds):
            seed = np.random.randint(0, 2, size=L, dtype=np.uint8)

            # 1. Run Perfect Clock (Control)
            sync_history = simulate_async_ca(lut, seed, T, sync_prob=1.0)

            # 2. Run Broken Clock (Experiment)
            async_history = simulate_async_ca(lut, seed, T, sync_prob=sync_prob)

            # 3. Measure Complexity (Use Zlib on the last 50 rows to ignore initial noise)
            sync_k = len(zlib.compress(sync_history[-50:].tobytes()))
            async_k = len(zlib.compress(async_history[-50:].tobytes()))

            # 4. Calculate Robustness (Ratio of Complexities)
            # If the system collapses to chaos, async_k explodes (Ratio > 1)
            # If it dies, async_k drops to ~10 (Ratio close to 0)
            # We want a metric where 1.0 means "Structure maintained", 0.0 means "Destroyed"

            # Deviation from 1.0 means failure
            deviation = abs(sync_k - async_k) / max(sync_k, 1)
            robustness = max(0.0, 1.0 - deviation)

            rule_base_k.append(sync_k)
            rule_robustness.append(robustness)

        baseline_complexities.append(np.mean(rule_base_k))
        robustness_scores.append(np.mean(rule_robustness))
        rule_labels.append(rule)

    # --- 3. Plotting the "Cliff" ---
    import pandas as pd
    import plotly.express as px
    import os

    # Matplotlib Static Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(baseline_complexities, robustness_scores, alpha=0.6, edgecolors='k')

    # Highlight specific rules
    for r in [30, 110, 192, 168]:
        idx = rule_labels.index(r)
        plt.scatter(baseline_complexities[idx], robustness_scores[idx], s=100, label=f"Rule {r}", zorder=5)

    plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Collapse Threshold')

    plt.title(
        f"The Synchronization Cliff (Sync Probability: {sync_prob * 100}%)\nDo complex rules collapse without a global clock?")
    plt.xlabel("Baseline Phenotype Complexity (Perfect Clock)")
    plt.ylabel("Structural Robustness (1.0 = Pattern Maintained)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Plotly Interactive HTML
    df = pd.DataFrame({
        'Baseline_Complexity': baseline_complexities,
        'Robustness': robustness_scores,
        'Rule': rule_labels
    })

    fig = px.scatter(
        df, 
        x='Baseline_Complexity', 
        y='Robustness',
        hover_data=['Rule', 'Baseline_Complexity', 'Robustness'],
        title=f"Interactive Cliff (Sync Prob: {sync_prob*100}%)",
        template="plotly_white"
    )

    fig.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
    
    save_dir = os.path.join("Saved Figures", "HTML")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename = os.path.join(save_dir, f"AsynchronyCliff_Prob_{int(sync_prob*100)}.html")
    fig.write_html(filename)
    print(f"Interactive HTML saved to: {filename}")


if __name__ == "__main__":
    test_synchronization_cliff()