import numpy as np
import matplotlib.pyplot as plt
import zlib
from tqdm import tqdm

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA


def get_zlib_complexity(engine, rule, seeds):
    """Calculates average complexity for a specific rule over fixed seeds."""
    total_k = 0
    for seed in seeds:
        # Get the phenotype (e.g., last 50 rows to capture the attractor)
        img = engine.run(rule, seed)[-50:]
        total_k += len(zlib.compress(img.tobytes()))
    return total_k / len(seeds)


def map_rule_neighborhoods(L=128, T=128, num_seeds=50):
    engine = ElementaryCA(L, T)

    # Use FIXED seeds so we are only measuring Rule differences, not Seed noise
    np.random.seed(42)
    fixed_seeds = [engine.generate_seed() for _ in range(num_seeds)]

    base_complexities = np.zeros(256)
    neighbor_avg_complexities = np.zeros(256)

    print("Exhaustively mapping the 8-Dimensional Rule Hypercube...")
    for rule in tqdm(range(256)):
        # 1. Base Complexity
        base_k = get_zlib_complexity(engine, rule, fixed_seeds)
        base_complexities[rule] = base_k

        # 2. Neighbor Complexities
        neighbors_k = []
        for bit in range(8):
            # Deterministically flip one bit
            mutant_rule = rule ^ (1 << bit)
            mut_k = get_zlib_complexity(engine, mutant_rule, fixed_seeds)
            neighbors_k.append(mut_k)

        neighbor_avg_complexities[rule] = np.mean(neighbors_k)

    # --- Plotting the Correlation ---
    plt.figure(figsize=(8, 8))
    plt.scatter(base_complexities, neighbor_avg_complexities, alpha=0.6, edgecolors='k')

    # Draw the y=x line (Perfect Correlation)
    max_val = max(np.max(base_complexities), np.max(neighbor_avg_complexities))
    min_val = min(np.min(base_complexities), np.min(neighbor_avg_complexities))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')

    # Calculate correlation
    from scipy.stats import pearsonr
    corr, _ = pearsonr(base_complexities, neighbor_avg_complexities)

    plt.title(f"Rule Space Landscape Smoothness\nPearson Correlation: {corr:.3f}")
    plt.xlabel("Complexity of Rule $R$")
    plt.ylabel("Average Complexity of $R$'s 8 Neighbors")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    map_rule_neighborhoods()