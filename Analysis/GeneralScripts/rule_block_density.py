import numpy as np
import matplotlib.pyplot as plt


def run_eca(rule_number, L=68, T=68):
    """Simulates 1D ECA for grid size LxT with periodic boundaries."""
    rule_bin = np.array([(rule_number >> i) & 1 for i in range(8)], dtype=np.uint8)
    state = np.random.randint(0, 2, size=L, dtype=np.uint8)
    grid = np.zeros((T, L), dtype=np.uint8)
    grid[0] = state

    for t in range(1, T):
        left = np.roll(state, 1)
        right = np.roll(state, -1)
        neighborhood = (left << 2) | (state << 1) | right
        state = rule_bin[neighborhood]
        grid[t] = state

    return grid


def analyze_block_clustering(rule_number, large_L=68, large_T=68, block_size=17, num_samples=1000):
    """
    Analyzes local block density distributions for a given rule.
    """
    target_T, target_L = large_T // block_size, large_L // block_size
    cells_per_block = block_size * block_size
    majority_threshold = cells_per_block / 2.0  # 144.5 for 17x17

    all_block_densities = []
    blocks_crossing_threshold = 0
    total_blocks = 0
    valid_cg_grids = 0  # Grids that don't collapse to all zeros

    for _ in range(num_samples):
        grid = run_eca(rule_number, L=large_L, T=large_T)
        cg_grid = np.zeros((target_T, target_L), dtype=int)

        for i in range(target_T):
            for j in range(target_L):
                block = grid[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                ones_count = np.sum(block)
                block_density = ones_count / cells_per_block
                all_block_densities.append(block_density)

                if ones_count > majority_threshold:
                    blocks_crossing_threshold += 1
                    cg_grid[i, j] = 1
                total_blocks += 1

        # A CG grid is valid for complexity experiments if it contains at least one active block
        if np.any(cg_grid > 0):
            valid_cg_grids += 1

    all_block_densities = np.array(all_block_densities)

    return {
        "global_mean": np.mean(all_block_densities),
        "block_std": np.std(all_block_densities),
        "pct_blocks_crossing_50pct": (blocks_crossing_threshold / total_blocks) * 100,
        "pct_non_zero_cg_grids": (valid_cg_grids / num_samples) * 100,
        "densities": all_block_densities
    }


if __name__ == "__main__":
    rules = [22, 146]
    results = {}

    print(
        f"{'Rule':<8} | {'Global Mean':<12} | {'Block Std Dev':<15} | {'% Blocks > 50%':<18} | {'% Non-Zero CG Grids'}")
    print("-" * 80)

    for rule in rules:
        res = analyze_block_clustering(rule_number=rule, num_samples=2000)
        results[rule] = res
        print(
            f"Rule {rule:<3} | {res['global_mean']:<12.4f} | {res['block_std']:<15.4f} | {res['pct_blocks_crossing_50pct']:<18.2f}% | {res['pct_non_zero_cg_grids']:.2f}%")

    # Plot Block Density Distributions
    plt.figure(figsize=(10, 5))
    for rule in rules:
        plt.hist(results[rule]["densities"], bins=40, alpha=0.6,
                 label=f"Rule {rule} (Std={results[rule]['block_std']:.3f})")

    plt.axvline(0.50, color='red', linestyle='--', label='50% Majority Threshold')
    plt.xlabel("17x17 Block Density (Fraction of 1s)")
    plt.ylabel("Frequency")
    plt.title("Local Block Density Distribution: Rule 22 vs Rule 146")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()