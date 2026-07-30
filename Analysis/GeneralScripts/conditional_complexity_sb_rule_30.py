import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
import scipy.stats as stats
from multiprocessing import Pool, cpu_count
from numba import njit
from sklearn.linear_model import QuantileRegressor

# Add project root to sys.path to resolve imports properly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Core.ComplexityMeasures.conditional_complexity import ZlibConditionalComplexity


# --- Numba Engine & Coarse-Graining ---
@njit(fastmath=True)
def _simulate_1d_numba(rule: int, seed: np.ndarray, T: int) -> np.ndarray:
    L = len(seed)
    output = np.zeros((T, L), dtype=np.uint8)
    output[0] = seed

    for t in range(1, T):
        prev = output[t - 1]
        curr = output[t]

        # Boundary i = 0
        left, center, right = prev[L - 1], prev[0], prev[1]
        nb = (left << 2) | (center << 1) | right
        curr[0] = (rule >> nb) & 1

        # Inner loop
        for i in range(1, L - 1):
            left, center, right = prev[i - 1], prev[i], prev[i + 1]
            nb = (left << 2) | (center << 1) | right
            curr[i] = (rule >> nb) & 1

        # Boundary i = L - 1
        left, center, right = prev[L - 2], prev[L - 1], prev[0]
        nb = (left << 2) | (center << 1) | right
        curr[L - 1] = (rule >> nb) & 1

    return output


@njit(fastmath=True)
def _coarse_grain_numba(phenotype: np.ndarray, block_size: int) -> np.ndarray:
    T, L = phenotype.shape
    new_T, new_L = T // block_size, L // block_size

    total_sum = 0
    for i in range(T):
        for j in range(L):
            total_sum += phenotype[i, j]
    threshold = total_sum / float(T * L)

    cg = np.zeros((new_T, new_L), dtype=np.uint8)
    block_area = float(block_size * block_size)

    for bi in range(new_T):
        for bj in range(new_L):
            b_sum = 0
            r_start = bi * block_size
            c_start = bj * block_size
            for r in range(r_start, r_start + block_size):
                for c in range(c_start, c_start + block_size):
                    b_sum += phenotype[r, c]
            if (b_sum / block_area) > threshold:
                cg[bi, bj] = 1

    return cg


@njit(fastmath=True)
def _simulate_batch_numba(rule: int, num_seeds: int, L: int, T: int, block_size: int) -> np.ndarray:
    target_T = T // block_size
    target_L = L // block_size
    packed_pairs = np.zeros(num_seeds, dtype=np.uint64)

    for s in range(num_seeds):
        # 1. Seed X
        seed_x = np.random.randint(0, 2, size=L).astype(np.uint8)
        pheno_x = _simulate_1d_numba(rule, seed_x, T)
        cg_x = _coarse_grain_numba(pheno_x, block_size)

        # 2. Seed Y (1-bit flip)
        bit_idx = np.random.randint(0, L)
        seed_y = seed_x.copy()
        seed_y[bit_idx] ^= 1
        pheno_y = _simulate_1d_numba(rule, seed_y, T)
        cg_y = _coarse_grain_numba(pheno_y, block_size)

        # 3. Pack 4x4 cg_x and cg_y into uint64
        x_val = np.uint64(0)
        y_val = np.uint64(0)
        idx = 0
        for r in range(target_T):
            for c in range(target_L):
                if cg_x[r, c] == 1:
                    x_val |= (np.uint64(1) << idx)
                if cg_y[r, c] == 1:
                    y_val |= (np.uint64(1) << idx)
                idx += 1

        packed_pairs[s] = (x_val << np.uint64(32)) | y_val

    return packed_pairs


def _worker_task(args):
    rule, num_seeds, large_L, large_T, block_size = args
    return _simulate_batch_numba(rule, num_seeds, large_L, large_T, block_size)


def _compute_zlib_batch(batch_pair_data):
    """Worker task to parallelize Zlib complexity calculation."""
    metric = ZlibConditionalComplexity()
    results = []
    for cg_x, cg_y in batch_pair_data:
        results.append(metric.calculate(cg_x, cg_y))
    return results


def run_conditional_complexity_sb_ruleX(
        rule=89, large_L=68, large_T=68, block_size=17, num_seeds=500000, num_workers=None
):
    if num_workers is None:
        num_workers = os.cpu_count() or 4

    target_L, target_T = large_L // block_size, large_T // block_size

    print(f"Running Rule {rule} Conditional Complexity SB Experiment...")
    print(
        f"Grid: {large_L}x{large_T} -> CG: {target_L}x{target_T} | {num_seeds:,} seeds across {num_workers} processes")

    print("Warming up Numba JIT compiler...")
    _simulate_batch_numba(rule, 10, large_L, large_T, block_size)

    # 1. Parallel Simulation
    chunk_size = max(5000, num_seeds // (num_workers * 10))
    tasks = []
    remaining = num_seeds
    while remaining > 0:
        current_chunk = min(chunk_size, remaining)
        tasks.append((rule, current_chunk, large_L, large_T, block_size))
        remaining -= current_chunk

    results = []
    with Pool(processes=num_workers) as pool:
        for batch in tqdm(pool.imap_unordered(_worker_task, tasks), total=len(tasks), desc="Simulating Seeds"):
            results.append(batch)

    print("Concatenating results and aggregating transition pairs...")
    all_packed = np.concatenate(results)

    # 2. Aggregation
    unique_packed, counts = np.unique(all_packed, return_counts=True)
    total_transitions = len(all_packed)
    num_unique = len(unique_packed)

    print(f"Post-processing: Computing Zlib complexity for {num_unique:,} unique observed transition pairs...")

    # 3. Vectorized Bit-Unpacking (Dynamic for any CG dimensions)
    x_vals = (unique_packed >> np.uint64(32)).astype(np.uint32)
    y_vals = (unique_packed & np.uint64(0xFFFFFFFF)).astype(np.uint32)

    # Calculate number of cells in the coarse-grained grid
    num_bits = target_T * target_L
    bits = np.arange(num_bits, dtype=np.uint32)

    cg_x_all = ((x_vals[:, None] >> bits[None, :]) & 1).astype(np.uint8).reshape(num_unique, target_T, target_L)
    cg_y_all = ((y_vals[:, None] >> bits[None, :]) & 1).astype(np.uint8).reshape(num_unique, target_T, target_L)

    # Prepare pair tuples for parallel Zlib processing
    pair_data = [(cg_x_all[i], cg_y_all[i]) for i in range(num_unique)]

    # 4. Parallel Zlib Complexity Computation
    zlib_batch_size = max(1000, num_unique // (num_workers * 10))
    zlib_batches = [pair_data[i:i + zlib_batch_size] for i in range(0, num_unique, zlib_batch_size)]

    Ks_list = []
    with Pool(processes=num_workers) as pool:
        for k_batch in tqdm(pool.imap(_compute_zlib_batch, zlib_batches), total=len(zlib_batches),
                            desc="Calculating Complexity"):
            Ks_list.extend(k_batch)

    Ks = np.array(Ks_list)
    log_probs = np.log10(counts / total_transitions)

    return Ks, log_probs, total_transitions, target_L, target_T


def plot_conditional_sb(Ks, log_probs, total_samples, rule, large_L, large_T, cg_L, cg_T):
    if len(Ks) < 2:
        print(f"Error: Not enough unique transitions observed (len(Ks) = {len(Ks)}). Check coarse-graining threshold.")
        return

    plt.figure(figsize=(10, 7))
    plt.scatter(Ks, log_probs, alpha=0.5, c='purple', s=20, label=f'Transitions (1-bit seed flip)')

    pearson_r, _ = stats.pearsonr(Ks, log_probs)
    spearman_rho, _ = stats.spearmanr(Ks, log_probs)

    # Upper bound: Extract maximum log_prob for each unique K value (Upper Envelope)
    unique_ks = np.unique(Ks)
    if len(unique_ks) > 1:
        # Get the highest probability point for each complexity value K
        max_log_probs = np.array([np.max(log_probs[Ks == k]) for k in unique_ks])

        # Fit 95th percentile quantile regression on the UPPER ENVELOPE points
        X_env = unique_ks.reshape(-1, 1)
        y_env = max_log_probs

        qr = QuantileRegressor(quantile=0.95, alpha=0)
        qr.fit(X_env, y_env)

        x_line = np.linspace(min(Ks), max(Ks), 100).reshape(-1, 1)
        y_line = qr.predict(x_line)

        plt.plot(x_line, y_line, color='red', linestyle='--', label='Upper Bound (Top Points 95th Quantile)')

    plt.xlabel("Conditional Complexity K(y|x) (Zlib Bytes)")
    plt.ylabel("Log10 Conditional Probability P(y|x)")
    plt.title(f"Conditional Simplicity Bias: Coarse-Grained Rule {rule}\n"
              f"Large: {large_L}x{large_T} -> CG: {cg_L}x{cg_T}\n"
              f"Pearson r={pearson_r:.3f}, Spearman ρ={spearman_rho:.3f}")

    plt.legend()
    plt.grid(True, alpha=0.2)

    save_dir = os.path.join("Analysis", "GeneralScripts", "Saved Figures")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"conditional_complexity_sb_rule{rule}.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    RULE = 154
    LARGE_L = 6
    LARGE_T = 6
    BLOCK_SIZE = 1
    NUM_SEEDS = 5000

    Ks, log_probs, total, cg_L, cg_T = run_conditional_complexity_sb_ruleX(
        rule=RULE,
        large_L=LARGE_L,
        large_T=LARGE_T,
        block_size=BLOCK_SIZE,
        num_seeds=NUM_SEEDS
    )

    plot_conditional_sb(Ks, log_probs, total, RULE, LARGE_L, LARGE_T, cg_L, cg_T)