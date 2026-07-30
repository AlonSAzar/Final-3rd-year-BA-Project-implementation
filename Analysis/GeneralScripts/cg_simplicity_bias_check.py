import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from collections import Counter
import scipy.stats as stats
from sklearn.linear_model import QuantileRegressor

# Add project root to sys.path to resolve imports properly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.ComplexityMeasures.complexity import ZlibComplexity
from Experiments.experiments import shuffle_space_time

def coarse_grain_phenotype(phenotype, block_size=8, shuffle_before=False):
    """
    Coarse-grains a 2D phenotype by block averaging.
    Args:
        phenotype: 2D numpy array (T, L)
        block_size: size of the squares to average
        shuffle_before: if True, shuffles pixels before coarse graining
    Returns:
        Coarse-grained 2D array of reduced size.
    """
    if shuffle_before:
        phenotype = shuffle_space_time(phenotype)

    T, L = phenotype.shape
    new_T = T // block_size
    new_L = L // block_size
    
    cg_phenotype = np.zeros((new_T, new_L), dtype=int)
    
    for i in range(new_T):
        for j in range(new_L):
            block = phenotype[i*block_size : (i+1)*block_size, 
                              j*block_size : (j+1)*block_size]
            # Majority rule for coarse graining
            cg_phenotype[i, j] = 1 if np.sum(block) > (block.size // 2) else 0
            
    return cg_phenotype

def run_coarse_grained_simplicity_bias(L=64, T=64, block_size=4, num_samples_per_rule=2, shuffle=False, shuffle_after=False):
    """
    Checks for simplicity bias in coarse-grained phenotypes.
    L, T define the size of the LARGE grid.
    block_size defines the coarse-graining reduction factor.
    """
    engine = ElementaryCA(L=L, T=T)
    metric = ZlibComplexity()
    
    target_L, target_T = L // block_size, T // block_size
    
    pattern_counts = Counter()
    pattern_to_complexity = {}
    
    print(f"Running Experiment: Large Grid {L}x{T} -> Coarse-Grained {target_L}x{target_T}")
    print(f"Block Size: {block_size}, Samples per Rule: {num_samples_per_rule}, Shuffled Before: {shuffle}, Shuffled After: {shuffle_after}")
    
    # We iterate over all rules to get a global distribution of phenotypes
    for rule in tqdm(range(256), desc="Rules"):
        for _ in range(num_samples_per_rule):
            seed = engine.generate_seed(seed_type="random")
            # Run CA (full history including t=0 to get clean blocks)
            phenotype = engine.run(rule, seed)
            
            # Coarse grain the phenotype
            cg_phenotype = coarse_grain_phenotype(phenotype, block_size, shuffle_before=shuffle)

            if shuffle_after:
                cg_phenotype = shuffle_space_time(cg_phenotype)
            
            p_hash = cg_phenotype.tobytes()
            
            if p_hash not in pattern_to_complexity:
                pattern_to_complexity[p_hash] = metric.calculate(cg_phenotype)
            
            pattern_counts[p_hash] += 1
            
    # Prepare data for plotting
    unique_hashes = list(pattern_counts.keys())
    total_samples = sum(pattern_counts.values())
    
    complexities = []
    log_probs = []
    
    for h in unique_hashes:
        k = pattern_to_complexity[h]
        prob = pattern_counts[h] / total_samples
        complexities.append(k)
        log_probs.append(np.log10(prob))
        
    return np.array(complexities), np.array(log_probs), total_samples, target_L, target_T

def plot_simplicity_bias(Ks, log_probs, total_samples, large_L, large_T, cg_L, cg_T, block_size):
    plt.figure(figsize=(10, 7))
    plt.scatter(Ks, log_probs, alpha=0.5, c='darkblue', s=20, label='Coarse-Grained Phenotypes')
    
    # Calculate correlations
    pearson_r, p_pearson = stats.pearsonr(Ks, log_probs)
    spearman_rho, p_spearman = stats.spearmanr(Ks, log_probs)
    
    # Upper bound fitting
    unique_ks = np.unique(Ks)
    r_squared = 0
    if len(unique_ks) > 1:
        max_log_probs = np.array([np.max(log_probs[Ks == k]) for k in unique_ks])
        
        # Quantile Regression
        qr = QuantileRegressor(quantile=0.95, alpha=0)
        qr.fit(unique_ks.reshape(-1, 1), max_log_probs)
        slope = qr.coef_[0]
        intercept = qr.intercept_

        x_line = np.linspace(min(Ks), max(Ks), 100)
        y_line = slope * x_line + intercept
        
        # Calculate R^2 or similar if needed, for now use standard linregress for R^2 value
        _, _, r_val, _, _ = stats.linregress(unique_ks, max_log_probs)
        r_squared = r_val**2

        plt.plot(x_line, y_line, color='red', linestyle='--', label=f'Upper Bound (QR 0.95)')
        r_squared = r_val**2
        
    plt.xlabel("Complexity K (Zlib Bytes)")
    plt.ylabel("Log10 Probability P(x)")
    
    title = (f"Simplicity Bias in Coarse-Grained CA\n"
             f"Large Grid: {large_L}x{large_T} -> Coarse-Grained: {cg_L}x{cg_T} (Block: {block_size})\n"
             f"Pearson r={pearson_r:.3f}, Spearman ρ={spearman_rho:.3f}")
    plt.title(title)
    
    # Add stats box
    stats_text = (f"Pearson r: {pearson_r:.3f})\n"
                  f"Spearman ρ: {spearman_rho:.3f})\n"
                  f"Upper Bound R²: {r_squared:.3f}\n"
                  f"Unique Patterns: {len(Ks)}\n"
                  f"Total Samples: {total_samples}")
    plt.text(0.02, 0.05, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    
    save_dir = os.path.join("Analysis", "GeneralScripts", "Saved Figures")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"cg_simplicity_bias_{cg_L}x{cg_T}.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # --- CONFIGURATION ---
    LARGE_L = 64
    LARGE_T = 64
    BLOCK_SIZE = 4
    SAMPLES_PER_RULE = 50
    # ---------------------

    Ks, log_probs, total, cg_L, cg_T = run_coarse_grained_simplicity_bias(
        L=LARGE_L, 
        T=LARGE_T, 
        block_size=BLOCK_SIZE, 
        num_samples_per_rule=SAMPLES_PER_RULE,
        shuffle_after=True
    )
    
    plot_simplicity_bias(Ks, log_probs, total, LARGE_L, LARGE_T, cg_L, cg_T, BLOCK_SIZE)

