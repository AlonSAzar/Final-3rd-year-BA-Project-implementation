import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from collections import Counter
import scipy.stats as stats

# Add project root to sys.path to resolve imports properly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.ComplexityMeasures.complexity import ZlibComplexity
from Experiments.experiments import shuffle_space_time

def coarse_grain_phenotype(phenotype, block_size=4, shuffle_before=False):
    """
    Coarse-grains a 2D phenotype by block averaging (majority rule).
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
            cg_phenotype[i, j] = 1 if np.sum(block) > (block.size // 2) else 0
            
    return cg_phenotype

def run_conditional_cg_simplicity_bias(rule=30, large_L=64, large_T=64, block_size=8, num_seeds=1000, shuffle=False, shuffle_after=False):
    """
    Checks for Simplicity Bias in Rule 30 specifically, conditioned on random seeds.
    Does coarse-grained patterns from a single rule still obey the complexity-probability bound?
    """
    engine = ElementaryCA(L=large_L, T=large_T)
    metric = ZlibComplexity()
    
    pattern_counts = Counter()
    pattern_to_complexity = {}
    
    target_L, target_T = large_L // block_size, large_T // block_size
    
    print(f"Running Rule {rule} Conditional Experiment...")
    print(f"Large Grid: {large_L}x{large_T} -> CG: {target_L}x{target_T}, Shuffled Before: {shuffle}, Shuffled After: {shuffle_after}")
    
    for _ in tqdm(range(num_seeds), desc=f"Seeds (Rule {rule})"):
        seed = engine.generate_seed(seed_type="random")
        phenotype = engine.run(rule, seed)
        
        # Coarse grain
        cg_phenotype = coarse_grain_phenotype(phenotype, block_size, shuffle_before=shuffle)

        if shuffle_after:
            cg_phenotype = shuffle_space_time(cg_phenotype)
        
        p_hash = cg_phenotype.tobytes()
        if p_hash not in pattern_to_complexity:
            pattern_to_complexity[p_hash] = metric.calculate(cg_phenotype)
        
        pattern_counts[p_hash] += 1
        
    # Data preparation
    unique_hashes = list(pattern_counts.keys())
    total_samples = sum(pattern_counts.values())
    
    Ks = []
    log_probs = []
    
    for h in unique_hashes:
        k = pattern_to_complexity[h]
        prob = pattern_counts[h] / total_samples
        Ks.append(k)
        log_probs.append(np.log10(prob))
        
    return np.array(Ks), np.array(log_probs), total_samples, target_L, target_T

def plot_conditional_sb(Ks, log_probs, total_samples, rule, large_L, large_T, cg_L, cg_T, block_size):
    plt.figure(figsize=(10, 7))
    plt.scatter(Ks, log_probs, alpha=0.6, c='crimson', s=25, label=f'Rule {rule} CG Patterns')
    
    # Correlations
    pearson_r, p_p = stats.pearsonr(Ks, log_probs)
    spearman_rho, p_s = stats.spearmanr(Ks, log_probs)
    
    # Upper bound
    unique_ks = np.unique(Ks)
    r_squared = 0
    if len(unique_ks) > 1:
        max_log_probs = np.array([np.max(log_probs[Ks == k]) for k in unique_ks])
        slope, intercept, r_val, p_val, std_err = stats.linregress(unique_ks, max_log_probs)
        x_line = np.linspace(min(Ks), max(Ks), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, color='black', linestyle='--')
        r_squared = r_val**2

    plt.xlabel("Complexity K (Zlib Bytes)")
    plt.ylabel("Log10 Probability P(x)")
    plt.title(f"Simplicity Bias: Coarse-Grained Rule {rule}\n"
              f"Grid {large_L}x{large_T} -> CG {cg_L}x{cg_T} (Block {block_size})")
    
    stats_text = (f"Pearson r: {pearson_r:.3f}\n"
                  f"Spearman ρ: {spearman_rho:.3f}\n"
                  f"Unique Patterns: {len(Ks)}\n"
                  f"Total Seeds: {total_samples}")
    
    plt.text(0.02, 0.05, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    save_dir = os.path.join("Analysis", "GeneralScripts", "Saved Figures")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"conditional_cg_sb_rule{rule}.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    RULE = 89
    LARGE_L = 68
    LARGE_T = 68
    BLOCK_SIZE = 17
    NUM_SEEDS = 1000000 # Higher seed count for better probability estimation for a single rule
    
    Ks, log_probs, total, cg_L, cg_T = run_conditional_cg_simplicity_bias(
        rule=RULE,
        large_L=LARGE_L,
        large_T=LARGE_T,
        block_size=BLOCK_SIZE,
        num_seeds=NUM_SEEDS,
        shuffle_after=False
    )
    
    plot_conditional_sb(Ks, log_probs, total, RULE, LARGE_L, LARGE_T, cg_L, cg_T, BLOCK_SIZE)
