import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from collections import Counter
import scipy.stats as stats

# Add the project root to sys.path to resolve imports properly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.ComplexityMeasures.complexity import ZlibComplexity

def run_global_simplicity_bias_plot(L=128, T=4096, piece_height=16, num_samples_per_rule=1):
    """
    Collects sub-patterns from all rules and plots their complexity vs log-probability
    in a unified Simplicity Bias plot (similar to phenotypic distribution plots).
    """
    engine = ElementaryCA(L=L, T=T)
    metric = ZlibComplexity()
    
    # Global counter for all unique sub-patterns found across ALL rules
    global_pattern_counts = Counter()
    global_pattern_to_complexity = {}
    
    print(f"Collecting sub-patterns from 256 rules...")
    print(f"Parameters: L={L}, T={T}, Piece Height={piece_height}")
    
    for rule in tqdm(range(256), desc="Rules"):
        for _ in range(num_samples_per_rule):
            seed = engine.generate_seed(seed_type="random")
            # Run CA (ignore first row/seed)
            full_history = engine.run(rule, seed)[1:] 
            
            # Slice into pieces
            num_pieces = T // piece_height
            for i in range(num_pieces):
                piece = full_history[i * piece_height : (i + 1) * piece_height]
                p_hash = piece.tobytes()
                
                # Check if we've already calculated complexity for this unique pattern
                if p_hash not in global_pattern_to_complexity:
                    global_pattern_to_complexity[p_hash] = metric.calculate(piece)
                
                global_pattern_counts[p_hash] += 1
                
    # Prepare data for plotting
    unique_hashes = list(global_pattern_counts.keys())
    total_samples = sum(global_pattern_counts.values())
    
    complexities = []
    log_probs = []
    
    for h in unique_hashes:
        k = global_pattern_to_complexity[h]
        prob = global_pattern_counts[h] / total_samples
        
        complexities.append(k)
        log_probs.append(np.log10(prob))
        
    return np.array(complexities), np.array(log_probs), total_samples

def plot_simplicity_bias(Ks, log_probs, total_samples, title_suffix=""):
    """
    Generates a Simplicity Bias plot with an upper-bound fit.
    """
    plt.figure(figsize=(12, 8))
    
    # Scale probabilities for better visibility
    plt.scatter(Ks, log_probs, alpha=0.3, c='teal', s=10, label='Sub-patterns (all rules)')
    
    # ---------------- UPPER BOUND FITTING ----------------
    # Find max probability for each unique complexity value
    unique_ks = np.unique(Ks)
    max_log_probs = []
    for k in unique_ks:
        max_val = np.max(log_probs[Ks == k])
        max_log_probs.append(max_val)

    unique_ks = np.array(unique_ks)
    max_log_probs = np.array(max_log_probs)

    # Fit linear regression to the upper bound (log10(P) = mK + c)
    if len(unique_ks) > 1:
        slope, intercept, r_val, p_val, std_err = stats.linregress(unique_ks, max_log_probs)
        x_line = np.linspace(min(Ks), max(Ks), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, color='red', linestyle='--', linewidth=2, label=f'Upper Bound Fit (R²={r_val**2:.3f})')
        
        # Convert to P = 2^(-aK - b) format for stats info
        log2_10 = np.log2(10)
        a_param = -slope * log2_10
        b_param = -intercept * log2_10
    else:
        a_param = b_param = 0

    plt.xlabel("Complexity K (Zlib Bytes)")
    plt.ylabel("Log10 Probability")
    plt.title(f"Global Simplicity Bias Test: Sub-patterns from all 256 CA rules\n(L=128, pieces=16x128) {title_suffix}")
    
    # Correlations for all data points
    if len(Ks) > 1:
        spearman_corr, _ = stats.spearmanr(Ks, log_probs)
        pearson_corr, _ = stats.pearsonr(Ks, log_probs)
    else:
        spearman_corr = pearson_corr = 0

    # Stats box
    stats_text = (
        f"Total Unique Patterns: {len(Ks)}\n"
        f"Total Samples: {total_samples}\n\n"
        f"Correlations (All data):\n"
        f"  Spearman: {spearman_corr:.3f}\n"
        f"  Pearson: {pearson_corr:.3f}\n\n"
        f"Upper Bound Fit (P = 2^(-aK-b)):\n"
        f"  a = {a_param:.3f}\n"
        f"  b = {b_param:.3f}"
    )
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.grid(True, alpha=0.2)
    plt.legend(loc='lower left')
    
    # Save figure
    save_dir = os.path.join("Analysis", "GeneralScripts", "Saved Figures")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "global_simplicity_bias_plot.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # Parameters matches existing GlobalTest but unified
    L_VAL = 64 # Shorter L for more unique pattern overlaps
    T_VAL = 1024
    HEIGHT = 16
    SAMPLES = 10 # Increase for better statistics
    
    Ks, log_ps, total = run_global_simplicity_bias_plot(L=L_VAL, T=T_VAL, piece_height=HEIGHT, num_samples_per_rule=SAMPLES)
    plot_simplicity_bias(Ks, log_ps, total)
