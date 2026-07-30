import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import sys
import os

# Add the project root to sys.path to allow imports from Core and Experiments
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.ComplexityMeasures.complexity import ZlibComplexity
from Core.ComplexityMeasures.conditional_complexity import ZlibConditionalComplexity

def get_avg_complexity(engine, metric, rule, num_seeds=20):
    """
    Calculates Average Phenotype Complexity (K).
    """
    ks = []
    for _ in range(num_seeds):
        img = engine.run(rule)[1:] # Skip first row
        ks.append(metric.calculate(img))
        
    return np.mean(ks)

def analyze_simplicity_bias_correlations(L=64, T=64, num_seeds=100):
    """
    Iterates through rules 0-255:
    1. Calculates Avg Complexity (K).
    2. Calculates Correlation for Simplicity Bias: Log(P(x)) vs K(x).
    3. Calculates Correlation for Conditional Simplicity Bias: Log(P(y|x)) vs K(y|x) under seed bit flip.
    """
    engine = ElementaryCA(L=L, T=T)
    metric = ZlibComplexity()
    cond_metric = ZlibConditionalComplexity()
    
    results = {
        'rule': [],
        'avg_complexity': [],
        'sb_correlation': [],
        'conditional_sb_correlation': [],
        'lambda_dist': []
    }
    
    print(f"Running Complexity vs Simplicity Bias Correlations (L={L}, T={T})...")
    
    for rule in tqdm(range(256), desc="Analyzing Rules"):
        # 1. Average Complexity
        avg_k = get_avg_complexity(engine, metric, rule, num_seeds=30)

        # --- Simplicity Bias (Global) ---
        phenotype_counts = {}
        phenotype_complexities = {}

        for _ in range(num_seeds):
            img = engine.run(rule)[1:]
            h = img.tobytes()

            if h not in phenotype_counts:
                phenotype_counts[h] = 0
                phenotype_complexities[h] = metric.calculate(img)
            phenotype_counts[h] += 1

        # Get unique phenotypes, their frequencies, and their K
        unique_probs = np.array(list(phenotype_counts.values())) / num_seeds
        unique_ks = np.array(list(phenotype_complexities.values()))

        if len(unique_probs) > 1 and np.var(unique_ks) > 0:
            sb_corr, _ = spearmanr(unique_ks, np.log(unique_probs))
        else:
            sb_corr = 0

        # --- Conditional Simplicity Bias (Mutation) ---
        transition_counts = {}
        transition_complexities = {}

        for _ in range(num_seeds):
            seed = engine.generate_seed()
            img_x = engine.run(rule, seed)[1:]

            # Mutate
            mut_seed = seed.copy()
            mut_seed[np.random.randint(len(seed))] ^= 1
            img_y = engine.run(rule, mut_seed)[1:]

            # Use tuple of hashes as the transition key
            h_transition = (img_x.tobytes(), img_y.tobytes())

            if h_transition not in transition_counts:
                transition_counts[h_transition] = 0
                transition_complexities[h_transition] = cond_metric.calculate(img_y, img_x)
            transition_counts[h_transition] += 1

        cond_probs = np.array(list(transition_counts.values())) / num_seeds
        cond_ks = np.array(list(transition_complexities.values()))

        if len(cond_probs) > 1 and np.var(cond_ks) > 0:
            cond_sb_corr, _ = spearmanr(cond_ks, np.log(cond_probs))
        else:
            cond_sb_corr = 0

        results['rule'].append(rule)
        results['avg_complexity'].append(avg_k)
        results['sb_correlation'].append(sb_corr)
        results['conditional_sb_correlation'].append(cond_sb_corr)
        
        lam = bin(rule).count('1') / 8.0
        results['lambda_dist'].append(abs(lam - 0.5))

    # --- Plotting ---
    
    # 1. Global Simplicity Bias Plot
    plt.figure(figsize=(10, 7))
    sc = plt.scatter(results['avg_complexity'], results['sb_correlation'], 
                 alpha=0.8, c=results['lambda_dist'], cmap='viridis', edgecolors='black')
    cbar = plt.colorbar(sc)
    cbar.set_label('Distance to λ=0.5')
    
    plt.xlabel("Average Phenotype Complexity ($K$)")
    plt.ylabel("Global Simplicity Bias Correlation (Spearman ρ)\nLog P(x) vs K(x)")
    plt.title(f"Average Complexity vs Global Simplicity Bias (L={L}, T={T})")
    plt.grid(True, alpha=0.3)

    mask2 = np.array(results['sb_correlation']) != 0
    if np.any(mask2):
        x2 = np.array(results['avg_complexity'])[mask2]
        y2 = np.array(results['sb_correlation'])[mask2]
        p2, _ = pearsonr(x2, y2)
        s2, _ = spearmanr(x2, y2)
        stats_text2 = f"Pearson ρ: {p2:.3f}\nSpearman ρ: {s2:.3f}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        plt.gca().text(0.05, 0.95, stats_text2, transform=plt.gca().transAxes, fontsize=10,
                      verticalalignment='top', horizontalalignment='left', bbox=props)
    
    plt.tight_layout()
    plt.savefig("Analysis/GeneralScripts/complexity_vs_global_sb_correlation.png")
    plt.show()

    # 2. Conditional Simplicity Bias Plot
    plt.figure(figsize=(10, 7))
    sc2 = plt.scatter(results['avg_complexity'], results['conditional_sb_correlation'], 
                 alpha=0.8, c=results['lambda_dist'], cmap='plasma', edgecolors='black')
    cbar2 = plt.colorbar(sc2)
    cbar2.set_label('Distance to λ=0.5')
    
    plt.xlabel("Average Phenotype Complexity ($K$)")
    plt.ylabel("Cond. Simplicity Bias Correlation (Spearman ρ)\nLog P(y|x) vs K(y|x)")
    plt.title(f"Average Complexity vs Conditional Simplicity Bias, Seed Bit-Flip (L={L}, T={T})")
    plt.grid(True, alpha=0.3)
    
    mask1 = np.array(results['conditional_sb_correlation']) != 0
    if np.any(mask1):
        x1 = np.array(results['avg_complexity'])[mask1]
        y1 = np.array(results['conditional_sb_correlation'])[mask1]
        p1, _ = pearsonr(x1, y1)
        s1, _ = spearmanr(x1, y1)
        stats_text1 = f"Pearson ρ: {p1:.3f}\nSpearman ρ: {s1:.3f}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        plt.gca().text(0.05, 0.95, stats_text1, transform=plt.gca().transAxes, fontsize=10,
                      verticalalignment='top', horizontalalignment='left', bbox=props)
    
    plt.tight_layout()
    plt.savefig("Analysis/GeneralScripts/complexity_vs_mutational_sb_correlation.png")
    plt.show()

    print(f"Analysis complete. Plots saved to:\n  Analysis/GeneralScripts/complexity_vs_global_sb_correlation.png\n  Analysis/GeneralScripts/complexity_vs_mutational_sb_correlation.png")
    
    plt.suptitle(f"Simplicity Bias Correlation vs Average Phenotype Complexity (L={L}, T={T})")
    plt.tight_layout()
    
    output_path = "Analysis/GeneralScripts/complexity_vs_simplicity_correlation.png"
    plt.savefig(output_path)
    print(f"Analysis complete. Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    analyze_simplicity_bias_correlations(L=8, T=8, num_seeds=500)
