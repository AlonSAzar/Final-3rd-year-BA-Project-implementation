import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import sys
from scipy.stats import pearsonr, spearmanr
from collections import Counter
import warnings

# Add the project root to sys.path to resolve imports properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.ComplexityMeasures.complexity import ZlibComplexity

def calculate_phenotype_sb_correlations(L, T, rules=None, num_seeds=20):
    """
    Calculates Pearson and Spearman correlations between Complexity and Log-Frequency
    of FULL PHENOTYPES across given rules.
    """
    engine = ElementaryCA(L=L, T=T)
    metric = ZlibComplexity()
    
    if rules is None:
        rules = range(256)

    # Generate a fixed set of seeds to be used across all rules
    seeds = [engine.generate_seed() for _ in range(num_seeds)]

    # Collect all phenotypes across all rules and seeds
    all_phenotypes = []
    
    for rule in rules:
        for seed in seeds:
            # Run the rule and get full phenotype (T x L)
            phenotype = engine.run(rule, seed)[1:] # Ignore seed row
            all_phenotypes.append(phenotype.tobytes())
    
    # Count frequencies of unique phenotypes in the entire set
    counts = Counter(all_phenotypes)
    unique_hashes = list(counts.keys())
    
    # If everything is unique (K=1 for all), correlation is undefined
    if len(unique_hashes) <= 2:
        return 0.0, 0.0

    complexities = []
    log_freqs = []
    total_samples = len(all_phenotypes)
    
    for h in unique_hashes:
        # Reconstruct phenotype
        img_flat = np.frombuffer(h, dtype=np.uint8)
        # Dynamically infer dimensions based on buffer size
        # We know the total size is T * L
        if len(img_flat) != T * L:
            # Handle cases where the engine result might differ from expectations
            actual_T = len(img_flat) // L
            img = img_flat.reshape(actual_T, L)
        else:
            img = img_flat.reshape(T, L)
            
        comp = metric.calculate(img)
        prob = counts[h] / total_samples
        
        complexities.append(comp)
        log_freqs.append(np.log10(prob))
        
    # Check for variability
    if np.std(complexities) < 1e-9 or np.std(log_freqs) < 1e-9:
        return 0.0, 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_corr, _ = pearsonr(complexities, log_freqs)
        s_corr, _ = spearmanr(complexities, log_freqs)
    
    # Handle NaNs
    if np.isnan(p_corr): p_corr = 0.0
    if np.isnan(s_corr): s_corr = 0.0
    
    return p_corr, s_corr

def main():
    # L, T values to iterate over for the heatmap
    L_values = [8, 16, 32, 64]
    T_values = [8, 16, 32, 64]
    
    # number of seeds per rule to explore the phenotype space
    num_seeds_per_rule = 10 
    
    print(f"Generating Phenotype Simplicity Bias Correlation Heatmaps...")
    print(f"Iterating over Rules 0-255 with {num_seeds_per_rule} seeds each.")

    pearson_matrix = np.zeros((len(T_values), len(L_values)))
    spearman_matrix = np.zeros((len(T_values), len(L_values)))

    for i, T in enumerate(tqdm(T_values, desc="T loop")):
        for j, L in enumerate(L_values):
            p_val, s_val = calculate_phenotype_sb_correlations(L, T, num_seeds=num_seeds_per_rule)
            pearson_matrix[i, j] = p_val
            spearman_matrix[i, j] = s_val

    output_dir = "Analysis/Saved Figures"
    os.makedirs(output_dir, exist_ok=True)

    # Plot Pearson Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(pearson_matrix, annot=True, fmt=".3f", 
                xticklabels=L_values, yticklabels=T_values,
                cmap="coolwarm", center=0)
    plt.title(f"Pearson Corr: K(Phenotype) vs log10(P)\n(Full Phenotypes across all ECA Rules)")
    plt.xlabel("Grid Width (L)")
    plt.ylabel("Time steps (T)")
    plt.savefig(os.path.join(output_dir, "Phenotype_SB_Pearson_Heatmap.png"))
    
    # Plot Spearman Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(spearman_matrix, annot=True, fmt=".3f", 
                xticklabels=L_values, yticklabels=T_values,
                cmap="coolwarm", center=0)
    plt.title(f"Spearman Corr: K(Phenotype) vs log10(P)\n(Full Phenotypes across all ECA Rules)")
    plt.xlabel("Grid Width (L)")
    plt.ylabel("Time steps (T)")
    plt.savefig(os.path.join(output_dir, "Phenotype_SB_Spearman_Heatmap.png"))
    
    print(f"\nPhenotype SB Heatmaps saved to {output_dir}")
    plt.show()

if __name__ == "__main__":
    main()
