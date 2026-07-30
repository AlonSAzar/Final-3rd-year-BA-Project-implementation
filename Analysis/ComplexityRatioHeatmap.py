import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import sys

# Add the project root to sys.path to resolve imports properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.ComplexityMeasures.complexity import ZlibComplexity
from Core.strategies import *

def calculate_complexity_ratio_stats(L, T, rules=None, num_seeds=3):
    """
    Calculates the Geometric Mean and Log-Standard Deviation of (parent complexity / child complexity).
    """
    engine = ElementaryCA(L=L, T=T)
    metric = ZlibComplexity()
    strategy = BitFlipSeedStrategy()
    
    if rules is None:
        rules = range(256)

    log_ratios = []
    
    for rule in rules:
        seeds = [engine.generate_seed() for _ in range(num_seeds)]
        
        for seed in seeds:
            # Parent Phenotype Complexity
            parent_img = engine.run(rule, seed)[1:]
            parent_comp = metric.calculate(parent_img)
            
            # Mutation Sampling
            num_vars = strategy.get_variations_count(engine, rule, seed, 0)
            sample_size = min(num_vars, 3) # Sample 10 mutations for speed
            mutation_indices = np.random.choice(range(num_vars), sample_size, replace=False)
            
            for idx in mutation_indices:
                m_rule, m_seed = strategy.apply(engine, rule, seed, idx)
                mut_img = engine.run(m_rule, m_seed)[1:]
                mut_comp = metric.calculate(mut_img)
                
                if mut_comp > 0 and parent_comp > 0:
                    log_ratios.append(np.log(parent_comp / mut_comp))
                
    if not log_ratios:
        return 1.0, 0, strategy.name()
        
    return np.exp(np.mean(log_ratios)), np.std(log_ratios), strategy.name()

def main():
    # Same L and T intervals as SelfTransitionHeatmap.py
    L_values = [10, 50, 100, 250, 500]
    T_values = [10, 50, 100, 250, 500]
    
    # Matrices to store results
    geo_mean_matrix = np.zeros((len(T_values), len(L_values)))
    log_std_matrix = np.zeros((len(T_values), len(L_values)))
    strategy_name = ""
    
    print("Starting Grid Size vs Time Steps Complexity Ratio Experiment (All 256 Rules)...")
    
    # Iterate over all 256 rules
    all_rules = range(256)
    num_seeds = 1 # Consistent with previous scripts
    
    for i, T in enumerate(tqdm(T_values, desc="T loop")):
        for j, L in enumerate(L_values):
            g_mean, l_std, s_name = calculate_complexity_ratio_stats(L, T, rules=all_rules, num_seeds=num_seeds)
            geo_mean_matrix[i, j] = g_mean
            log_std_matrix[i, j] = l_std
            strategy_name = s_name
            print(f"L={L}, T={T} -> Geo Mean: {g_mean:.4f}, Log-Std: {l_std:.4f}")

    output_dir = "Saved Figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plotting the Geometric Mean Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(geo_mean_matrix, annot=True, fmt=".4f", 
                xticklabels=L_values, yticklabels=T_values,
                cmap="viridis")
    plt.title(f"Geometric Mean of Parent/Child Complexity Ratio ({strategy_name})\nGrid Size vs Time Steps")
    plt.xlabel("Grid Size (L)")
    plt.ylabel("Time Steps (T)")
    avg_path = os.path.join(output_dir, "Complexity_Ratio_GeoMean_Heatmap.png")
    plt.savefig(avg_path)
    print(f"Geometric Mean Heatmap saved to {avg_path}")
    plt.close()

    # Plotting the Log-Standard Deviation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(log_std_matrix, annot=True, fmt=".4f", 
                xticklabels=L_values, yticklabels=T_values,
                cmap="magma")
    plt.title(f"Log-Standard Deviation of Complexity Ratio ({strategy_name})\nGrid Size vs Time Steps")
    plt.xlabel("Grid Size (L)")
    plt.ylabel("Time Steps (T)")
    std_path = os.path.join(output_dir, "Complexity_Ratio_LogStd_Heatmap.png")
    plt.savefig(std_path)
    print(f"Log-Std Heatmap saved to {std_path}")
    plt.close()

if __name__ == "__main__":
    main()