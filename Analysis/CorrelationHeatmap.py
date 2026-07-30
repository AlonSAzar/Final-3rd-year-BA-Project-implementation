import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import sys
from scipy.stats import pearsonr, spearmanr

# Add the project root to sys.path to resolve imports properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.strategies import BitFlipSeedStrategy, BitFlipRuleStrategy
from Core.ComplexityMeasures.complexity import ZlibComplexity

def calculate_correlations(L, T, strategy, rules=None, num_seeds=1):
    """
    Calculates Pearson and Spearman correlations between parent and child complexity.
    Returns (pearson_val, spearman_val, strategy_name).
    """
    engine = ElementaryCA(L=L, T=T)
    metric = ZlibComplexity()
    
    if rules is None:
        rules = range(256)

    parent_complexities = []
    child_complexities = []
    
    for rule in rules:
        seeds = [engine.generate_seed() for _ in range(num_seeds)]
        
        for seed in seeds:
            # Parent Phenotype
            parent_img = engine.run(rule, seed)[1:]
            p_comp = metric.calculate(parent_img)
            
            # Mutation Sampling
            num_vars = strategy.get_variations_count(engine, rule, seed, 0)
            sample_size = min(num_vars, 5) # Sample mutations for speed
            mutation_indices = np.random.choice(range(num_vars), sample_size, replace=False)
            
            for idx in mutation_indices:
                m_rule, m_seed = strategy.apply(engine, rule, seed, idx)
                child_img = engine.run(m_rule, m_seed)[1:]
                c_comp = metric.calculate(child_img)
                
                parent_complexities.append(p_comp)
                child_complexities.append(c_comp)
    
    # Calculate correlations
    if len(parent_complexities) < 2:
        return 0.0, 0.0, strategy.name()
        
    p_corr, _ = pearsonr(parent_complexities, child_complexities)
    s_corr, _ = spearmanr(parent_complexities, child_complexities)
    
    # Handle NaNs from zero variation
    p_corr = np.nan_to_num(p_corr)
    s_corr = np.nan_to_num(s_corr)
    
    return p_corr, s_corr, strategy.name()

def main():
    L_values = [10, 50, 100, 250, 500]
    T_values = [10, 50, 100, 250, 500]
    
    strategies = [BitFlipSeedStrategy(), BitFlipRuleStrategy()]
    all_rules = range(256)
    num_seeds = 1 # One seed per rule to keep runtime manageable while covering all rules
    
    output_dir = "Saved Figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for strategy in strategies:
        print(f"\n--- Starting Experiment: {strategy.name()} ---")
        pearson_matrix = np.zeros((len(T_values), len(L_values)))
        spearman_matrix = np.zeros((len(T_values), len(L_values)))
        
        for i, T in enumerate(tqdm(T_values, desc=f"T loop ({strategy.name()})")):
            for j, L in enumerate(L_values):
                p_val, s_val, s_name = calculate_correlations(L, T, strategy, rules=all_rules, num_seeds=num_seeds)
                pearson_matrix[i, j] = p_val
                spearman_matrix[i, j] = s_val
                print(f"L={L}, T={T} -> Pearson: {p_val:.4f}, Spearman: {s_val:.4f}")

        # Plotting Pearson
        plt.figure(figsize=(10, 8))
        sns.heatmap(pearson_matrix, annot=True, fmt=".3f", 
                    xticklabels=L_values, yticklabels=T_values,
                    cmap="coolwarm", center=0)
        plt.title(f"Pearson Correlation (Parent vs Child Complexity) - {strategy.name()}")
        plt.xlabel("Grid Size (L)")
        plt.ylabel("Time Steps (T)")
        plt.savefig(os.path.join(output_dir, f"Pearson_Complexity_{strategy.name().replace(' ', '_')}.png"))
        
        # Plotting Spearman
        plt.figure(figsize=(10, 8))
        sns.heatmap(spearman_matrix, annot=True, fmt=".3f", 
                    xticklabels=L_values, yticklabels=T_values,
                    cmap="coolwarm", center=0)
        plt.title(f"Spearman Correlation (Parent vs Child Complexity) - {strategy.name()}")
        plt.xlabel("Grid Size (L)")
        plt.ylabel("Time Steps (T)")
        plt.savefig(os.path.join(output_dir, f"Spearman_Complexity_{strategy.name().replace(' ', '_')}.png"))
        
    print(f"\nAll correlation heatmaps saved to {output_dir}")
    plt.show()

if __name__ == "__main__":
    main()
