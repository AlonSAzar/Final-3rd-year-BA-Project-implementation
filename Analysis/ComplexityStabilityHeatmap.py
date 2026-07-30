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

def calculate_complexity_stability_percentage(L, T, rules=None, num_seeds=1, threshold=0.10):
    """
    Calculates the percentage of mutations that keep complexity within ±10% of the parent.
    """
    engine = ElementaryCA(L=L, T=T)
    metric = ZlibComplexity()
    strategy = BitFlipSeedStrategy()
    
    if rules is None:
        rules = range(256)

    total_mutations = 0
    stable_mutations = 0
    
    for rule in rules:
        seeds = [engine.generate_seed() for _ in range(num_seeds)]
        
        for seed in seeds:
            # Parent Phenotype Complexity
            parent_img = engine.run(rule, seed)[1:]
            parent_comp = metric.calculate(parent_img)
            
            # Mutation Sampling
            num_vars = strategy.get_variations_count(engine, rule, seed, 0)
            sample_size = min(num_vars, 10) # Sample 10 mutations for speed
            mutation_indices = np.random.choice(range(num_vars), sample_size, replace=False)
            
            for idx in mutation_indices:
                m_rule, m_seed = strategy.apply(engine, rule, seed, idx)
                mut_img = engine.run(m_rule, m_seed)[1:]
                mut_comp = metric.calculate(mut_img)
                
                if parent_comp > 0:
                    # Calculate relative difference
                    relative_diff = abs(mut_comp - parent_comp) / parent_comp
                    if relative_diff <= threshold:
                        stable_mutations += 1
                elif mut_comp == 0:
                    # Both are 0, technically "stable"
                    stable_mutations += 1
                
                total_mutations += 1
                
    if total_mutations == 0:
        return 0.0, strategy.name()
        
    return (stable_mutations / total_mutations) * 100, strategy.name()

def main():
    # Same L and T intervals as requested
    L_values = [10, 50, 100, 250, 500]
    T_values = [10, 50, 100, 250, 500]
    
    # Matrix to store results
    stability_matrix = np.zeros((len(T_values), len(L_values)))
    strategy_name = ""
    
    print("Starting Grid Size vs Time Steps Complexity Stability Experiment (All 256 Rules)...")
    
    # Iterate over all 256 rules
    all_rules = range(256)
    num_seeds = 1 # Consistent with previous scripts
    
    for i, T in enumerate(tqdm(T_values, desc="T loop")):
        for j, L in enumerate(L_values):
            percentage, s_name = calculate_complexity_stability_percentage(L, T, rules=all_rules, num_seeds=num_seeds)
            stability_matrix[i, j] = percentage
            strategy_name = s_name
            print(f"L={L}, T={T} -> Stable Mutations (±10%): {percentage:.2f}%")

    # Plotting the Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(stability_matrix, annot=True, fmt=".2f", 
                xticklabels=L_values, yticklabels=T_values,
                cmap="YlGnBu")
    
    plt.title(f"Complexity Stability Heatmap ({strategy_name})\nPercentage of Mutations within ±10% of Parent Complexity")
    plt.xlabel("Grid Size (L)")
    plt.ylabel("Time Steps (T)")
    
    # Save the figure
    output_dir = "Saved Figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_path = os.path.join(output_dir, "Complexity_Stability_Heatmap.png")
    plt.savefig(save_path)
    print(f"Heatmap saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()