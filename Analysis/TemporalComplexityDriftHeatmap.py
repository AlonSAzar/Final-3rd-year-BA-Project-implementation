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
from Core.strategies import BitFlipRuleStrategy

def calculate_temporal_complexity_ratio(L, T, rules=[30], num_seeds=100):
    """
    Calculates the Geometric Mean of the ratio (Complexity of 2nd third / Complexity of 3rd third).
    Thirds are divided along the time dimension (rows of the lattice).
    """
    engine = ElementaryCA(L=L, T=T)
    metric = ZlibComplexity()
    strategy = BitFlipRuleStrategy()
    
    if rules is None:
        rules = range(256)

    log_ratios = []
    
    # Calculate size of each third (discarding remainder)
    # T represents total time steps (excluding seed row 0)
    slice_size = T // 3
    
    # Second Third: rows [slice_size + 1 : 2 * slice_size + 1]
    # Third Third: rows [2 * slice_size + 1 : 3 * slice_size + 1]
    # This ensures both slices are exactly 'slice_size' rows long.
    
    for rule in rules:
        seeds = [engine.generate_seed() for _ in range(num_seeds)]
        
        for seed in seeds:
            # Full History (T+1, L)
            full_history = engine.run(rule, seed)
            
            # Extract equal-sized thirds
            second_third = full_history[slice_size + 1 : 2 * slice_size + 1]
            third_third = full_history[2 * slice_size + 1 : 3 * slice_size + 1]
            
            if second_third.size == 0 or third_third.size == 0:
                continue
                
            comp2 = metric.calculate(second_third)
            comp3 = metric.calculate(third_third)
            
            if comp2 > 0 and comp3 > 0:
                log_ratios.append(np.log(comp2 / comp3))
                
    if not log_ratios:
        return 1.0, strategy.name()
        
    return np.exp(np.mean(log_ratios)), strategy.name()

def main():
    # Same L and T intervals
    L_values = [10, 50, 100, 250, 500]
    T_values = [10, 50, 100, 250, 500]
    
    # Matrix to store results
    ratio_matrix = np.zeros((len(T_values), len(L_values)))
    strategy_name = ""
    
    print("Starting Temporal Complexity Drift Experiment (Ratio of 2nd Third / 3rd Third)...")
    
    all_rules = range(256)
    num_seeds = 1
    
    for i, T in enumerate(tqdm(T_values, desc="T loop")):
        for j, L in enumerate(L_values):
            # For small T, thirds might be empty or invalid. Handle gracefully.
            if T < 3:
                ratio_matrix[i, j] = 1.0
                continue
                
            ratio, s_name = calculate_temporal_complexity_ratio(L, T)
            ratio_matrix[i, j] = ratio
            strategy_name = s_name
            print(f"L={L}, T={T} -> Geo Mean Ratio (2nd/3rd): {ratio:.4f}")

    # Plotting the Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(ratio_matrix, annot=True, fmt=".4f", 
                xticklabels=L_values, yticklabels=T_values,
                cmap="coolwarm", center=1.0) # Center at 1.0 for drift visualization
    
    plt.title(f"Temporal Complexity Drift\nGeo Mean Ratio: K(2nd Third) / K(3rd Third)")
    plt.xlabel("Grid Size (L)")
    plt.ylabel("Time Steps (T)")
    
    output_dir = "Saved Figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_path = os.path.join(output_dir, "Temporal_Complexity_Drift_Heatmap.png")
    plt.savefig(save_path)
    print(f"Heatmap saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()