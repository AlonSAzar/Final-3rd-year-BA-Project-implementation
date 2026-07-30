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

def calculate_complexity(L, T, rules=[30], num_seeds=100):
    """
    Calculates the average complexity score for a given L and T.
    Complexity is measured by Zlib compression vector size.
    """
    if rules is None:
        rules = [30]
        
    engine = ElementaryCA(L=L, T=T)
    metric = ZlibComplexity()

    total_complexity = 0
    
    for rule in rules:
        rule_score = 0
        seeds = [engine.generate_seed() for _ in range(num_seeds)]
        
        for seed in seeds:
            img = engine.run(rule, seed)[1:] # Exclude initial seed for consistency
            rule_score += metric.calculate(img)
        
        total_complexity += (rule_score / num_seeds)
    
    return total_complexity / len(rules), metric.name()

def main():
    # Grid Sizes (L) and Time Steps (T)
    L_values = [10, 50, 100, 250, 500]
    T_values = [10, 50, 100, 250, 500]
    
    # Matrix to store results
    robustness_matrix = np.zeros((len(T_values), len(L_values)))
    complexity_per_pixel_matrix = np.zeros((len(T_values), len(L_values)))
    metric_name = ""
    
    # Define rules here or set to None to use default in calculate_complexity
    all_rules = [30]
    
    print(f"Starting Grid Size vs Time Steps Complexity Experiment (Rules: {all_rules if all_rules is not None else 'Default'})...")
    
    # Sample multiple random seeds per rule for better statistical coverage
    num_seeds = 10 
    
    for i, T in enumerate(tqdm(T_values, desc="T loop")):
        for j, L in enumerate(L_values):
            # Use default rules if all_rules is None
            score, m_name = calculate_complexity(L, T, rules=all_rules, num_seeds=num_seeds)
            robustness_matrix[i, j] = score
            metric_name = m_name
            
            # Calculate complexity per pixel (excluding initial seed: T-1 steps)
            num_pixels = (T - 1) * L
            complexity_per_pixel_matrix[i, j] = score / num_pixels if num_pixels > 0 else 0
            
            print(f"L={L}, T={T} -> Complexity: {score:.2f}, Per Pixel: {complexity_per_pixel_matrix[i, j]:.4f}")

    # --- Plotting Total Complexity Heatmap ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(robustness_matrix, annot=True, fmt=".0f", 
                xticklabels=L_values, yticklabels=T_values,
                cmap="rocket")
    
    plt.title(f"Complexity Heatmap ({metric_name}) - Grid Size vs Time Steps")
    plt.xlabel("Grid Size (L)")
    plt.ylabel("Time Steps (T)")
    
    output_dir = "Saved Figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(os.path.join(output_dir, "Complexity_L_vs_T_Heatmap.png"))
    print(f"Heatmap saved to {os.path.join(output_dir, 'Complexity_L_vs_T_Heatmap.png')}")

    # --- Plotting Complexity Per Pixel Heatmap ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(complexity_per_pixel_matrix, annot=True, fmt=".4f", 
                xticklabels=L_values, yticklabels=T_values,
                cmap="mako")
    
    plt.title(f"Complexity Per Pixel Heatmap ({metric_name}/L*T) - Grid Size vs Time Steps")
    plt.xlabel("Grid Size (L)")
    plt.ylabel("Time Steps (T)")
    
    plt.savefig(os.path.join(output_dir, "Complexity_Per_Pixel_L_vs_T_Heatmap.png"))
    print(f"Heatmap saved to {os.path.join(output_dir, 'Complexity_Per_Pixel_L_vs_T_Heatmap.png')}")
    
    plt.show()

if __name__ == "__main__":
    main()
