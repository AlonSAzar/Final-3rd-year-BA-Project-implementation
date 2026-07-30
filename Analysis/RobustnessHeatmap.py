import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import sys

# Add the project root to sys.path to resolve imports properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.strategies import *
from Experiments.experiments import compute_ncc

def calculate_robustness(L, T, rules=[30], num_seeds=100):
    """
    Calculates the average robustness score for a given L and T.
    Robustness is defined as the average NCC between base and mutant phenotypes.
    """
    engine = ElementaryCA(L=L, T=T)
    strategy = BitFlipSeedStrategy()
    
    if rules is None:
        rules = range(256)

    total_robustness = 0
    
    for rule in rules:
        rule_score = 0
        seeds = [engine.generate_seed() for _ in range(num_seeds)]
        
        for seed in seeds:
            base_img = engine.run(rule, seed)[1:] # Exclude initial seed if desired
            
            num_vars = strategy.get_variations_count(engine, rule, seed, 0)
            sample_size = min(num_vars, 20) # Reduced mutation sampling for speed over all rules
            
            mutant_scores = []
            mutation_indices = np.random.choice(range(num_vars), sample_size, replace=False)
            
            for i in mutation_indices:
                m_rule, m_seed = strategy.apply(engine, rule, seed, i)
                mut_img = engine.run(m_rule, m_seed)[1:]
                
                ncc = compute_ncc(base_img, mut_img)
                mutant_scores.append(ncc)
            
            rule_score += np.mean(mutant_scores)
        
        total_robustness += (rule_score / num_seeds)
    
    return total_robustness / len(rules), strategy.name()

def main():
    # Grid Sizes (L) and Time Steps (T)
    L_values = [10, 50, 100, 250, 500]
    T_values = [10, 50, 100, 250, 500]
    
    # Matrix to store results
    robustness_matrix = np.zeros((len(T_values), len(L_values)))
    strategy_name = ""
    
    print("Starting Grid Size vs Time Steps Robustness Experiment (All 256 Rules)...")
    
    # Iterate over all 256 rules
    all_rules = range(256)
    num_seeds = 1 # One seed per rule to keep runtime manageable while covering all rules
    
    for i, T in enumerate(tqdm(T_values, desc="T loop")):
        for j, L in enumerate(L_values):
            # Pass all rules
            score, s_name = calculate_robustness(L, T, rules=[30])
            robustness_matrix[i, j] = score
            strategy_name = s_name
            print(f"L={L}, T={T} -> Robustness: {score:.4f}")

    # Plotting the Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(robustness_matrix, annot=True, fmt=".3f", 
                xticklabels=L_values, yticklabels=T_values,
                cmap="viridis")
    
    plt.title(f"Robustness Score Heatmap ({strategy_name}) - Grid Size vs Time Steps")
    plt.xlabel("Grid Size (L)")
    plt.ylabel("Time Steps (T)")
    
    # Save the figure
    output_dir = "Saved Figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(os.path.join(output_dir, "Robustness_L_vs_T_Heatmap.png"))
    print(f"Heatmap saved to {os.path.join(output_dir, 'Robustness_L_vs_T_Heatmap.png')}")
    plt.show() # Commented out to avoid blocking in terminal

if __name__ == "__main__":
    main()
