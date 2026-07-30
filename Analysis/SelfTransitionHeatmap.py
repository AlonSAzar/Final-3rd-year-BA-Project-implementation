import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import sys
import hashlib

# Add the project root to sys.path to resolve imports properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.strategies import *

def compute_hash(image: np.ndarray) -> str:
    """Computes a hash of the image for fast equality checking."""
    return hashlib.sha256(image.tobytes()).hexdigest()

def calculate_self_transition_chance(L, T, rules=None, num_seeds=1):
    """
    Calculates the average chance for a self-transition for a given L and T.
    A self-transition is an exact match (parent phenotype == mutant phenotype).
    """
    engine = ElementaryCA(L=L, T=T)
    strategy = BitFlipRuleStrategy()
    
    if rules is None:
        rules = range(256)

    total_matches = 0
    total_samples = 0
    
    for rule in rules:
        seeds = [engine.generate_seed() for _ in range(num_seeds)]
        
        for seed in seeds:
            # Parent Phenotype
            parent_img = engine.run(rule, seed)[1:]
            parent_hash = compute_hash(parent_img)
            
            # Mutation Sampling
            num_vars = strategy.get_variations_count(engine, rule, seed, 0)
            sample_size = min(num_vars, 10) # Sample 10 mutations for speed
            mutation_indices = np.random.choice(range(num_vars), sample_size, replace=False)
            
            for idx in mutation_indices:
                m_rule, m_seed = strategy.apply(engine, rule, seed, idx)
                mut_img = engine.run(m_rule, m_seed)[1:]
                mut_hash = compute_hash(mut_img)
                
                if parent_hash == mut_hash:
                    total_matches += 1
                total_samples += 1
                
    return total_matches / total_samples if total_samples > 0 else 0, strategy.name()

def main():
    # Same L and T intervals as requested
    L_values = [10, 50, 100, 250, 500]
    T_values = [10, 50, 100, 250, 500]
    
    # Matrix to store results
    chance_matrix = np.zeros((len(T_values), len(L_values)))
    strategy_name = ""
    
    print("Starting Grid Size vs Time Steps Self-Transition Experiment (All 256 Rules)...")
    
    # Iterate over all 256 rules
    all_rules = range(256)
    num_seeds = 1 # Consistent with previous scripts
    
    for i, T in enumerate(tqdm(T_values, desc="T loop")):
        for j, L in enumerate(L_values):
            chance, s_name = calculate_self_transition_chance(L, T, rules=all_rules, num_seeds=num_seeds)
            chance_matrix[i, j] = chance
            strategy_name = s_name
            print(f"L={L}, T={T} -> Self-Transition Chance: {chance:.4f}")

    # Plotting the Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(chance_matrix, annot=True, fmt=".4f", 
                xticklabels=L_values, yticklabels=T_values,
                cmap="plasma")
    
    plt.title(f"Self-Transition Probability Heatmap ({strategy_name}) - Grid Size vs Time Steps\n(Exact Phenotype Match)")
    plt.xlabel("Grid Size (L)")
    plt.ylabel("Time Steps (T)")
    
    # Save the figure
    output_dir = "Saved Figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(os.path.join(output_dir, "Self_Transition_Chance_L_vs_T_Heatmap.png"))
    print(f"Heatmap saved to {os.path.join(output_dir, 'Self_Transition_Chance_L_vs_T_Heatmap.png')}")
    plt.show()

if __name__ == "__main__":
    main()
