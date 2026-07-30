import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.ComplexityMeasures.complexity import ZlibComplexity

def run_accessible_states_experiment(L=32, T=32, samples_per_rule=500):
    """
    Measures how the number of UNIQUE accessible states grows with Complexity (K).
    """
    engine = ElementaryCA(L=L, T=T)
    metric = ZlibComplexity()
    
    # Store unique phenotypes. Key: K, Value: Set of hashable phenotypes
    complexity_to_unique_states = defaultdict(set)
    
    print(f"Sampling {samples_per_rule} seeds for all 256 rules...")
    for rule in tqdm(range(256), desc="Rules"):
        for _ in range(samples_per_rule):
            seed = engine.generate_seed(seed_type="random")
            img = engine.run(rule, seed)
            
            # Create a hashable version of the 2D array to check for uniqueness
            img_hash = img.tobytes()
            
            k = metric.calculate(img)
            complexity_to_unique_states[k].add(img_hash)
            
    # Count unique states per K
    unique_ks = sorted(list(complexity_to_unique_states.keys()))
    unique_counts = [len(complexity_to_unique_states[k]) for k in unique_ks]
    
    return np.array(unique_ks), np.array(unique_counts)

def plot_growth(Ks, counts):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Semi-Log Plot (To check for exponential growth e^(beta * K))
    axes[0].plot(Ks, counts, marker='o', color='teal', linestyle='-', alpha=0.7)
    axes[0].set_yscale('log')
    axes[0].set_title(f"Number of Unique States vs Complexity (Semi-Log)\nDoes it grow exponentially?")
    axes[0].set_xlabel("Complexity K")
    axes[0].set_ylabel("Count of Unique Phenotypes (Log)")
    axes[0].grid(True, which="both", alpha=0.2)

    # 2. Log-Log Plot (To check for polynomial/power-law volume growth)
    axes[1].plot(Ks, counts, marker='s', color='purple', linestyle='-', alpha=0.7)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_title(f"Number of Unique States vs Complexity (Log-Log)\nDoes it grow polynomially?")
    axes[1].set_xlabel("Complexity K (Log)")
    axes[1].set_ylabel("Count of Unique Phenotypes (Log)")
    axes[1].grid(True, which="both", alpha=0.2)
    
    # Save the figure
    save_dir = os.path.join("Analysis", "GeneralScripts", "Saved Figures")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename = "accessible_states_growth.png"
    save_path = os.path.join(save_dir, filename)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # We use a slightly smaller L and T so we can heavily undersample the space and find overlaps
    Ks, counts = run_accessible_states_experiment(L=128, T=128, samples_per_rule=50)
    plot_growth(Ks, counts)
