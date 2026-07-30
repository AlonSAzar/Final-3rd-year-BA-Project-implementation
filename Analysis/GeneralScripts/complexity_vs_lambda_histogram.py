import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

# Add the project root to sys.path to allow imports from Core and Experiments
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.ComplexityMeasures.complexity import ZlibComplexity

def get_lambda(rule_int):
    """
    Calculates Langton's Lambda for a 1D CA rule.
    For elementary CA (r=1, k=2), 8 bits define the rule.
    Lambda is the fraction of non-quiescent (usually 1) outputs in the LUT.
    """
    binary = [int(x) for x in bin(rule_int)[2:].zfill(8)]
    return sum(binary) / 8.0

def run_lambda_complexity_experiment(L=100, T=100, num_seeds=20):
    engine = ElementaryCA(L=L, T=T)
    metric = ZlibComplexity()
    
    # Dictionary to store complexities for each lambda value
    # For Elementary CA, lambda can only be 0/8, 1/8, ..., 8/8
    lambda_to_complexities = {}
    
    print(f"Running Complexity vs Lambda Experiment (L={L}, T={T}, seeds={num_seeds})...")
    
    for rule in tqdm(range(256), desc="Rules"):
        lam = get_lambda(rule)
        if lam not in lambda_to_complexities:
            lambda_to_complexities[lam] = []
            
        rule_complexities = []
        for _ in range(num_seeds):
            img = engine.run(rule)
            # Remove first row (seed) to match standard analysis
            complexity = metric.calculate(img[1:])
            rule_complexities.append(complexity)
            
        # Store the average complexity for this specific rule
        lambda_to_complexities[lam].append(np.mean(rule_complexities))
    
    # Calculate global average complexity per lambda value
    sorted_lambdas = sorted(lambda_to_complexities.keys())
    avg_complexities = [np.mean(lambda_to_complexities[l]) for l in sorted_lambdas]
    std_complexities = [np.std(lambda_to_complexities[l]) for l in sorted_lambdas]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Bar plot for the averages
    bars = plt.bar([str(l) for l in sorted_lambdas], avg_complexities, 
                   yerr=std_complexities, capsize=5, color='skyblue', edgecolor='navy', alpha=0.8)
    
    plt.xlabel("Langton's Lambda ($\lambda$)")
    plt.ylabel(f"Average Phenotype Complexity ({metric.name()})")
    plt.title(f"Average Complexity vs Langton's Lambda,\nwith Standard Deviation Bars\n1D Elementary CA (L={L}, T={T})\n")
    
    # Add data labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}', ha='center', va='bottom', fontsize=9)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_path = "Analysis/GeneralScripts/complexity_vs_lambda_histogram.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    run_lambda_complexity_experiment(L=64, T=64, num_seeds=50)
