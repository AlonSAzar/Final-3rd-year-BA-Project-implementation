import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
from collections import Counter

import itertools

# Add the project root to sys.path to resolve imports properly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.ComplexityMeasures.complexity import ZlibComplexity

def generate_simplicity_bias_csv(L=8, T=16, output_path="simplicity_bias_phenotypes.csv"):
    """
    Generates a CSV file where each row is a unique phenotype.
    Iterates through ALL possible seeds of length L for every rule.
    """
    engine = ElementaryCA(L=L, T=T)
    metric = ZlibComplexity()
    
    # phenotype_hash -> {rule: count}
    phenotype_rule_distribution = {}
    # phenotype_hash -> first_found_phenotype_array
    phenotype_lookup = {}
    
    # Generate all possible binary seeds of length L
    all_seeds = [np.array(s) for s in itertools.product([0, 1], repeat=L)]
    total_seeds = len(all_seeds)
    
    total_samples = 0
    rules = range(256)
    
    print(f"Starting Exhaustive Simplicity Bias Experiment...")
    print(f"Parameters: L={L}, T={T}, Total Seeds per rule={total_seeds}")
    
    for rule in tqdm(rules, desc="Simulating Rules"):
        for seed in all_seeds:
            # Run CA (returns history of size T+1, L)
            # Remove the initial seed row [1:] as is standard for simplicity bias 
            # to ensure we only measure the rule's output patterns.
            phenotype = engine.run(rule, seed)[1:]
            
            # Use hex hash of bytes as unique identifier
            p_hash = phenotype.tobytes().hex()
            
            if p_hash not in phenotype_rule_distribution:
                phenotype_rule_distribution[p_hash] = Counter()
                phenotype_lookup[p_hash] = phenotype
            
            phenotype_rule_distribution[p_hash][rule] += 1
            total_samples += 1
            
    # Prepare data for CSV
    data = []
    for p_hash, rule_counts in tqdm(phenotype_rule_distribution.items(), desc="Calculating Complexities"):
        total_freq = sum(rule_counts.values())
        complexity = metric.calculate(phenotype_lookup[p_hash])
        
        entry = {
            "Phenotype_ID": p_hash, 
            "Complexity": complexity,
            "Frequency": total_freq,
            "Probability": total_freq / total_samples,
            "Num_Unique_Rules": len(rule_counts)
        }
        
        # Add a column for every single rule (0-255) as requested
        for r in range(256):
            entry[f"Rule_{r}"] = rule_counts.get(r, 0)
            
        data.append(entry)
        
    print(f"Creating DataFrame and saving to {output_path}...")
    df = pd.DataFrame(data)
    
    # Verification: Ensure Phenotype_ID is unique
    if df['Phenotype_ID'].duplicated().any():
        print("Warning: Duplicate Phenotype_IDs found! This shouldn't happen with full hashes.")
        df = df.drop_duplicates(subset=['Phenotype_ID'])
    
    # Sort by Frequency (Simplicity Bias usually shows high frequency for low complexity)
    df = df.sort_values(by="Frequency", ascending=False)
    
    df.to_csv(output_path, index=False)
    print(f"Done! Saved {len(df)} unique phenotypes to {output_path}")

if __name__ == "__main__":
    # Seed length L=8 means 2^8 = 256 seeds per rule.
    # Total samples = 256 * 256 = 65,536. 
    # This is manageable for a standard CSV and runs quickly.
    generate_simplicity_bias_csv(
        L=8,
        T=64,
        output_path="simplicity_bias_results_exhaustive.csv"
    )
