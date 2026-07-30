import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.ComplexityMeasures.conditional_complexity import ZlibConditionalComplexity
from Core.strategies import BitFlipSeedStrategy

def run_lambda_distance_experiments_combined(L=128, T=128, num_samples_per_rule=100):
    """
    Groups rules by their absolute distance from Langton's Lambda = 0.5.
    Runs the seed bit-flip conditional complexity experiment for each group,
    and saves all log-log plots into a single combined figure.
    """
    # 1. Group rules by |Lambda - 0.5|
    dist_to_rules = {}
    for rule in range(256):
        # Lambda is the fraction of 1s in the 8-bit binary representation of the rule
        lam = bin(rule).count('1') / 8.0
        dist = round(abs(lam - 0.5), 3)
        if dist not in dist_to_rules:
            dist_to_rules[dist] = []
        dist_to_rules[dist].append(rule)
        
    engine = ElementaryCA(L=L, T=T)
    metric = ZlibConditionalComplexity()
    strat_seed = BitFlipSeedStrategy()
    
    save_dir = os.path.join("Analysis", "GeneralScripts", "Saved Figures")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dist_sorted = sorted(list(dist_to_rules.keys()))
    num_clusters = len(dist_sorted)
    
    # Create the combined figure with a single plot
    plt.figure(figsize=(12, 8))
    
    # Define a color map for the different distances
    colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))
    
    # 2. Iterate over each cluster distance
    for i, dist in enumerate(dist_sorted):
        rules = dist_to_rules[dist]
        print(f"\nProcessing Lambda Distance: |λ - 0.5| = {dist} ({len(rules)} rules)")
        
        complexities = []
        
        # 3. Gather data for this specific cluster
        for rule in tqdm(rules, desc=f"Dist {dist}"):
            for _ in range(num_samples_per_rule):
                seed = engine.generate_seed(seed_type="random")
                x = engine.run(rule, seed)
                
                # Flip a random bit in the seed
                bit_idx = np.random.randint(0, L)
                _, new_seed = strat_seed.apply(engine, rule, seed, bit_idx)
                y = engine.run(rule, new_seed)
                
                k_cond = metric.calculate(x, y)
                complexities.append(k_cond)
                
        # 4. Process and Plot on the same figure
        complexities = np.array(complexities)
        unique_ks, counts = np.unique(complexities, return_counts=True)
        
        mask = (unique_ks > 0) & (counts > 0)
        log_ks = unique_ks[mask]
        log_counts = counts[mask]
        
        # Plot raw data points with high transparency
        plt.scatter(log_ks, log_counts, color=colors[i], alpha=0.15, s=15)
        
        # Plot the Fit line with full opacity for the legend
        if len(log_ks) > 1:
            slope, intercept = np.polyfit(np.log10(log_ks), np.log10(log_counts), 1)
            fit_line = 10**(slope * np.log10(log_ks) + intercept)
            plt.plot(log_ks, fit_line, color=colors[i], linestyle='-', linewidth=2.5, 
                     label=f"|λ - 0.5| = {dist} (slope: {slope:.2f})")
        
    plt.xscale('log')
    plt.yscale('log')
    
    plt.title(f"Overlay: Conditional Complexity K(y|x) Log-Log per Lambda Distance\n(L={L}, T={T}, {num_samples_per_rule} samples/rule)", fontsize=16)
    plt.xlabel("Log K(y|x)")
    plt.ylabel("Log Frequency")
    plt.grid(True, which="both", alpha=0.1)
    
    # Legend shows the distances and their corresponding slopes
    plt.legend(title="Lambda Distances", loc='upper right', framealpha=0.9)
    plt.tight_layout()
    
    filename = "cond_complexity_loglog_lambda_overlay.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    print(f"\nOverlay plot saved to: {filepath}")
    plt.show()

if __name__ == "__main__":
    # Adjust SAMPLES higher (e.g., 400) for final run quality
    run_lambda_distance_experiments_combined(L=128, T=128, num_samples_per_rule=100)
