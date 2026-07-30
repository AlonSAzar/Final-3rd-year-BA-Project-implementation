import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.ComplexityMeasures.complexity import ZlibComplexity

def run_ca_complexity_experiment(L=64, T=64, num_samples_per_rule=10, rules=range(256)):
    """
    Measures the frequency of complexities for outputs of CA rules with random seeds.
    """
    engine = ElementaryCA(L=L, T=T)
    metric = ZlibComplexity()
    
    complexities = []
    
    print(f"Running CA experiment: L={L}, T={T}, {num_samples_per_rule} samples/rule...")
    for rule in tqdm(rules, desc="Rules"):
        for _ in range(num_samples_per_rule):
            seed = engine.generate_seed(seed_type="random")
            # Run CA and take the entire phenotype
            full_history = engine.run(rule, seed)
            phenotype = full_history
            
            k = metric.calculate(phenotype)
            complexities.append(k)
            
    return np.array(complexities)

def plot_complexity_bar_log(complexities, title="Complexity Frequency for CA Random Seeds"):
    """
    Plots a bar plot of complexity frequencies with a logarithmic y-axis.
    """
    plt.figure(figsize=(12, 6))
    
    # Calculate frequencies for unique complexity values
    unique_ks, counts = np.unique(complexities, return_counts=True)
    
    # Sorting for bar chart (even if X is log, we plot bars at unique values)
    order = np.argsort(unique_ks)
    unique_ks = unique_ks[order]
    counts = counts[order]
    
    # Plotting as bar chart
    plt.bar(unique_ks, counts, color='teal', edgecolor='black', alpha=0.7, width=np.min(np.diff(unique_ks))*0.8 if len(unique_ks) > 1 else 1.0)
    
    # Logarithmic scale for both axes
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel("Complexity K (Zlib) [Log Scale]")
    plt.ylabel("Frequency [Log Scale]")
    plt.title(title)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Add some stats to the plot
    stats_text = (
        f"Mean K: {np.mean(complexities):.2f}\n"
        f"Median K: {np.median(complexities):.1f}\n"
        f"Total Samples: {len(complexities)}"
    )
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Ensure directory exists before saving
    import os
    save_dir = os.path.join("Analysis", "GeneralScripts", "Saved Figures")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, "ca_complexity_frequency_bar.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # Parameters matching common workspace context (L=T=64)
    # Using 50 samples per rule for a representative distribution across all 256 rules.
    data = run_ca_complexity_experiment(L=128, T=128, num_samples_per_rule=50)
    plot_complexity_bar_log(data)
