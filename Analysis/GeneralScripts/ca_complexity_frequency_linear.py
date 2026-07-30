import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.ComplexityMeasures.complexity import ZlibComplexity
from Experiments.experiments import shuffle_space_time
import os

def run_ca_complexity_experiment(L=64, T=64, num_samples_per_rule=10, rules=range(256), shuffle_control=False):
    """
    Measures the frequency of complexities for outputs of CA rules with random seeds.
    """
    engine = ElementaryCA(L=L, T=T)
    metric = ZlibComplexity()
    
    complexities = []
    
    shuffle_str = " (Shuffled Control)" if shuffle_control else ""
    print(f"Running CA experiment{shuffle_str}: L={L}, T={T}, {num_samples_per_rule} samples/rule...")
    for rule in tqdm(rules, desc=f"Rules{shuffle_str}"):
        for _ in range(num_samples_per_rule):
            seed = engine.generate_seed(seed_type="random")
            # Run CA and take the entire phenotype
            full_history = engine.run(rule, seed)
            phenotype = full_history
            
            if shuffle_control:
                phenotype = shuffle_space_time(phenotype)
            
            k = metric.calculate(phenotype)
            complexities.append(k)
            
    return np.array(complexities)

def plot_complexity_bar_linear(complexities, title="Complexity Frequency for CA Random Seeds (Linear Scale)"):
    """
    Plots a bar plot of complexity frequencies with linear axes.
    """
    plt.figure(figsize=(12, 6))
    
    # Calculate frequencies for unique complexity values
    unique_ks, counts = np.unique(complexities, return_counts=True)
    
    # Sorting for bar chart
    order = np.argsort(unique_ks)
    unique_ks = unique_ks[order]
    counts = counts[order]
    
    # Plotting as bar chart
    plt.bar(unique_ks, counts, color='coral', edgecolor='black', alpha=0.7)
    
    # Linear scale (default)
    plt.xscale('linear')
    plt.yscale('linear')
    
    plt.xlabel("Complexity K (Zlib)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Add stats to the plot
    stats_text = (
        f"Mean K: {np.mean(complexities):.2f}\n"
        f"Median K: {np.median(complexities):.1f}\n"
        f"Total Samples: {len(complexities)}"
    )
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Ensure directory exists before saving
    save_dir = os.path.join("Analysis", "GeneralScripts", "Saved Figures")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, "ca_complexity_frequency_linear.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

def plot_complexity_bar_semilog(complexities, title="Complexity Frequency for CA Random Seeds (Semi-Log Scale)"):
    """
    Plots a bar plot of complexity frequencies where ONLY the Y-axis is log.
    """
    plt.figure(figsize=(12, 6))
    
    unique_ks, counts = np.unique(complexities, return_counts=True)
    order = np.argsort(unique_ks)
    unique_ks = unique_ks[order]
    counts = counts[order]
    
    plt.bar(unique_ks, counts, color='mediumpurple', edgecolor='black', alpha=0.7)
    
    plt.xscale('linear')
    plt.yscale('log')
    
    plt.xlabel("Complexity K (Zlib) [Linear]")
    plt.ylabel("Frequency [Log Scale]")
    plt.title(title)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    
    save_dir = os.path.join("Analysis", "GeneralScripts", "Saved Figures")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "ca_complexity_frequency_semilog.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__" :
    # Parameters matching workspace context (L=T=64)
    SHUFFLE_CONTROL = True # Set to True to run the shuffled control experiment
    
    data = run_ca_complexity_experiment(L=12, T=64, num_samples_per_rule=50, shuffle_control=SHUFFLE_CONTROL)
    
    title_suffix = " (Shuffled Control)" if SHUFFLE_CONTROL else ""
    plot_complexity_bar_linear(data, title=f"Complexity Frequency for CA Random Seeds (Linear Scale){title_suffix}")
    plot_complexity_bar_semilog(data, title=f"Complexity Frequency for CA Random Seeds (Semi-Log Scale){title_suffix}")