import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.Engines.OneDimensionEngines.OneDimensionNoiseEngine import RandomNoiseCA
from Core.ComplexityMeasures.conditional_complexity import ZlibConditionalComplexity
from Core.strategies import BitFlipRuleStrategy, BitFlipSeedStrategy
from Experiments.experiments import shuffle_space_time

def run_conditional_complexity_experiment(L=64, T=64, num_samples_per_rule=10, rules=range(256), shuffle_toggle=False, perturbation_type="noise"):
    """
    Measures the frequency of conditional complexities K(y|x) where:
    x is the original CA output (ElementaryCA)
    y is the perturbed CA output
    """
    # Engines
    original_engine = ElementaryCA(L=L, T=T)
    metric = ZlibConditionalComplexity()
    
    # Setup perturbation
    noise_level = 0.003
    noise_engine = RandomNoiseCA(noise_level=noise_level, L=L, T=T)
    
    strat_rule = BitFlipRuleStrategy()
    strat_seed = BitFlipSeedStrategy()
    
    conditional_complexities = []
    
    perturb_name = ""
    if perturbation_type == "noise":
        perturb_name = noise_engine.name()
    elif perturbation_type == "rule_bit":
        perturb_name = strat_rule.name()
    elif perturbation_type == "seed_bit":
        perturb_name = strat_seed.name()

    label_prefix = " (Shuffled)" if shuffle_toggle else ""
    print(f"Running Conditional Complexity Experiment ({perturb_name}){label_prefix}: L={L}, T={T}, {num_samples_per_rule} samples/rule...")
    
    for rule in tqdm(rules, desc=f"Rules{label_prefix}"):
        for _ in range(num_samples_per_rule):
            seed = original_engine.generate_seed(seed_type="random")
            
            # Generate original (x)
            x = original_engine.run(rule, seed)
            
            # Generate perturbed (y)
            if perturbation_type == "noise":
                y = noise_engine.run(rule, seed)
            elif perturbation_type == "rule_bit":
                # Flip a random bit in the rule (0-7)
                bit_idx = np.random.randint(0, 8)
                new_rule, _ = strat_rule.apply(original_engine, rule, seed, bit_idx)
                y = original_engine.run(new_rule, seed)
            elif perturbation_type == "seed_bit":
                # Flip a random bit in the seed
                bit_idx = np.random.randint(0, L)
                _, new_seed = strat_seed.apply(original_engine, rule, seed, bit_idx)
                y = original_engine.run(rule, new_seed)
            else:
                raise ValueError(f"Unknown perturbation type: {perturbation_type}")
            
            if shuffle_toggle:
                x = shuffle_space_time(x)
                y = shuffle_space_time(y)
                
            k_cond = metric.calculate(x, y)
            conditional_complexities.append(k_cond)
            
    return np.array(conditional_complexities), perturb_name

def plot_conditional_complexity_histogram(complexities, engine_name, L=64, T=64, shuffle_toggle=False):
    """
    Plots a histogram of conditional complexity frequencies in Linear, Semilog-Y, and Log-Log scales.
    These visualizations help identify if the distribution follows a Power Law, Log-Normal, or Exponential decay.
    Each plot is created as a separate figure.
    """
    unique_ks, counts = np.unique(complexities, return_counts=True)
    
    # Sorting for plotting
    order = np.argsort(unique_ks)
    unique_ks = unique_ks[order]
    counts = counts[order]
    
    # Remove zeros for log-log plotting if they exist
    mask = (unique_ks > 0) & (counts > 0)
    log_ks = unique_ks[mask]
    log_counts = counts[mask]

    shuffle_suffix = " (Shuffled)" if shuffle_toggle else ""
    base_title = f"Conditional Complexity K(y|x) Frequency (L={L}, T={T})\nPerturbation: {engine_name}{shuffle_suffix}"
    
    save_dir = os.path.join("Analysis", "GeneralScripts", "Saved Figures")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. Linear Plot (Standard)
    plt.figure(figsize=(10, 6))
    plt.bar(unique_ks, counts, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f"{base_title} (Linear Scale)")
    plt.xlabel("Conditional Complexity K(y|x) [Zlib]")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.2)
    
    # Add stats to the linear plot
    stats_text = (
        f"Engine: {engine_name}\n"
        f"Mean K(y|x): {np.mean(complexities):.2f}\n"
        f"Median K(y|x): {np.median(complexities):.1f}\n"
        f"Samples: {len(complexities)}"
    )
    plt.gca().text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    filename_lin = f"conditional_complexity_linear{'_shuffled' if shuffle_toggle else ''}.png"
    plt.savefig(os.path.join(save_dir, filename_lin))
    plt.show()

    # 2. Semi-Log Y Plot (Log Frequency)
    plt.figure(figsize=(10, 6))
    plt.bar(unique_ks, counts, color='salmon', edgecolor='black', alpha=0.7)
    plt.yscale('log')
    plt.title(f"{base_title} (Semi-Log Y Scale)")
    plt.xlabel("Conditional Complexity K(y|x) [Zlib]")
    plt.ylabel("Frequency (Log)")
    plt.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    filename_logy = f"conditional_complexity_semilog{'_shuffled' if shuffle_toggle else ''}.png"
    plt.savefig(os.path.join(save_dir, filename_logy))
    plt.show()

    # 3. Log-Log Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(log_ks, log_counts, color='purple', alpha=0.6, s=15, label="Data")
    
    # Linear Fit on Log-Log Scale (Power Law Fit)
    if len(log_ks) > 1:
        # Perform linear regression on log values
        slope, intercept = np.polyfit(np.log10(log_ks), np.log10(log_counts), 1)
        fit_line = 10**(slope * np.log10(log_ks) + intercept)
        plt.plot(log_ks, fit_line, color='darkorange', linestyle='--', linewidth=2, 
                 label=f"Power Law Fit (slope: {slope:.2f})")
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"{base_title} (Log-Log Scale)")
    plt.xlabel("Log K(y|x)")
    plt.ylabel("Log Frequency")
    plt.grid(True, which="both", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    filename_loglog = f"conditional_complexity_loglog{'_shuffled' if shuffle_toggle else ''}.png"
    plt.savefig(os.path.join(save_dir, filename_loglog))
    plt.show()
    
    print(f"Separate plots saved to {save_dir}")

if __name__ == "__main__":
    # Parameters
    L_VAL = 128
    T_VAL = 128
    SAMPLES = 40000  # Reduced for quicker testing, increase for final run
    SHUFFLE = False
    
    # PERTURBATION_TYPE options: "noise", "rule_bit", "seed_bit"
    PERTURB_TYPE = "seed_bit"
    
    # Run for a subset of rules if needed, or range(256)
    RULES_TO_RUN = [110]

    data, perturb_name = run_conditional_complexity_experiment(
        L=L_VAL, T=T_VAL, 
        num_samples_per_rule=SAMPLES, 
        rules=RULES_TO_RUN,
        shuffle_toggle=SHUFFLE,
        perturbation_type=PERTURB_TYPE
    )
    
    plot_conditional_complexity_histogram(data, perturb_name, L=L_VAL, T=T_VAL, shuffle_toggle=SHUFFLE)
