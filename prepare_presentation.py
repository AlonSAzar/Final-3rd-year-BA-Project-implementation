import numpy as np
import os
import matplotlib.pyplot as plt
from Experiments.BasicComplexityExperiments.SimplicityBiasExperiment import SimplicityBiasExperiment
from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.ComplexityMeasures.complexity import ZlibComplexity

def prepare_presentation_plots():
    # Set localized EVEN BIGGER font sizes for poster presentation
    plt.rcParams.update({
        'font.size': 23,
        'axes.titlesize': 26,
        'axes.labelsize': 24,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'figure.titlesize': 32
    })
    
    L = 12
    T = 64
    
    # Toggle for deterministic (all possible) vs sampled seeds
    USE_ALL_SEEDS = True
    if USE_ALL_SEEDS:
        NUM_SEEDS = 2**L
    else:
        NUM_SEEDS = 10
    
    engine = ElementaryCA(L, T)
    metric = ZlibComplexity()
    sb_exp = SimplicityBiasExperiment(engine, metric)
    
    print("Running Regular Simplicity Bias...")
    # Essential: Must run the experiment first to populate results
    sb_exp.run(num_seeds=NUM_SEEDS, rules=range(256), just_distribute=True, use_all_seeds=USE_ALL_SEEDS)
    
    # Get regular data
    freqs_reg, comps_reg, img_reg = sb_exp.analyze(shuffle_control=False)
    
    print("Running Shuffled Simplicity Bias...")
    # Get shuffled data
    freqs_shuff, comps_shuff, img_shuff = sb_exp.analyze(shuffle_control=True)
    
    # Calculate global axes
    total_reg = sum(freqs_reg.values())
    total_shuff = sum(freqs_shuff.values())
    
    log_reg = [np.log10(c/total_reg) for c in freqs_reg.values()]
    log_shuff = [np.log10(c/total_shuff) for c in freqs_shuff.values()]
    
    k_reg = list(comps_reg.values())
    k_shuff = list(comps_shuff.values())
    
    xlim = (min(min(k_reg), min(k_shuff)) * 0.9, max(max(k_reg), max(k_shuff)) * 1.1)
    ylim = (min(min(log_reg), min(log_shuff)) - 0.5, -1.5)
    
    # --- PLOTTING REGULAR ---
    print("Plotting Regular (Clean)...")
    sb_exp.run(num_seeds=NUM_SEEDS, rules=range(256), shuffle_control=False, 
               annotated=False, xlim=xlim, ylim=ylim, hide_stats=True, skip_simulation=True)
    
    print("Plotting Regular (Annotated)...")
    sb_exp.run(num_seeds=NUM_SEEDS, rules=range(256), shuffle_control=False, 
               annotated=True, xlim=xlim, ylim=ylim, hide_stats=True, save_examples=True, skip_simulation=True)
    
    # --- PLOTTING SHUFFLED ---
    print("Plotting Shuffled (Clean)...")
    sb_exp.run(num_seeds=NUM_SEEDS, rules=range(256), shuffle_control=True, 
               annotated=False, xlim=xlim, ylim=ylim, hide_stats=True, skip_simulation=True)

    print("Plotting Shuffled (Annotated)...")
    sb_exp.run(num_seeds=NUM_SEEDS, rules=range(256), shuffle_control=True, 
               annotated=True, xlim=xlim, ylim=ylim, hide_stats=True, save_examples=True, skip_simulation=True)

if __name__ == "__main__":
    prepare_presentation_plots()
