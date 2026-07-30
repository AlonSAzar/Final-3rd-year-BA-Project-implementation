import numpy as np
import os
import matplotlib.pyplot as plt
from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.ComplexityMeasures.conditional_complexity import ZlibConditionalComplexity
from Experiments.ConditionalComplexityExperiments.ConditionalTransitionExperiment import ConditionalTransitionExperiment
from Core import strategies

def prepare_csb_presentation():
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

    L = 6
    T = 6
    
    # Toggle for deterministic (all possible) vs sampled parent seeds
    USE_ALL_SEEDS = True
    if USE_ALL_SEEDS:
        NUM_PARENTS = 2**L
    else:
        NUM_PARENTS = 256
    
    engine = ElementaryCA(L, T)
    metric = ZlibConditionalComplexity()
    strategy = strategies.BitFlipRuleStrategy()
    
    # Run Experiment Single Pass
    print("Running Conditional Simplicity Bias (CSB) simulation...")
    csb_exp = ConditionalTransitionExperiment(engine, metric, last_row_only=False)
    # We need to run the simulation first to populate metrics
    csb_exp.run(rules=range(256), num_parents=NUM_PARENTS, strategy=strategy, just_distribute=True, use_all_seeds=USE_ALL_SEEDS)
    
    print("Plotting CSB Transitions (Clean)...")
    # To get a seamless transition, we use the annotated plotter but tell it not to show annotations yet
    from Visualizations.annotated_plotter import AnnotatedSimplicityBiasPlotter
    plotter = AnnotatedSimplicityBiasPlotter(metric, engine, NUM_PARENTS)
    
    # Pre-analyze data to pass to plotter
    plot_data_k = []
    plot_data_prob = []
    total_mutations_per_parent = {}
    for (h_x, h_y), count in csb_exp.transitions.items():
        total_mutations_per_parent[h_x] = total_mutations_per_parent.get(h_x, 0) + count
    for (h_x, h_y), count in csb_exp.transitions.items():
        img_x = csb_exp.phenotype_cache[h_x]
        img_y = csb_exp.phenotype_cache[h_y]
        plot_data_k.append(metric.calculate(img_x, img_y))
        plot_data_prob.append(np.log10(count / total_mutations_per_parent[h_x]))
    
    # Plot clean version using the SAME layout as annotated (by calling plot_csb with num_annotations=0)
    plotter.plot_csb(plot_data_k, plot_data_prob, {}, strategy, False, num_annotations=0, title_pad=0)
    
    print("Plotting CSB Transitions (Annotated)...")
    # This will show parent -> child phenotype pairs
    csb_exp.run(rules=range(256), num_parents=NUM_PARENTS, strategy=strategy, annotated=True, skip_simulation=True, title_pad=0)

if __name__ == "__main__":
    prepare_csb_presentation()
