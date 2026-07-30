from Analysis.PCA.AnalyzeFlowField import analyze_flow_field
from Analysis.PCA.TextureFlowField import analyze_texture_flow_field
from Analysis.PCA.AttractorBasin import analyze_attractor_basin
from Analysis.PCA.PCARawPhenotypes import analyze_pca
from Analysis.PCA.SelfTransitionPlot import analyze_self_transitions
from Analysis.TSNE.TSNERawPhenotypes import analyze_tsne
from Core import strategies
from Core.ComplexityMeasures.complexity import ZlibComplexity
from Core.ComplexityMeasures.conditional_complexity import ZlibConditionalComplexity
from Core.Engines.OneDimensionEngines.OneDimensionAsynchronousEngine import AsynchronousCA
from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.Engines.OneDimensionEngines.OneDimensionFrozenCellsEngine import FrozenCellsCA
from Core.Engines.OneDimensionEngines.OneDimensionNoiseEngine import RandomNoiseCA
from Experiments.ConditionalComplexityExperiments.ConditionalTransitionExperiment import ConditionalTransitionExperiment
from Experiments.EvolutionExperiments.GliderExperiments.EvolutionaryGliderDynamicsExperiment import \
    EvolutionaryDynamicsAnalyzer
from Experiments.EvolutionExperiments.GliderExperiments.GliderEvolutionExperiment import GliderEvolutionExperiment
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from Analysis.PCA.PCAPatchHistogram import get_1d_texture_vector, analyze_texture_pca
from Experiments.BasicComplexityExperiments.PopulationGrowthExperiment import PopulationGrowthExperiment
from Experiments.BasicComplexityExperiments.RobustnessExperiment import RobustnessExperiment
from Experiments.BasicComplexityExperiments.SimplicityBiasExperiment import SimplicityBiasExperiment
from Visualizations.interactive_utils import plot_interactive_scatter
from Analysis.SpectralGapTemperatureExperiment import run_spectral_temp_scan
from Analysis.ConditionalComplexityTransitionMatrix import plot_conditional_transition_matrix, \
    analyze_stationary_distribution
from Core import ComplexityMeasures
import matplotlib.pyplot as plt

# Set global font sizes (x1.5 of defaults)
plt.rcParams.update({
    'font.size': 18,  # Increased from 15
    'axes.titlesize': 22,  # Increased from 17
    'axes.labelsize': 20,  # Increased from 15
    'xtick.labelsize': 16,  # Increased from 12
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 24  # Increased from 21
})


def main():
    # 1. Setup Configuration
    # TODO with bigger L, it seems further from the limit, weird
    L = 64
    T = 64
    NUM_SEEDS = 100

    # 2. Instantiate Objects
    engine = ElementaryCA(L, T)
    frozen_engine = FrozenCellsCA(0.1, L, T)
    random_noise_engine = RandomNoiseCA(0.01, L, T)
    asynchronous_engine = AsynchronousCA(L, T)

    metric = ZlibComplexity()  # Can easily swap this with MutualInfoComplexity()

    # --- Experiment A: Simplicity Bias ---
    """
    We generate NUM_SEEDS (default 100) different seeds per the 256 different rules.
    """
    print("--- Starting Simplicity Bias Experiment ---")
    sb_exp = SimplicityBiasExperiment(engine, metric)
    sb_exp.run(num_seeds=NUM_SEEDS, annotated=True)
    sb_exp.run(num_seeds=NUM_SEEDS, shuffle_control=True)
    #
    # # --- Experiment B: Population Growth Simplicity Bias ---
    print("--- Starting Population Growth Experiment ---")
    pop_exp = PopulationGrowthExperiment(engine, metric)
    pop_exp.run(num_seeds=NUM_SEEDS, iterate_all_seeds=False)
    pop_exp.run(num_seeds=NUM_SEEDS, shuffle_control=True, iterate_all_seeds=True)

    # --- Experiment C: Robustness ---
    print("--- Starting Robustness Experiments ---")
    rob_exp = RobustnessExperiment(engine, metric)

    rob_exp.run(strategy=strategies.SameSeedAndRuleStrategy(), mut_engine=frozen_engine)
    rob_exp.run(strategy=strategies.SameSeedAndRuleStrategy(), mut_engine=random_noise_engine)
    rob_exp.run(strategy=strategies.SameSeedAndRuleStrategy(), mut_engine=asynchronous_engine)


    print("Running Rule Flip...")
    rob_exp.run(strategy=strategies.BitFlipRuleStrategy())

    print("Running Rule Flib, Shuffled...")
    rob_exp.run(strategy=strategies.BitFlipRuleStrategy(), shuffle_control=True)

    print("Running Seed Flip...")
    rob_exp.run(strategy=strategies.BitFlipSeedStrategy())

    rob_exp.run(strategy=strategies.BitFlipSeedStrategy(), shuffle_control=True)


    print("Running Random Seed...")
    rob_exp.run(strategy=strategies.RandomSeedStrategy())

    print("Running Random Rule...")
    rob_exp.run(strategy=strategies.RandomRuleStrategy())

    print("Running Random Seed and Rule...")
    rob_exp.run(strategy=strategies.RandomSeedAndRuleStrategy())

    # After all robustness experiments, plot the comparison
    RobustnessExperiment.plot_session_comparison()
    # #
    # Initialize Experiment
    # Complexity Bonus = True means we reward simple seeds creating complex patterns
    evo_exp = GliderEvolutionExperiment(engine, pop_size=50, mutation_rate=0.03, complexity_bonus=True)

    # Run Scan
    evo_exp.scan_all_rules(generations_per_rule=20)

    # 1. View the Landscape
    # This shows you which rules support "Life" (Class 4) vs "Death" (Class 1) or "Chaos" (Class 3)
    evo_exp.plot_distribution()

    # 2. View the Champions
    # This shows the specific structures evolved.
    evo_exp.visualize_top_rules(top_n=5)

    # 1. Instantiate the Analyzer
    analyzer = EvolutionaryDynamicsAnalyzer(evo_exp)

    # 2. Pick your winners
    # (These are the rules you found in the bar chart earlier)
    top_rules = [169, 73, 192, 110, 54]  # Replace with your actual winners

    # 3. Run and Plot
    analyzer.compare_top_rules(top_rules, generations=100)
    #
    # --- Experiment: Conditional Complexity ---
    zlib_cond_complexity = ZlibConditionalComplexity()

    # last_row_only toggle (default is True)
    last_row = True
    strategy_obj = strategies.BitFlipSeedStrategy()

    cond_trans_exp = ConditionalTransitionExperiment(engine, zlib_cond_complexity, last_row_only=last_row)
    # TODO iterate on rules and see how conditional comp bias changes according to rule
    # Increased num_parents to 200 for tighter confidence intervals in results
    cond_trans_exp.run(strategy=strategy_obj, shuffle_control=False, num_parents=100)
    # freq_map, complexity_map = cond_trans_exp.run(shuffle_control=True)

    # Transition Matrix from Conditional Data
    plot_conditional_transition_matrix(cond_trans_exp, bins=10, log_bins=True)

    # 5. Markov Chain Analysis
    markov_results = analyze_stationary_distribution(cond_trans_exp)

    phenotype_cache = cond_trans_exp.return_phenotype_cache()
    parent_cache = cond_trans_exp.return_parent_cache()
    transitions = cond_trans_exp.return_transitions()
    creations = cond_trans_exp.return_creations()
    parent_creations = cond_trans_exp.return_parent_creations()

    # --- Virtual Temperature Analysis ---
    from Analysis.VirtualTemperatureSpectralGap import plot_reweighted_spectral_gap
    plot_reweighted_spectral_gap(cond_trans_exp)

    metadata = {
        'L': L,
        'T': T,
        'NUM_SEEDS': NUM_SEEDS,
        'strategy': strategy_obj.name(),
        'last_row': last_row
    }

    # Robustness / Self-Transition Map
    analyze_self_transitions(phenotype_cache, transitions, parent_cache=parent_cache, metadata=metadata)

    # Flow field (using weighted transitions)
    # Background points will now be filtered to parents only in the flow scripts
    analyze_flow_field(phenotype_cache, transitions, parent_cache=parent_cache, metadata=metadata)
    analyze_texture_flow_field(phenotype_cache, transitions, parent_cache=parent_cache, metadata=metadata)

    # TODO change color pallet s.t more frequent will be brighter
    # Texture Analysis - Now using Parents Only
    analyze_texture_pca(parent_cache, transitions, window_size=6, creations=parent_creations, metadata=metadata)

    # Attractor Flow - Now using Parents Only
    analyze_pca(parent_cache, transitions, creations=parent_creations, metadata=metadata)

    # TODO replace with UMAP
    # analyze_tsne(phenotype_cache, transitions)

    # Interactive Plots (using PARENTS ONLY for scatter points)
    parent_hashes = list(parent_cache.keys())
    print("Computing Texture Vectors for Interactive Plot (Parents Only)...")
    texture_matrix = []
    for h in parent_hashes:
        vec = get_1d_texture_vector(parent_cache[h], window_size=6)
        texture_matrix.append(vec)
    texture_matrix = np.array(texture_matrix)

    pca_tex = PCA(n_components=2)
    coords_tex = pca_tex.fit_transform(texture_matrix)

    # Use parent_cache and parent_creations for the interactive plots
    plot_interactive_scatter(parent_cache, transitions, coords_tex, title="Texture PCA Interactive",
                             creations=parent_creations)

    # Raw Pixel PCA Interactive
    data_pixels = np.array([parent_cache[h].flatten() for h in parent_hashes])
    pca_pix = PCA(n_components=2)
    coords_pix = pca_pix.fit_transform(data_pixels)
    plot_interactive_scatter(parent_cache, transitions, coords_pix, title="Morphospace PCA Interactive",
                             creations=parent_creations)

    # t-SNE Interactive
    print("Computing t-SNE...")
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    coords_tsne = tsne.fit_transform(texture_matrix)
    plot_interactive_scatter(parent_cache, transitions, coords_tsne, title="tSNE Morphospace Interactive",
                             creations=parent_creations)

    # Generate Interactive Plot
    plot_interactive_scatter(phenotype_cache, transitions, coords_tex, title="Texture PCA Interactive")

    # 2. Morphospace PCA (Raw Pixels)
    print("Computing Morphospace Vectors...")
    raw_matrix = np.array([phenotype_cache[h].flatten() for h in unique_hashes])

    pca_raw = PCA(n_components=2)
    coords_raw = pca_raw.fit_transform(raw_matrix)

    # Generate Interactive Plot
    plot_interactive_scatter(phenotype_cache, transitions, coords_raw, title="Morphospace PCA Interactive")

    # 3. t-SNE (The Islands)
    print("Computing t-SNE (this may take a moment)...")
    # Using 'pca' init for better stability
    tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42)
    coords_tsne = tsne.fit_transform(raw_matrix)

    # Generate Interactive Plot
    plot_interactive_scatter(phenotype_cache, transitions, coords_tsne, title="tSNE Morphospace Interactive")


if __name__ == "__main__":
    main()