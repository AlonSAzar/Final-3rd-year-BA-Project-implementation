import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from Core.ComplexityMeasures.complexity import ZlibComplexity
import scipy.linalg as la

def plot_conditional_transition_matrix(experiment, bins=10, log_bins=False):
    """
    Creates a transition matrix based on the output of ConditionalTransitionExperiment.
    Maps: Complexity of Parent X -> Complexity of Child Y.
    """
    print("Generating Conditional Transition Matrix...")
    
    # Capture metadata from experiment/engine
    metadata = {
        'L': getattr(experiment.engine, 'L', 'N/A'),
        'T': getattr(experiment.engine, 'T', 'N/A'),
        'strategy': getattr(experiment, 'strategy_name', 'Unknown'),
        'last_row_only': getattr(experiment, 'last_row_only', 'N/A')
    }

    # 1. Prepare complexity calculator
    metric = ZlibComplexity()
    
    # 2. Extract transitions (h_x, h_y) and their counts
    transitions = experiment.return_transitions()
    phenotype_cache = experiment.return_phenotype_cache()
    
    if not transitions:
        print("No transitions found. Did you run the experiment?")
        return

    # 3. Calculate complexities
    comp_cache = {}
    all_sync_k = []
    all_async_k = []
    weights = []

    for (h_x, h_y), count in transitions.items():
        if h_x not in comp_cache:
            comp_cache[h_x] = metric.calculate(phenotype_cache[h_x])
        if h_y not in comp_cache:
            comp_cache[h_y] = metric.calculate(phenotype_cache[h_y])
            
        all_sync_k.append(comp_cache[h_x])
        all_async_k.append(comp_cache[h_y])
        weights.append(count)

    all_sync_k = np.array(all_sync_k)
    all_async_k = np.array(all_async_k)
    weights = np.array(weights)

    # 4. Binning
    min_k = min(all_sync_k.min(), all_async_k.min())
    max_k = max(all_sync_k.max(), all_async_k.max())
    
    if log_bins:
        # Ensure min_k > 0 for logspace. If it's 0 (or less), start from a small epsilon above 0
        effective_min = max(min_k, 1e-3)
        bin_edges = np.logspace(np.log10(effective_min), np.log10(max_k), bins + 1)
    else:
        bin_edges = np.linspace(min_k, max_k, bins + 1)

    # 5. Create 2D Weighted Histogram
    matrix, xedges, yedges = np.histogram2d(
        all_sync_k, 
        all_async_k, 
        bins=bin_edges, 
        weights=weights
    )

    # 6. Normalize Row-wise
    row_sums = matrix.sum(axis=1)
    norm_matrix = np.divide(
        matrix.T, 
        row_sums, 
        out=np.zeros_like(matrix.T), 
        where=row_sums != 0
    ).T

    # 7. Plotting
    plt.figure(figsize=(12, 10))
    
    # Create range labels for bins instead of just centers
    x_labels = [f"[{xedges[i]:.1f}, {xedges[i+1]:.1f})" for i in range(len(xedges)-1)]
    y_labels = [f"[{yedges[i]:.1f}, {yedges[i+1]:.1f})" for i in range(len(yedges)-1)]

    sns.heatmap(
        norm_matrix.T, 
        annot=True, 
        fmt=".2f", 
        cmap="YlOrBr",
        xticklabels=x_labels,
        yticklabels=y_labels,
        cbar_kws={'label': 'Transition Probability P(K_child | K_parent)'}
    )
    
    plt.gca().invert_yaxis()
    plt.xlabel("Parent Phenotype Complexity (K_x)")
    plt.ylabel("Child Phenotype Complexity (K_y)")
    plt.title("Phenotype Complexity Transition Matrix")
    
    # Metadata text box
    stats_text = (
        f"L: {metadata.get('L', 'N/A')}, T: {metadata.get('T', 'N/A')}\n"
        f"Strategy: {metadata.get('strategy', 'N/A')}\n"
        f"Last Row Only: {metadata.get('last_row_only', 'N/A')}"
    )
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             fontsize=9, verticalalignment='top', family='monospace')

    plt.plot([0, bins], [0, bins], color='blue', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show(block=True)

def analyze_stationary_distribution(experiment):
    """
    Analyzes the transition matrix as a Markov Chain.
    1. Maps phenotype hashes to Integer IDs.
    2. Builds the exact transition matrix M.
    3. Calculates the stationary distribution via the leading eigenvector.
    """
    print("Performing Markov Chain Eigen-Analysis on Phenotypes...")
    transitions = experiment.return_transitions()
    
    # Capture metadata from experiment/engine
    metadata = {
        'L': getattr(experiment.engine, 'L', 'N/A'),
        'T': getattr(experiment.engine, 'T', 'N/A'),
        'strategy': getattr(experiment, 'strategy_name', 'Unknown'),
        'last_row_only': getattr(experiment, 'last_row_only', 'N/A')
    }
    
    # 1. Map every unique phenotype hash to an Integer ID (0, 1, 2... N)
    unique_hashes = set()
    for (h1, h2) in transitions.keys():
        unique_hashes.add(h1)
        unique_hashes.add(h2)
    
    unique_hashes = sorted(list(unique_hashes))
    hash_to_id = {h: i for i, h in enumerate(unique_hashes)}
    N = len(unique_hashes)
    print(f"Total Unique Phenotypes (States): {N}")

    # 2. Initialize a square matrix M of size (N, N) with zeros.
    M = np.zeros((N, N))

    # 3. Fill the matrix using the 'transitions' counter:
    for (h_parent, h_child), count in transitions.items():
        i = hash_to_id[h_parent]
        j = hash_to_id[h_child]
        M[i, j] = count

    # 4. Normalize the rows so they sum to 1 (Probabilities)
    # Handle cases where row sum might be zero (though unlikely in this experiment)
    row_sums = M.sum(axis=1, keepdims=True)
    M_prob = np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums != 0)

    # 5. Analysis: Eigenvalues and Eigenvectors
    try:
        # We use la.eig for general matrices (M_prob is typically not symmetric)
        # We are looking for the stationary distribution (eigenvalue = 1)
        eigenvalues, left_eigenvectors = la.eig(M_prob, left=True, right=False)
        
        # Sort eigenvalues by magnitude
        abs_eigenvalues = np.abs(eigenvalues)
        sorted_indices = np.argsort(abs_eigenvalues)[::-1]
        sorted_evals = eigenvalues[sorted_indices]

        # The stationary distribution corresponds to the left eigenvector of lambda=1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(left_eigenvectors[:, idx])
        stationary = stationary / np.sum(stationary)

        # Spectral Gap: 1 - |lambda_2|
        # This measures the relaxation time to equilibrium. 
        # Large gap = fast convergence; Small gap = "Critical Slowing Down"
        
        num_one_evals = np.sum(np.abs(np.abs(eigenvalues) - 1.0) < 1e-8)
        print(f"Number of eigenvalues with value 1: {num_one_evals}")
        if num_one_evals > 1:
            print("Warning: The transition matrix is disconnected (multiple stationary states).")

        if len(sorted_evals) > 1:
            spectral_gap = 1.0 - np.abs(sorted_evals[1])
            relaxation_time = 1.0 / spectral_gap if spectral_gap > 1e-10 else np.inf
        else:
            spectral_gap = 0
            relaxation_time = 0

        print(f"--- Markov Chain Analysis ---")
        print(f"Top 10 Eigenvalues: {np.abs(sorted_evals[:10])}")
        print(f"Spectral Gap: {spectral_gap:.4f}")
        print(f"Relaxation Time (T_relax): {relaxation_time:.2f} mutations")
        
        # Plot Stationary Distribution
        spectral_info = {
            'gap': spectral_gap,
            'tau': relaxation_time,
            'components': num_one_evals,
            'metadata': metadata
        }
        plot_stationary_dist(stationary, unique_hashes, experiment.phenotype_cache, spectral_info=spectral_info)

        return {
            'matrix': M_prob,
            'stationary_dist': stationary,
            'spectral_gap': spectral_gap,
            'relaxation_time': relaxation_time,
            'hashes': unique_hashes,
            'hash_to_id': hash_to_id
        }
    except Exception as e:
        print(f"Eigen-analysis failed: {e}")
        return None

def plot_stationary_dist(stationary, hashes, phenotype_cache, spectral_info=None):
    """
    Visualizes the stationary distribution against phenotype complexity.
    Now with mean and confidence intervals across complexity bins.
    """
    metric = ZlibComplexity()
    complexities = np.array([metric.calculate(phenotype_cache[h]) for h in hashes])
    stationary = np.array(stationary)
    
    # Binning to calculate confidence intervals
    num_bins = 15
    # Using fixed 15 bins for the plot data aggregation
    bins = np.linspace(complexities.min(), complexities.max(), 15)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    bin_means = []
    bin_stds = []
    
    for i in range(len(bins)-1):
        mask = (complexities >= bins[i]) & (complexities < bins[i+1])
        if np.any(mask):
            bin_vals = stationary[mask]
            bin_means.append(np.mean(bin_vals))
            # Standard error of the mean
            bin_stds.append(np.std(bin_vals) / np.sqrt(len(bin_vals)))
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)

    plt.figure(figsize=(12, 7))
    
    # Individual points
    plt.scatter(complexities, stationary, alpha=0.15, c='gray', s=8, label='Individual Phenotypes')
    
    # Error bars (95% CI roughly 1.96 * SEM)
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    valid = ~np.isnan(bin_means)
    
    # Calculate CI and determine bottom limit for y-axis
    yerr = 1.96 * bin_stds[valid]
    lower_bounds = bin_means[valid] - yerr
    # In log scale, non-positive values can't be plotted. Filter them for limit calculation.
    valid_lower = lower_bounds[lower_bounds > 0]
    min_y = min(stationary.min(), valid_lower.min()) if len(valid_lower) > 0 else stationary.min()
    
    plt.errorbar(bin_centers[valid], bin_means[valid], yerr=yerr, 
                 fmt='o', color='forestgreen', ecolor='red', capsize=5, capthick=2, 
                 label='Binned Mean ± 95% CI')
    
    # Metadata text box
    if spectral_info:
        meta = spectral_info.get('metadata', {})
        stats_text = (
            f"L: {meta.get('L', 'N/A')}, T: {meta.get('T', 'N/A')}\n"
            f"Strategy: {meta.get('strategy', 'N/A')}\n"
            f"Last Row Only: {meta.get('last_row_only', 'N/A')}\n"
            f"States (N): {len(hashes)}\n"
            f"Spectral Gap (γ): {spectral_info['gap']:.4f}\n"
            f"Relaxation Time ($T_{{relax}}$): {spectral_info['tau']:.1f}\n"
            f"Components ($λ=1$): {spectral_info['components']}"
        )
        plt.text(0.02, 0.05, stats_text, transform=plt.gca().transAxes, 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                 fontsize=9, verticalalignment='bottom', family='monospace')

    plt.yscale('log')
    # Set y-limit to ensure all CIs and individual points are visible
    # We use a small epsilon for the lower bound to avoid log(0) issues
    buffer = 0.8
    y_min_data = min(stationary.min(), min_y)
    plt.ylim(y_min_data * buffer, stationary.max() * 1.5)
    
    plt.xlabel("Phenotype Complexity (Zlib)")
    plt.ylabel("Stationary Probability (log scale)")
    plt.title("Stationary Distribution vs. Complexity")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show(block=True)

if __name__ == "__main__":

    print("This file contains the transition matrix logic for ConditionalTransitionExperiment.")
