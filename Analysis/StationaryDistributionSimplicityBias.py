import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to sys.path to allow imports from Core and Experiments
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scipy.stats import linregress
import scipy.linalg as la
from Core.ComplexityMeasures.complexity import ZlibComplexity
from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Experiments.ConditionalComplexityExperiments.ConditionalTransitionExperiment import ConditionalTransitionExperiment
from Core import strategies

def run_stationary_simplicity_bias(L=6, T=6, num_parents=100):
    """
    Calculates the stationary distribution of phenotypes and fits a simplicity bias boundary.
    Displays probabilities on a log-scale without complexity bins or bootstrapping.
    """
    print(f"Initializing Experiment (L={L}, T={T}, Parents={num_parents})...")
    engine = ElementaryCA(L, T)
    metric = ZlibComplexity()
    
    # We use ConditionalTransitionExperiment to get phenotype transitions
    # last_row_only=True is often used to define phenotypes by their final state
    last_row_only = True
    exp = ConditionalTransitionExperiment(engine, metric, last_row_only=last_row_only)
    
    # Run over all 256 rules with mutations
    strategy = strategies.BitFlipRuleStrategy()
    exp.run(rules=range(256), num_parents=num_parents, strategy=strategy)
    
    transitions = exp.return_transitions()
    phenotype_cache = exp.return_phenotype_cache()
    
    print("Building Transition Matrix...")
    unique_hashes = sorted(list(phenotype_cache.keys()))
    hash_to_id = {h: i for i, h in enumerate(unique_hashes)}
    N = len(unique_hashes)
    
    M = np.zeros((N, N))
    for (h_parent, h_child), count in transitions.items():
        if h_parent in hash_to_id and h_child in hash_to_id:
            M[hash_to_id[h_parent], hash_to_id[h_child]] = count
            
    # Normalize rows to get probabilities
    row_sums = M.sum(axis=1, keepdims=True)
    # Avoid division by zero for states with no outgoing transitions (shouldn't happen with BitFlip)
    M_prob = np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums != 0)
    
    print("Calculating Stationary Distribution via Eigen-analysis...")
    try:
        eigenvalues, left_eigenvectors = la.eig(M_prob, left=True, right=False)
        
        # Stationary distribution corresponds to eigenvalue 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(left_eigenvectors[:, idx])
        stationary = stationary / np.sum(stationary)
        
        # Ensure all probabilities are positive for log scale
        stationary = np.maximum(stationary, 1e-15)
        
    except Exception as e:
        print(f"Eigen-analysis failed: {e}")
        return

    print("Calculating Complexities and Fitting Simplicity Bias Boundary...")
    complexities = np.array([metric.calculate(phenotype_cache[h]) for h in unique_hashes])
    log_probs = np.log10(stationary)
    
    # Fit Upper Bound: Find max log_prob for each unique complexity
    unique_ks = np.unique(complexities)
    max_log_probs = []
    for k in unique_ks:
        max_log_probs.append(np.max(log_probs[complexities == k]))
        
    unique_ks = np.array(unique_ks)
    max_log_probs = np.array(max_log_probs)
    
    slope, intercept, r_value, p_value, std_err = linregress(unique_ks, max_log_probs)
    
    # AIT form: P(x) <= 2^(-aK - b)
    # log10(P) = slope * K + intercept
    # P = 10^(slope * K + intercept) = 2^(log2(10) * (slope * K + intercept))
    # P = 2^( (slope*log2(10))*K + (intercept*log2(10)) )
    # So a = -slope * log2(10), b = -intercept * log2(10)
    a = -slope * np.log2(10)
    b = -intercept * np.log2(10)
    
    print(f"Fit Results: slope={slope:.4f}, intercept={intercept:.4f}, R^2={r_value**2:.4f}")
    print(f"Simplicity Bias Bound: P(x) <= 2^(-{a:.2f}K - {b:.2f})")

    # Plotting
    plt.figure(figsize=(10, 7))
    
    # Scatter plot of all phenotypes
    plt.scatter(complexities, stationary, alpha=0.4, s=15, edgecolors='none', label='Phenotypes')
    
    # Plot the fit boundary
    fit_ks = np.linspace(complexities.min(), complexities.max(), 100)
    fit_log_probs = slope * fit_ks + intercept
    plt.plot(fit_ks, 10**fit_log_probs, color='red', linestyle='--', linewidth=2, 
             label=f'Upper Bound: $P \\leq 2^{{-{a:.1f}K - {b:.1f}}}$')
    
    plt.yscale('log')
    plt.xlabel("Complexity $K(x)$ (Zlib)")
    plt.ylabel("Stationary Probability $P(x)$")
    plt.title(f"Stationary Phenotype Distribution (L={L}, T={T})")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    # Add stats box
    stats_text = (
        f"L: {L}, T: {T}\n"
        f"Strategy: {strategy.name()}\n"
        f"Last Row Only: {last_row_only}\n"
        f"States (N): {N}\n"
        f"Slope: {slope:.3f}\n"
        f"Intercept: {intercept:.3f}\n"
        f"$R^2$: {r_value**2:.3f}\n"
        f"Bound: $2^{{-{a:.2f}K - {b:.2f}}}$"
    )
    plt.text(0.05, 0.05, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             fontsize=10, verticalalignment='bottom', family='monospace')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Small default size for testing
    run_stationary_simplicity_bias(L=36, T=6, num_parents=50)
