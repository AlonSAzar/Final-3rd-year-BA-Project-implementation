import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from tqdm import tqdm
from Core.ComplexityMeasures.complexity import ZlibComplexity

def plot_reweighted_spectral_gap(experiment, temps=None, num_subsamples=10, subsample_fraction=0.8):
    """
    Takes a ran ConditionalTransitionExperiment and re-weights its transitions
    using a 'Virtual Temperature' (Selection Pressure).
    Now with Confidence Intervals calculated by Bootstrapping (subsampling transitions).
    """
    print(f"Performing Virtual Temperature Re-weighting with {num_subsamples} Bootstrap trials...")
    all_transitions = list(experiment.return_transitions().items())
    phenotype_cache = experiment.return_phenotype_cache()
    
    if not all_transitions:
        print("No transitions found.")
        return

    # Pre-calculate unique hashes and fitnesses across ALL data for consistent indexing
    unique_hashes = set()
    for (h1, h2), _ in all_transitions:
        unique_hashes.add(h1); unique_hashes.add(h2)
    h_list = sorted(list(unique_hashes))
    h_to_id = {h: i for i, h in enumerate(h_list)}
    N = len(h_list)
    
    metric = ZlibComplexity()
    fitness_cache = {h: -metric.calculate(phenotype_cache[h]) for h in h_list}

    if temps is None:
        temps = np.logspace(-3, 3, 20)

    # Matrix to store gaps per trial
    gap_trials = np.zeros((len(temps), num_subsamples))
    
    for trial in range(num_subsamples):
        # 1. Subsample transitions to estimate variance
        indices = np.random.choice(len(all_transitions), 
                                   size=int(len(all_transitions) * subsample_fraction), 
                                   replace=True)
        trial_transitions = [all_transitions[i] for i in indices]
        
        for t_idx, T in enumerate(tqdm(temps, desc=f"Trial {trial+1}/{num_subsamples}", leave=False)):
            M = np.zeros((N, N))
            
            for (h_x, h_y), count in trial_transitions:
                i, j = h_to_id[h_x], h_to_id[h_y]
                f_x, f_y = fitness_cache[h_x], fitness_cache[h_y]
                
                delta_f = f_y - f_x
                acceptance = 1.0 if delta_f >= 0 else np.exp(delta_f / T)
                
                weight = count * acceptance
                M[i, j] += weight
                M[i, i] += count * (1.0 - acceptance)

            row_sums = M.sum(axis=1, keepdims=True)
            M_prob = np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums != 0)
            
            try:
                evals = la.eigvals(M_prob)
                abs_evals = np.sort(np.abs(evals))[::-1]
                lambda_2 = abs_evals[1] if len(abs_evals) > 1 else 1.0
                gap_trials[t_idx, trial] = 1.0 - lambda_2
            except:
                gap_trials[t_idx, trial] = 0.0

    # Calculate statistics
    means = np.mean(gap_trials, axis=1)
    # Standard Error of the Mean (SEM)
    sems = np.std(gap_trials, axis=1) / np.sqrt(num_subsamples)

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Area for 95% Confidence Interval
    plt.fill_between(temps, means - 1.96*sems, means + 1.96*sems, color='royalblue', alpha=0.2, label='95% CI (Bootstrap)')
    # Main line
    plt.plot(temps, means, marker='o', linewidth=2, color='royalblue', label='Average Gap')
    
    plt.xscale('log')
    plt.xlabel("Virtual Temperature (Selection Pressure)")
    plt.ylabel("Spectral Gap (1 - |λ₂|)")
    plt.title("Spectral Gap vs. Virtual Temperature\n(Bootstrap Estimates from Transition Data)")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    # min_idx = np.argmin(means)
    # plt.scatter(temps[min_idx], means[min_idx], color='red', s=100, zorder=5, label='Edge of Chaos (Min Gap)')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

from collections import Counter
if __name__ == "__main__":
    print("This file contains the re-weighting logic for Spectral Gap analysis.")
