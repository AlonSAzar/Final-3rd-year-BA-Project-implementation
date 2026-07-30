import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import sys
import os

# Add the project root to sys.path to allow imports from Core and Experiments
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.ComplexityMeasures.complexity import ZlibComplexity
from Core.ComplexityMeasures.conditional_complexity import ZlibConditionalComplexity

def get_avg_complexity(engine, metric, rule, num_seeds=20):
    """
    Calculates Average Phenotype Complexity (K).
    """
    ks = []
    for _ in range(num_seeds):
        img = engine.run(rule)[1:] # Skip first row
        ks.append(metric.calculate(img))
        
    return np.mean(ks)

def analyze_targeted_lambda_sb(L=8, T=8, num_seeds=500):
    """
    Analyzes only rules with Lambda = 0.375 or 0.625 (Distance 0.125 from 0.5).
    Calculates SB and CSB correlations.
    """
    target_lambdas = [0.375, 0.625]
    targeted_rules = []
    
    for rule in range(256):
        lam = bin(rule).count('1') / 8.0
        if lam in target_lambdas:
            targeted_rules.append(rule)
            
    print(f"Targeting {len(targeted_rules)} rules with λ ∈ {target_lambdas}")
    
    engine = ElementaryCA(L=L, T=T)
    metric = ZlibComplexity()
    cond_metric = ZlibConditionalComplexity()
    
    results = {
        'rule': [],
        'avg_complexity': [],
        'sb_correlation': [],
        'conditional_sb_correlation': []
    }
    
    for rule in tqdm(targeted_rules, desc="Targeted Analysis"):
        avg_k = get_avg_complexity(engine, metric, rule, num_seeds=30)

        # --- Global SB ---
        counts = {}
        ks = {}
        for _ in range(num_seeds):
            img = engine.run(rule)[1:]
            h = img.tobytes()
            if h not in counts:
                counts[h] = 0
                ks[h] = metric.calculate(img)
            counts[h] += 1
        
        probs = np.array(list(counts.values())) / num_seeds
        unique_ks = np.array(list(ks.values()))
        sb_corr = spearmanr(unique_ks, np.log(probs))[0] if len(probs) > 1 and np.var(unique_ks) > 0 else 0

        # --- Conditional SB ---
        t_counts = {}
        t_ks = {}
        for _ in range(num_seeds):
            seed = engine.generate_seed()
            img_x = engine.run(rule, seed)[1:]
            mut_seed = seed.copy()
            mut_seed[np.random.randint(len(seed))] ^= 1
            img_y = engine.run(rule, mut_seed)[1:]
            
            h_trans = (img_x.tobytes(), img_y.tobytes())
            if h_trans not in t_counts:
                t_counts[h_trans] = 0
                t_ks[h_trans] = cond_metric.calculate(img_y, img_x)
            t_counts[h_trans] += 1

        c_probs = np.array(list(t_counts.values())) / num_seeds
        c_ks = np.array(list(t_ks.values()))
        c_sb_corr = spearmanr(c_ks, np.log(c_probs))[0] if len(c_probs) > 1 and np.var(c_ks) > 0 else 0

        results['rule'].append(rule)
        results['avg_complexity'].append(avg_k)
        results['sb_correlation'].append(sb_corr)
        results['conditional_sb_correlation'].append(c_sb_corr)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Global SB
    ax1.scatter(results['avg_complexity'], results['sb_correlation'], alpha=0.7, c='teal', edgecolors='k')
    ax1.set_title(f"Global SB (λ ∈ {target_lambdas})")
    ax1.set_xlabel("Avg K")
    ax1.set_ylabel("Spearman ρ")
    ax1.grid(True, alpha=0.3)
    
    # Conditional SB
    ax2.scatter(results['avg_complexity'], results['conditional_sb_correlation'], alpha=0.7, c='crimson', edgecolors='k')
    ax2.set_title(f"Conditional SB (λ ∈ {target_lambdas})")
    ax2.set_xlabel("Avg K")
    ax2.set_ylabel("Spearman ρ")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Simplicity Bias Analysis for Lambda-Targeted Class (L={L}, T={T})")
    plt.tight_layout()
    
    output = "Analysis/GeneralScripts/lambda_0375_targeted_sb.png"
    plt.savefig(output)
    print(f"Plot saved to {output}")
    plt.show()

if __name__ == "__main__":
    analyze_targeted_lambda_sb(L=8, T=8, num_seeds=500)
