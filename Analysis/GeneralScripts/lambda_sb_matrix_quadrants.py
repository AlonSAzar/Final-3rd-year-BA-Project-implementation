import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm import tqdm
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.ComplexityMeasures.complexity import ZlibComplexity
from Core.ComplexityMeasures.conditional_complexity import ZlibConditionalComplexity

def get_lambda(rule_int):
    binary = [int(x) for x in bin(rule_int)[2:].zfill(8)]
    return sum(binary) / 8.0

def analyze_sb_proportions_by_lambda(L=8, T=8, num_seeds=500, threshold=-0.2):
    """
    Categorizes rules into a 2x2 matrix:
    - Global SB (Yes/No)
    - Conditional SB (Yes/No)
    And shows the Langton's Lambda distribution for each slot.
    """
    engine = ElementaryCA(L=L, T=T)
    metric = ZlibComplexity()
    cond_metric = ZlibConditionalComplexity()

    # Matrix: [Global SB Yes/No][Cond SB Yes/No]
    # Rows: 0=No Global SB, 1=Yes Global SB
    # Cols: 0=No Cond SB, 1=Yes Cond SB
    matrix_rules = [[[] for _ in range(2)] for _ in range(2)]

    print(f"Categorizing rules (L={L}, T={T}, threshold={threshold})...")
    
    for rule in tqdm(range(256), desc="Rules"):
        lam = get_lambda(rule)
        
        # --- Global SB ---
        pheno_counts = {}
        ks = {}
        for _ in range(num_seeds):
            img = engine.run(rule)[1:]
            h = img.tobytes()
            if h not in pheno_counts:
                pheno_counts[h] = 0
                ks[h] = metric.calculate(img)
            pheno_counts[h] += 1
        
        probs = np.array(list(pheno_counts.values())) / num_seeds
        unique_ks = np.array(list(ks.values()))
        
        sb_exists = False
        if len(probs) > 1 and np.var(unique_ks) > 0:
            corr, _ = spearmanr(unique_ks, np.log(probs))
            if corr < threshold: # Significant negative correlation
                sb_exists = True

        # --- Conditional SB ---
        trans_counts = {}
        trans_ks = {}
        for _ in range(num_seeds):
            seed = engine.generate_seed()
            img_x = engine.run(rule, seed)[1:]
            
            mut_seed = seed.copy()
            mut_seed[np.random.randint(len(seed))] ^= 1
            img_y = engine.run(rule, mut_seed)[1:]
            
            h = (img_x.tobytes(), img_y.tobytes())
            if h not in trans_counts:
                trans_counts[h] = 0
                trans_ks[h] = cond_metric.calculate(img_y, img_x)
            trans_counts[h] += 1
            
        c_probs = np.array(list(trans_counts.values())) / num_seeds
        c_ks = np.array(list(trans_ks.values()))
        
        csb_exists = False
        if len(c_probs) > 1 and np.var(c_ks) > 0:
            c_corr, _ = spearmanr(c_ks, np.log(c_probs))
            if c_corr < threshold:
                csb_exists = True

        row = 1 if sb_exists else 0
        col = 1 if csb_exists else 0
        matrix_rules[row][col].append(lam)

    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    titles = [
        ["No Global SB, No Cond SB", "No Global SB, Yes Cond SB"],
        ["Yes Global SB, No Cond SB", "Yes Global SB, Yes Cond SB"]
    ]
    colors = [['lightgray', 'salmon'], ['lightblue', 'lightgreen']]

    for r in range(2):
        for c in range(2):
            ax = axes[r][c]
            lambdas = matrix_rules[r][c]
            count = len(lambdas)
            
            if count > 0:
                ax.hist(lambdas, bins=np.linspace(0, 1, 10), color=colors[r][c], edgecolor='black', alpha=0.7)
                avg_l = np.mean(lambdas)
                ax.axvline(avg_l, color='red', linestyle='--', label=f'Avg λ={avg_l:.2f}')
                ax.legend()
            
            ax.set_title(f"{titles[r][c]}\n(N={count} rules)")
            ax.set_xlim(-0.05, 1.05)
            ax.set_xlabel("λ")
            ax.set_ylabel("Count")

    plt.suptitle(f"Langton's λ Distribution in SB/CSB Quadrants\n(Threshold ρ < {threshold}, L={L}, T={T})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = "Analysis/GeneralScripts/lambda_sb_matrix_quadrants.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    analyze_sb_proportions_by_lambda(L=8, T=8, num_seeds=500)
