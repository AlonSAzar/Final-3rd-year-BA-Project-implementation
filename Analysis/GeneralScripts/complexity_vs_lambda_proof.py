import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import zlib
from numba import njit
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from sklearn.metrics import mutual_info_score
import os

# --- 1. The Core Engines (Optimized) ---
@njit
def simulate_ca(rule_lut, seed, T, sync_prob=1.0):
    L = len(seed)
    output = np.zeros((T, L), dtype=np.uint8)
    output[0] = seed
    for t in range(1, T):
        for i in range(L):
            left = output[t - 1][(i - 1) % L]
            center = output[t - 1][i]
            right = output[t - 1][(i + 1) % L]
            neighborhood = (left << 2) | (center << 1) | right
            if sync_prob == 1.0 or np.random.random() < sync_prob:
                output[t][i] = rule_lut[neighborhood]
            else:
                output[t][i] = center
    return output

def get_lut(rule):
    return np.array([(rule >> i) & 1 for i in range(7, -1, -1)], dtype=np.uint8)

def calculate_lambda(rule):
    lut = get_lut(rule)
    return np.sum(lut) / 8.0

# --- 2. Data Collection ---
def collect_experiment_data(L=256, T=256, sync_prob=0.99, num_seeds=20):
    print(f"Collecting data for statistical proof (L={L}, T={T}, Sync Prob: {sync_prob})...")
    
    results = []
    for rule in tqdm(range(256), desc="Rules"):
        lut = get_lut(rule)
        lam = calculate_lambda(rule)
        
        rule_robustness = []
        rule_complexities = []
        
        for _ in range(num_seeds):
            seed = np.random.randint(0, 2, size=L, dtype=np.uint8)
            
            # Control (Perfect Clock)
            sync_history = simulate_ca(lut, seed, T, sync_prob=1.0)
            # Experiment (Broken Clock)
            async_history = simulate_ca(lut, seed, T, sync_prob=sync_prob)
            
            # Structural Robustness (Hamming Distance)
            # 1.0 = identical histories, 0.0 = completely different
            hamming_dist = np.mean(sync_history[1:] != async_history[1:])
            robustness = 1.0 - hamming_dist
            
            # Complexity calculated on synchronous history
            sync_k = len(zlib.compress(sync_history[1:].tobytes()))
            
            rule_robustness.append(robustness)
            rule_complexities.append(sync_k)
            
        results.append({
            'Rule': rule,
            'Lambda': lam,
            'Complexity': np.mean(rule_complexities),
            'Robustness': np.mean(rule_robustness)
        })
    
    return pd.DataFrame(results)

# --- 3. Statistical Analysis ---
def perform_statistical_proof(df):
    print("\n--- STATISTICAL PROOF: Complexity vs Langton's Lambda ---")
    
    # 1. Filter out stationary/dead rules (Complexity too low or high means nothing)
    # This focuses analysis on "Complex" rules where transitions happen
    df_complex = df[(df['Complexity'] > df['Complexity'].quantile(0.1)) & 
                    (df['Complexity'] < df['Complexity'].quantile(0.95))].copy()
    
    # 2. Linearize Lambda: Use Distance from 0.5 (Captures the U-shape)
    df_complex['Lambda_Dist_05'] = np.abs(df_complex['Lambda'] - 0.5)

    # Model A: Robustness ~ |Lambda - 0.5|
    X_lam = sm.add_constant(df_complex['Lambda_Dist_05'])
    model_lam = sm.OLS(df_complex['Robustness'], X_lam).fit()
    
    # Model B: Robustness ~ Complexity
    X_comp = sm.add_constant(df_complex['Complexity'])
    model_comp = sm.OLS(df_complex['Robustness'], X_comp).fit()
    
    # 3. Spearman Correlation
    s_lam, _ = spearmanr(df_complex['Lambda_Dist_05'], df_complex['Robustness'])
    s_comp, _ = spearmanr(df_complex['Complexity'], df_complex['Robustness'])

    print(f"Dataset Size (Filtered): {len(df_complex)} / 256 rules")
    print(f"Explained Variance (Linearized Lambda R^2): {model_lam.rsquared:.4f}")
    print(f"Explained Variance (Complexity R^2):        {model_comp.rsquared:.4f}")
    print("-" * 40)
    print(f"Spearman Corr (|Lambda-0.5| vs R): {s_lam:.4f}")
    print(f"Spearman Corr (Complexity vs R):   {s_comp:.4f}")
    
    # 4. Joint Model - Information Gain
    X_both = sm.add_constant(df_complex[['Lambda_Dist_05', 'Complexity']])
    model_both = sm.OLS(df_complex['Robustness'], X_both).fit()
    
    comp_p_value = model_both.pvalues['Complexity']
    print(f"P-value for Complexity in joint model: {comp_p_value:.4e}")
    
    # --- Plotting --- (Using the filtered complex set)
    plt.figure(figsize=(14, 6))

    # Plot 1: Lambda vs Robustness
    plt.subplot(1, 2, 1)
    plt.scatter(df_complex['Lambda'], df_complex['Robustness'], alpha=0.6, c='seagreen', edgecolor='k')
    plt.title(f"Lambda vs Robustness ($R^2$={model_lam.rsquared:.3f})")
    plt.xlabel("Langton's Lambda")
    plt.ylabel("Robustness (1 - Hamming)")

    # Plot 2: Complexity vs Robustness
    plt.subplot(1, 2, 2)
    plt.scatter(df_complex['Complexity'], df_complex['Robustness'], alpha=0.6, c='royalblue', edgecolor='k')
    plt.title(f"Complexity vs Robustness ($R^2$={model_comp.rsquared:.3f})")
    plt.xlabel("Phenotype Complexity")
    plt.ylabel("Robustness (1 - Hamming)")

    plt.tight_layout()
    save_path = os.path.join("Analysis", "GeneralScripts", "Saved Figures", "statistical_proof_refined.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Refined proof plots saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    df = collect_experiment_data(L=256, T=256, sync_prob=0.99, num_seeds=15)
    perform_statistical_proof(df)
