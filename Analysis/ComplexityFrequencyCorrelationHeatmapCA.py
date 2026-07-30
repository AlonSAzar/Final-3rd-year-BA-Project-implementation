import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import sys
from scipy.stats import pearsonr, spearmanr
from collections import Counter

# Add the project root to sys.path to resolve imports properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.ComplexityMeasures.complexity import ZlibComplexity

def calculate_ca_sb_correlations(L, T, rules=None, num_seeds=20, piece_height=16):
    """
    Calculates the average Pearson and Spearman correlations between Complexity and Log-Frequency
    across all 256 rules.
    """
    engine = ElementaryCA(L=L, T=T)
    metric = ZlibComplexity()
    
    if rules is None:
        rules = range(256)

    rule_pearsons = []
    rule_spearmans = []
    
    for rule in rules:
        pieces = []
        for _ in range(num_seeds):
            seed = engine.generate_seed()
            full_history = engine.run(rule, seed)[1:] # Ignore seed row
            
            # Divide into pieces of height 16
            num_pieces = T // piece_height
            for i in range(num_pieces):
                piece = full_history[i * piece_height : (i + 1) * piece_height]
                pieces.append(piece.tobytes())
        
        # Count frequencies for this specific rule
        counts = Counter(pieces)
        unique_hashes = list(counts.keys())
        
        # Need variability to calculate correlation
        if len(unique_hashes) < 3:
            continue

        complexities = []
        log_freqs = []
        total = len(pieces)
        
        for h in unique_hashes:
            # Reconstruct piece from buffer
            # Note: np.frombuffer output size depends on the L value used when the piece was created
            img_flat = np.frombuffer(h, dtype=np.uint8)
            # Infer current L from the buffer size divided by piece_height
            current_L = len(img_flat) // piece_height
            img = img_flat.reshape(piece_height, current_L)
            comp = metric.calculate(img)
            prob = counts[h] / total
            complexities.append(comp)
            log_freqs.append(np.log10(prob))
            
        import warnings
        if np.std(complexities) > 1e-9 and np.std(log_freqs) > 1e-9:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_corr, _ = pearsonr(complexities, log_freqs)
                s_corr, _ = spearmanr(complexities, log_freqs)
            
            if not np.isnan(p_corr) and not np.isnan(s_corr):
                rule_pearsons.append(p_corr)
                rule_spearmans.append(s_corr)
    
    # Return average correlation across all valid rules
    avg_p = np.mean(rule_pearsons) if rule_pearsons else 0.0
    avg_s = np.mean(rule_spearmans) if rule_spearmans else 0.0
    
    return avg_p, avg_s

def main():
    # Grid sizes to test
    L_values = [16, 32, 64, 128, 256]
    T_values = [16, 32, 64, 128, 256]
    
    # Parameters for sampling
    num_seeds_per_rule = 5 
    piece_height = 16 # Height of the patterns we are correlating
    
    print(f"Generating CA Simplicity Bias Correlation Heatmaps...")
    print(f"Settings: Rules 0-255, {num_seeds_per_rule} seeds/rule, Piece Height: {piece_height}")

    pearson_matrix = np.zeros((len(T_values), len(L_values)))
    spearman_matrix = np.zeros((len(T_values), len(L_values)))

    for i, T in enumerate(tqdm(T_values, desc="T (Total History Length)")):
        for j, L in enumerate(L_values):
            p_val, s_val = calculate_ca_sb_correlations(L, T, num_seeds=num_seeds_per_rule, piece_height=piece_height)
            pearson_matrix[i, j] = p_val
            spearman_matrix[i, j] = s_val

    output_dir = "Analysis/Saved Figures"
    os.makedirs(output_dir, exist_ok=True)

    # Plot Pearson Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(pearson_matrix, annot=True, fmt=".3f", 
                xticklabels=L_values, yticklabels=T_values,
                cmap="coolwarm", center=0)
    plt.title(f"Pearson Corr: Complexity vs. log(Frequency)\n(CA Rules 0-255, Piece Height={piece_height})")
    plt.xlabel("Grid Width (L)")
    plt.ylabel("Total Time steps (T)")
    plt.savefig(os.path.join(output_dir, "CA_SB_Pearson_Heatmap.png"))
    
    # Plot Spearman Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(spearman_matrix, annot=True, fmt=".3f", 
                xticklabels=L_values, yticklabels=T_values,
                cmap="coolwarm", center=0)
    plt.title(f"Spearman Corr: Complexity vs. log(Frequency)\n(CA Rules 0-255, Piece Height={piece_height})")
    plt.xlabel("Grid Width (L)")
    plt.ylabel("Total Time steps (T)")
    plt.savefig(os.path.join(output_dir, "CA_SB_Spearman_Heatmap.png"))
    
    print(f"\nHeatmaps saved to {output_dir}")
    plt.show()

if __name__ == "__main__":
    main()
