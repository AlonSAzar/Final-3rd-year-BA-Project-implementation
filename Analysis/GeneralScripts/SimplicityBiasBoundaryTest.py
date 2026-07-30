import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from scipy.stats import linregress
from collections import Counter

# Add the project root to sys.path to resolve imports properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.ComplexityMeasures.complexity import ZlibComplexity

def check_simplicity_bias_boundary(L=128, T=256, piece_height=8):
    """
    Runs ALL 256 rules, aggregates all unique 8x128 pieces found across all rules,
    and plots Log(Global Frequency) vs Complexity to check for the theoretical 
    boundary: P(x) <= 2^-K(x).
    """
    engine = ElementaryCA(L=L, T=T)
    metric = ZlibComplexity()
    
    global_piece_counts = Counter()
    piece_to_complexity = {}

    print(f"Sampling pieces from 256 rules (L={L}, T={T})...")
    for rule in tqdm(range(256), desc="Rules"):
        seed = engine.generate_seed()
        full_history = engine.run(rule, seed)[1:] # Ignore seed row
        
        # Divide into pieces
        num_pieces = T // piece_height
        for i in range(num_pieces):
            piece = full_history[i * piece_height : (i + 1) * piece_height]
            h = piece.tobytes()
            
            # Increment global count
            global_piece_counts[h] += 1
            
            # Cache complexity if new
            if h not in piece_to_complexity:
                piece_to_complexity[h] = metric.calculate(piece)

    # Prepare data for plotting
    unique_hashes = list(global_piece_counts.keys())
    complexities = [piece_to_complexity[h] for h in unique_hashes]
    
    total_samples = sum(global_piece_counts.values())
    # Frequencies (Probability P)
    probabilities = [global_piece_counts[h] / total_samples for h in unique_hashes]
    log_probabilities = [np.log2(p) for p in probabilities]

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.scatter(complexities, log_probabilities, alpha=0.5, s=15, color='royalblue', label='Observed Phenotypes')
    
    # The Theoretical "Simplicity Bias" Boundary: P(x) = 2^-K(x)
    # In log2 space, this is log2(P) = -K
    # We calibrate K slightly because Zlib complexity reflects K but isn't exactly K bits.
    # We can draw y = -x + C to see the slope.
    K_range = np.linspace(min(complexities), max(complexities), 100)
    
    # Calculate a simple linear fit for the top edge (boundary)
    # We use a quantile regression or just a bounding line for visualization
    plt.plot(K_range, -K_range + max(log_probabilities + K_range), color='red', linestyle='--', linewidth=2, label='Theoretical Boundary (Slope -1)')

    plt.title("Checking the Simplicity Bias Boundary (All Rules Aggregated)\nGlobal Phenotype Probability vs. Complexity")
    plt.xlabel("Complexity K(x) [Zlib]")
    plt.ylabel("Log2(Probability) log2(P(x))")
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_dir = "Saved Figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    save_path = os.path.join(output_dir, "Simplicity_Bias_Global_Boundary.png")
    plt.savefig(save_path)
    print(f"\nBoundary Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    check_simplicity_bias_boundary()