import numpy as np
import matplotlib.pyplot as plt
import zlib
from tqdm import tqdm
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA

def coarse_grain_phenotype(image, block_size=2):
    """
    Applies coarse-graining by averaging blocks of size (block_size x block_size)
    and thresholding to binary. This simulates the Israeli & Goldenfeld (2006) 
    approach where complex CA (like Rule 30) can map to simpler CA behaviors 
    at higher scales.
    """
    T, L = image.shape
    new_T = T // block_size
    new_L = L // block_size
    
    coarse_img = np.zeros((new_T, new_L), dtype=np.uint8)
    
    for t in range(new_T):
        for l in range(new_L):
            # Take a block
            block = image[t*block_size:(t+1)*block_size, l*block_size:(l+1)*block_size]
            # Average and threshold (majority rule)
            if np.mean(block) > 0.5:
                coarse_img[t, l] = 1
            else:
                coarse_img[t, l] = 0
                
    return coarse_img

def calculate_complexity(image):
    """Simple Zlib-based complexity."""
    return len(zlib.compress(image.tobytes(), level=9))

def run_israeli_goldenfeld_experiment(L=128, T=128, num_seeds=1000, block_sizes=[1, 2, 4, 8]):
    """
    Tests Simplicity Bias at different scales using Coarse-Graining for Rule 30.
    """
    engine = ElementaryCA(L=L, T=T)
    rule = 30
    
    results = {bs: [] for bs in block_sizes}
    
    print(f"Running Rule 30 Coarse-Graining Experiment (N={num_seeds} seeds)...")
    
    for _ in tqdm(range(num_seeds)):
        seed = engine.generate_seed(seed_type="random")
        full_img = engine.run(rule, seed)
        
        for bs in block_sizes:
            if bs == 1:
                phenotype = full_img
            else:
                phenotype = coarse_grain_phenotype(full_img, block_size=bs)
            
            raw_k = calculate_complexity(phenotype)
            # Normalize by the number of pixels to get Compression Ratio (Complexity per pixel)
            normalized_k = raw_k / phenotype.size if phenotype.size > 0 else 0
            results[bs].append(normalized_k)
            
    return results

def plot_israeli_goldenfeld_results(results):
    """
    Plots the complexity distributions at different scales.
    """
    plt.figure(figsize=(12, 8))
    
    # Generate a list of colors based on the number of scales
    num_scales = len(results)
    cmap = plt.get_cmap('viridis')
    colors = [cmap(val) for val in np.linspace(0, 0.8, num_scales)]
    
    for i, (bs, complexities) in enumerate(results.items()):
        # Calculate frequencies for the histogram
        unique_ks, counts = np.unique(complexities, return_counts=True)
        probs = counts / np.sum(counts)
        
        # We plot prob vs K in log-linear to see the simplicity bias shift
        plt.scatter(unique_ks, np.log10(probs), alpha=0.6, label=f"Scale {bs}x{bs}", c=colors[i])
        
        # Fit a simple line to the "bound" (Simplicity Bias)
        if len(unique_ks) > 1:
            # Find the upper bound by taking max prob for each K
            z = np.polyfit(unique_ks, np.log10(probs), 1)
            p = np.poly1d(z)
            plt.plot(unique_ks, p(unique_ks), color=colors[i], linestyle='--', alpha=0.4)

    plt.title("Simplicity Bias in Coarse-Grained Rule 30\nFollowing Israeli & Goldenfeld (2006)")
    plt.xlabel("Normalized Complexity K (Compression Ratio - Bytes / Pixel)")
    plt.ylabel("Log10 Probability")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # Save the figure
    save_dir = os.path.join("Analysis", "GeneralScripts", "Saved Figures")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    filename = "israeli_goldenfeld_coarse_grained_rule30.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # Parameters
    L_VAL = 64
    T_VAL = 256
    SEEDS = 2000 # Adjust for speed/accuracy
    SCALES = [1, 2, 4, 8, 16] # Higher scales require larger T/L
    
    results = run_israeli_goldenfeld_experiment(L=L_VAL, T=T_VAL, num_seeds=SEEDS, block_sizes=SCALES)
    plot_israeli_goldenfeld_results(results)
