import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA

def compute_ncc(img1, img2):
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    img1 -= np.mean(img1)
    img2 -= np.mean(img2)
    std1 = np.std(img1)
    std2 = np.std(img2)
    if std1 == 0 or std2 == 0:
        return 0.0
    return np.mean(img1 * img2) / (std1 * std2)

def coarse_grain(seed, block_size=8):
    """
    Coarse-grain a binary seed by block averaging (majority rule).
    Returns a new binary array of the same length, but with repeated blocks.
    """
    L = len(seed)
    n_blocks = L // block_size
    cg = np.zeros(n_blocks, dtype=int)
    for i in range(n_blocks):
        block = seed[i*block_size:(i+1)*block_size]
        cg[i] = 1 if np.sum(block) > (block_size // 2) else 0
    # Expand back to original length by repeating
    return np.repeat(cg, block_size)

def random_seed_with_coarse_grain(target_cg, block_size=8):
    """
    Generate a random binary seed with the same coarse-grained pattern as target_cg,
    but with random bits within each block.
    """
    n_blocks = len(target_cg) // block_size
    seed = np.zeros(len(target_cg), dtype=int)
    for i in range(n_blocks):
        block_val = target_cg[i*block_size]
        # Randomly permute bits, but majority matches block_val
        n_ones = block_size//2 + 1 if block_val == 1 else block_size//2 - 1
        block = np.zeros(block_size, dtype=int)
        block[:n_ones] = block_val
        np.random.shuffle(block)
        seed[i*block_size:(i+1)*block_size] = block
    return seed

def test_rule30_coarse_grain_ncc(L=65, T=65, num_seeds=500, block_size=5):
    rule = 30
    engine = ElementaryCA(L=L, T=T)
    nccs_cg = []
    nccs_control = []
    print(f"Testing Rule 30 NCC for coarse-grained-similar but bitwise-different seeds (L={L}, T={T})...")
    for _ in tqdm(range(num_seeds)):
        # 1. Generate a random seed and its coarse-grained version
        seed1 = engine.generate_seed()
        cg1 = coarse_grain(seed1, block_size)
        # 2. Generate a different seed with the same coarse-grained pattern
        seed2 = random_seed_with_coarse_grain(cg1, block_size)
        # 3. Run both through Rule 30
        img1 = engine.run(rule, seed1)[1:]
        img2 = engine.run(rule, seed2)[1:]
        
        # Calculate and print NCC between original seeds
        seed_ncc = compute_ncc(seed1.reshape(1, -1), seed2.reshape(1, -1))
        print(f"Seed NCC (CG Similar): {seed_ncc:.4f}")
        
        nccs_cg.append(compute_ncc(img1, img2))
        # 4. Control: two completely random seeds
        seed3 = engine.generate_seed()
        
        # Calculate and print NCC between original and random seed
        control_seed_ncc = compute_ncc(seed1.reshape(1, -1), seed3.reshape(1, -1))
        print(f"Seed NCC (Random):     {control_seed_ncc:.4f}")
        
        img3 = engine.run(rule, seed3)[1:]
        nccs_control.append(compute_ncc(img1, img3))
    # --- Plotting ---
    plt.figure(figsize=(8, 6))
    plt.hist(nccs_cg, bins=20, alpha=0.7, label='Coarse-Grained Similar', color='royalblue')
    plt.hist(nccs_control, bins=20, alpha=0.7, label='Control (Random)', color='gray')
    plt.xlabel('Normalized Cross-Correlation (NCC)')
    plt.ylabel('Frequency')
    plt.title(f'Rule 30: NCC for Coarse-Grained Similar vs Random Seeds\nL={L}, T={T}, block_size={block_size}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_path = "Analysis/GeneralScripts/rule30_coarse_grain_ncc.png"
    plt.savefig(output_path)
    print(f"Test complete. Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    test_rule30_coarse_grain_ncc()
