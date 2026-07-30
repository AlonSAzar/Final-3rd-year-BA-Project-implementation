import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA

def compute_ncc(img1, img2):
    """Calculates Normalized Cross-Correlation."""
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    
    # Subtract mean
    img1 -= np.mean(img1)
    img2 -= np.mean(img2)
    
    # Standard deviations
    std1 = np.std(img1)
    std2 = np.std(img2)
    
    if std1 == 0 or std2 == 0:
        return 0.0
        
    # Correlation
    corr = np.mean(img1 * img2) / (std1 * std2)
    return corr

def test_rule30_convergence(sizes=[8, 16, 32, 64, 128, 256, 512, 1024, 2048], num_seeds=50):
    """
    Checks if Rule 30's NCC increase as L,T grows is just a statistical artifact 
    of density convergence (Law of Large Numbers) or actual structural similarity.
    """
    rule = 30
    
    actual_nccs = {size: [] for size in sizes}
    control_nccs = {size: [] for size in sizes} # Independent seeds (Control for LLN)

    print("Running Rule 30 NCC Convergence Test...")
    
    for size in sizes:
        engine = ElementaryCA(L=size, T=size)
        
        for _ in range(num_seeds):
            # 1. Actual Mutation Case (1-bit seed change)
            seed_x = engine.generate_seed()
            img_x = engine.run(rule, seed_x)[1:]
            
            mut_seed = seed_x.copy()
            mut_seed[np.random.randint(size)] ^= 1
            img_y = engine.run(rule, mut_seed)[1:]
            
            actual_nccs[size].append(compute_ncc(img_x, img_y))
            
            # 2. Control Case (Two completely independent random seeds)
            # If Rule 30 is just "pseudo-random balanced noise", 
            # two independent noisy images will have higher NCC as size increases 
            # JUST because their mean and stdev converge to the same global values (LLN).
            seed_z = engine.generate_seed()
            img_z = engine.run(rule, seed_z)[1:]
            
            control_nccs[size].append(compute_ncc(img_x, img_z))

    # Calculate means and errors
    avg_actual = [np.mean(actual_nccs[s]) for s in sizes]
    std_actual = [np.std(actual_nccs[s]) for s in sizes]
    
    avg_control = [np.mean(control_nccs[s]) for s in sizes]
    std_control = [np.std(control_nccs[s]) for s in sizes]

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(sizes, avg_actual, yerr=std_actual, fmt='-o', capsize=5, label='Mutant (1-bit change)', color='crimson')
    plt.errorbar(sizes, avg_control, yerr=std_control, fmt='-o', capsize=5, label='Control (Independent Seeds)', color='gray', linestyle='--')
    
    plt.xscale('log', base=2)
    plt.xticks(sizes, sizes)
    plt.xlabel('Lattice Size (L=T)')
    plt.ylabel('Normalized Cross-Correlation (NCC)')
    plt.title('Rule 30: Is higher NCC just a Law of Large Numbers artifact?\nMutant vs. Independent Random Control')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    
    # Interpretation text
    plt.figtext(0.15, 0.02, 
                "If Mutant NCC > Control NCC: Rule 30 retains structural memory of the seed.\n"
                "If Mutant NCC approx Control NCC: The 'Similarity' is just statistical noise converging to the mean (LLN).",
                ha="left", fontsize=9, bbox={"facecolor":"white", "alpha":0.8, "pad":5})

    output_path = "Analysis/GeneralScripts/rule30_ncc_lln_test.png"
    plt.savefig(output_path)
    print(f"Test complete. Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    test_rule30_convergence()
