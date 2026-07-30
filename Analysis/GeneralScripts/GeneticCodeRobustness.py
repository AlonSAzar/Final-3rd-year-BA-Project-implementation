import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
from collections import Counter

# Standard Genetic Code mapping from RNA codons to Amino Acids
# 64 codons -> 20 Amino Acids + '*' (Stop codon)
# This is mathematically equivalent to a CA with L=3, 4 states per site, yielding 64 'neighborhoods'
STANDARD_GENETIC_CODE = {
    'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
    'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M',
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
    
    'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
    'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    
    'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*',
    'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    
    'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W',
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}

BASES = ['U', 'C', 'A', 'G']
CODONS = [''.join(c) for c in itertools.product(BASES, repeat=3)]

def get_1_point_neighbors(codon):
    """Returns all 9 codons that are exactly 1 point mutation away from the given codon."""
    neighbors = []
    for i in range(3):
        for base in BASES:
            if base != codon[i]:
                new_codon = codon[:i] + base + codon[i+1:]
                neighbors.append(new_codon)
    return neighbors

def calculate_robustness(code_map):
    """
    Calculates the 'Robustness' of a genetic code map.
    Robustness = Fraction of 1-point mutations that are SYNONYMOUS 
                 (i.e., they map to the same amino acid).
    """
    total_mutations = 0
    synonymous_mutations = 0
    
    for codon in CODONS:
        amino_acid = code_map[codon]
        neighbors = get_1_point_neighbors(codon)
        
        for neighbor in neighbors:
            total_mutations += 1
            if code_map[neighbor] == amino_acid:
                synonymous_mutations += 1
                
    return synonymous_mutations / total_mutations

def generate_random_code(amino_acids_pool):
    """
    Generates a random genetic code by shuffling the amino acid assignments
    across the 64 codons. We use the original pool so the degeneracy 
    (number of codons per amino acid) remains exactly the same as the real code.
    """
    shuffled_aas = np.random.permutation(amino_acids_pool)
    return {c: a for c, a in zip(CODONS, shuffled_aas)}

def run_experiment(num_random_codes=100000):
    print("--- Running Genetic Code Robustness Experiment ---")
    
    # 1. Calculate Standard Genetic Code Robustness
    sgc_robustness = calculate_robustness(STANDARD_GENETIC_CODE)
    print(f"Standard Genetic Code Robustness: {sgc_robustness:.4f} "
          f"({sgc_robustness * 100:.2f}% of single-point mutations are completely neutral!)")
    
    # 2. Create the null model (random codes with same degeneracy)
    # Get all 64 output states as a list to shuffle
    amino_acids_pool = [STANDARD_GENETIC_CODE[c] for c in CODONS]
    
    print(f"Generating {num_random_codes} random genetic codes representing alternative evolutionary paths...")
    random_robustness_scores = []
    
    for _ in range(num_random_codes):
        random_code = generate_random_code(amino_acids_pool)
        random_robustness_scores.append(calculate_robustness(random_code))
        
    random_robustness_scores = np.array(random_robustness_scores)
    mean_random = np.mean(random_robustness_scores)
    std_random = np.std(random_robustness_scores)
    
    z_score = (sgc_robustness - mean_random) / std_random
    print(f"Mean Random Code Robustness: {mean_random:.4f}")
    print(f"Standard Genetic Code is {z_score:.2f} Standard Deviations above the random mean (p-value ~ 0)")
    
    # 3. Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(random_robustness_scores, bins=100, color='grey', alpha=0.7, density=True, label='Random Permuted Genetic Codes')
    
    # Vertical line for the actual code
    plt.axvline(sgc_robustness, color='red', linestyle='dashed', linewidth=3, 
                label=f'Standard Genetic Code (Z={z_score:.2f})')
    
    # Normal curve overlay (optional visual aid)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, max(xmax, sgc_robustness + 0.05), 200)
    p = (1 / (std_random * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_random) / std_random) ** 2)
    plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution Fit')

    plt.title('Evolutionary Robustness: Standard Genetic Code vs. Random Alternatives\n'
              '(Modeling DNA Translation as a 64-Condition Rule Map)')
    plt.xlabel('Robustness (Fraction of Synonymous 1-Point Mutations)')
    plt.ylabel('Density (Probability)')
    
    # Annotate stats
    stats_text = (
        f"Simulations: {num_random_codes}\n"
        f"SGC Robustness: {sgc_robustness:.3f}\n"
        f"Random Mean: {mean_random:.3f}\n"
        f"Z-Score: +{z_score:.1f}σ"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.70, 0.50, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='left', bbox=props)

    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Ensure directory exists before saving
    save_dir = os.path.join("Analysis", "GeneralScripts", "Saved Figures")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, "GeneticCodeRobustness.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    run_experiment()
