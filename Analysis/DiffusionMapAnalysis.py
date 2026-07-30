import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import scipy.linalg as la
from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.ComplexityMeasures.complexity import ZlibComplexity
from tqdm import tqdm

class DiffusionMapAnalyzer:
    """
    Implements Diffusion Maps for Phenotype Space Analysis.
    Diffusion maps reveal the underlying geometry of the phenotype manifold
    by looking at the connectivity (diffusion process) between phenotypes.
    """
    
    def __init__(self, rule, L=64, T_sim=64):
        self.rule = rule
        self.L = L
        self.T_sim = T_sim
        self.engine = ElementaryCA(L, T_sim)
        self.metric = ZlibComplexity()

    def collect_phenotypes(self, num_samples=1000):
        """Generates a dataset of phenotypes from random seeds."""
        print(f"Collecting {num_samples} phenotypes for Rule {self.rule}...")
        phenos = []
        complexities = []
        hashes = set()
        
        for _ in tqdm(range(num_samples)):
            seed = self.engine.generate_seed()
            img = self.engine.run(self.rule, seed)
            # Use last row as the representative state for faster processing
            state = img[-1:].astype(np.float32)
            h = state.tobytes()
            
            if h not in hashes:
                hashes.add(h)
                phenos.append(state.flatten())
                complexities.append(self.metric.calculate(img))
                
        return np.array(phenos), np.array(complexities)

    def compute_diffusion_map(self, data, sigma=None, n_components=3, alpha=0.5):
        """
        Computes the diffusion coordinates for the given data.
        sigma: Kernel width (auto-calculated if None)
        alpha: Re-normalization parameter (0.5 for Fokker-Planck diffusion, 1.0 for Laplace-Beltrami)
        """
        print("Computing Diffusion Map...")
        # 1. Compute Pairwise Distance Matrix (Hamming distance is natural for bits)
        dist_sq = pairwise_distances(data, metric='hamming') ** 2
        
        # 2. Kernel Matrix (Gaussian)
        if sigma is None:
            sigma = np.median(dist_sq) * 0.5
        
        K = np.exp(-dist_sq / sigma)
        
        # 3. Density Normalization (Re-weighting to ignore sampling density)
        d = np.sum(K, axis=1)
        D_inv_alpha = np.diag(1.0 / (d**alpha))
        K_alpha = D_inv_alpha @ K @ D_inv_alpha
        
        # 4. Markov Transition Matrix M
        d_alpha = np.sum(K_alpha, axis=1)
        M = np.diag(1.0 / d_alpha) @ K_alpha
        
        # 5. Eigen-decomposition
        evals, evecs = la.eig(M)
        
        # Sort by eigenvalue magnitude
        idx = np.argsort(np.abs(evals))[::-1]
        evals = np.real(evals[idx])
        evecs = np.real(evecs[:, idx])
        
        # 6. Map to Diffusion Space
        # Diffusion coordinates Psi_t = lambda^t * eigenvector
        # We skip the first constant eigenvector (v0 = 1)
        diffusion_coords = evecs[:, 1:n_components+1] * evals[1:n_components+1]
        
        return diffusion_coords, evals[1:]

    def plot_diffusion_space(self, coords, complexities, evals):
        """Plots the phenotypes in the first two diffusion coordinates."""
        plt.figure(figsize=(10, 7))
        sc = plt.scatter(coords[:, 0], coords[:, 1], c=complexities, cmap='viridis', 
                        alpha=0.7, edgecolors='none', s=20)
        plt.colorbar(sc, label='Zlib Complexity')
        
        plt.xlabel(f"Diffusion Coord 1 (λ₁ = {evals[0]:.4f})")
        plt.ylabel(f"Diffusion Coord 2 (λ₂ = {evals[1]:.4f})")
        plt.title(f"Diffusion Map of Rule {self.rule} Phenotypes\nGeometry revealed by $P(y|x)$ connectivity")
        plt.grid(True, alpha=0.1)
        plt.show()

def run_diffusion_analysis(rule=110, samples=800):
    analyzer = DiffusionMapAnalyzer(rule=rule)
    data, complexities = analyzer.collect_phenotypes(num_samples=samples)
    
    if len(data) < 10:
        print("Not enough unique phenotypes found.")
        return
        
    coords, evals = analyzer.compute_diffusion_map(data)
    analyzer.plot_diffusion_space(coords, complexities, evals)

if __name__ == "__main__":
    # Test on Rule 110 (The Edge of Chaos)
    run_diffusion_analysis(rule=110)
