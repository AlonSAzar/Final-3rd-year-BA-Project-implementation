import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from tqdm import tqdm
from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA

class SpectralThermodynamicsExperiment:
    """
    Analyzes the Relaxation Time (Spectral Gap) of the Phenotype Landscape 
    under different Selection Temperatures (T).
    
    Prediction: 
    - Low T (Frozen/Simple): Large Gap, Fast Relaxation.
    - Edge of Chaos: Minimal Gap (Phase Transition), Critical Slowing Down.
    - High T (Random Noise): Large Gap, Instant Equilibrium (Uniform).
    """
    
    def __init__(self, rule, L=64, T_sim=64):
        self.rule = rule
        self.L = L
        self.T_sim = T_sim
        self.engine = ElementaryCA(L, T_sim)

    def get_phenotype_hash(self, seed):
        """Returns the hash of the resulting phenotype."""
        # Using last row for state mapping to keep state space manageable
        history = self.engine.run(self.rule, seed)
        return history[-1:].tobytes()

    def run_metropolis_transitions(self, temp, n_mutations=1000):
        """
        Record transitions P(x -> y) using Metropolis-Hastings at a given Temp.
        Fitness = Negative Complexity (Simplicity Bias) or any other metric.
        """
        from Core.ComplexityMeasures.complexity import ZlibComplexity
        metric = ZlibComplexity()
        
        current_seed = self.engine.generate_seed()
        current_pheno_h = self.get_phenotype_hash(current_seed)
        current_fit = -metric.calculate(self.engine.run(self.rule, current_seed)) # Fitness is Simplicity
        
        transitions = {} # (h_x, h_y) -> count
        pheno_cache = {current_pheno_h: current_fit}
        
        for _ in range(n_mutations):
            # 1. Mutate (1-bit flip)
            bit = np.random.randint(self.L)
            next_seed = current_seed.copy()
            next_seed[bit] = 1 - next_seed[bit]
            
            # 2. Get Phenotype and Fitness
            next_pheno_h = self.get_phenotype_hash(next_seed)
            if next_pheno_h not in pheno_cache:
                next_fit = -metric.calculate(self.engine.run(self.rule, next_seed))
                pheno_cache[next_pheno_h] = next_fit
            else:
                next_fit = pheno_cache[next_pheno_h]
            
            # 3. Metropolis Acceptance
            delta = next_fit - current_fit
            if delta >= 0:
                accept = True
            else:
                if temp <= 0:
                    accept = False
                else:
                    prob = np.exp(delta / temp)
                    accept = np.random.random() < prob
            
            # 4. Record Transition
            # Even if we reject, we count a transition to ourselves (staying in state x)
            target_h = next_pheno_h if accept else current_pheno_h
            pair = (current_pheno_h, target_h)
            transitions[pair] = transitions.get(pair, 0) + 1
            
            if accept:
                current_seed = next_seed
                current_pheno_h = next_pheno_h
                current_fit = next_fit
                
        return transitions

    def calculate_spectral_gap(self, transitions):
        """Builds matrix and extracts the spectral gap."""
        unique_hashes = set()
        for h1, h2 in transitions.keys():
            unique_hashes.add(h1); unique_hashes.add(h2)
        
        h_list = list(unique_hashes)
        h_to_id = {h: i for i, h in enumerate(h_list)}
        N = len(h_list)
        
        if N < 2: return 0.0
        
        M = np.zeros((N, N))
        for (h1, h2), count in transitions.items():
            M[h_to_id[h1], h_to_id[h2]] = count
            
        # Normalize rows
        row_sums = M.sum(axis=1, keepdims=True)
        M_prob = np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums != 0)
        
        try:
            evals = la.eigvals(M_prob)
            abs_evals = np.sort(np.abs(evals))[::-1]
            # Gap = 1 - |lambda_2|
            # Note: We filter for the first lambda that is NOT 1 (or very close)
            # if multiple components exist, the gap for the whole matrix is 0
            # we focus on the largest component or just the sorted diff
            lambda_2 = abs_evals[1] if len(abs_evals) > 1 else 1.0
            return 1.0 - lambda_2
        except:
            return 0.0

def run_spectral_temp_scan(rule=110, trials=50):
    # Logarithmic temperature scale
    temps = [0.001, 0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.4, 0.8, 1.0, 10.0, 100.0, 1000.0]
    exp = SpectralThermodynamicsExperiment(rule=rule, L=32, T_sim=32)
    
    all_gaps = []
    
    print(f"Scanning Phenotype Spectral Gaps for Rule {rule} with {trials} trials per T...")
    for T in temps:
        trial_gaps = []
        for _ in range(trials):
            transitions = exp.run_metropolis_transitions(T, n_mutations=2000)
            gap = exp.calculate_spectral_gap(transitions)
            trial_gaps.append(gap)
        all_gaps.append(trial_gaps)
        
    all_gaps = np.array(all_gaps)
    means = np.mean(all_gaps, axis=1)
    sems = np.std(all_gaps, axis=1) / np.sqrt(trials)
    
    plt.figure(figsize=(10, 6))
    
    # Error bars for 95% Confidence Interval (1.96 * SEM)
    plt.errorbar(temps, means, yerr=1.96*sems, fmt='o-', color='darkorange', 
                 ecolor='red', capsize=4, elinewidth=1, label='Gap ± 95% CI')
    
    plt.xscale('log')
    plt.xlabel("Temperature (Mutation Selection Pressure)")
    plt.ylabel("Spectral Gap (1 - |λ₂|)")
    plt.title(f"Structural Integrity vs. Temperature (Rule {rule})\nWith 95% Confidence Intervals")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.show()
    plt.xlabel("Temperature (Mutation Selection Pressure)")
    plt.ylabel("Spectral Gap (1 - |λ₂|)")
    plt.title(f"Structural Integrity vs. Temperature (Rule {rule})\nLow Gap = Critical Slowing Down (Edge of Chaos)")
    plt.grid(True, which="both", alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_spectral_temp_scan(110)
