import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from collections import Counter
from scipy.stats import linregress
import os

from Core.ComplexityMeasures.complexity import ZlibComplexity
from Core.ComplexityMeasures.conditional_complexity import ZlibConditionalComplexity

class GameOfLife:
    def __init__(self, size=(64, 64)):
        self.size = size
        self.kernel = np.array([[1, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]])

    def step(self, state):
        neighbors = convolve2d(state, self.kernel, mode='same', boundary='wrap')
        new_state = np.zeros_like(state)
        # Rule: 
        # 1. Any live cell with 2 or 3 neighbors survives.
        # 2. Any dead cell with exactly 3 neighbors becomes a live cell.
        new_state[(state == 1) & ((neighbors == 2) | (neighbors == 3))] = 1
        new_state[(state == 0) & (neighbors == 3)] = 1
        return new_state

    def run(self, initial_state, steps=100):
        history = [initial_state]
        current = initial_state
        for _ in range(steps):
            current = self.step(current)
            history.append(current)
        return np.array(history)

def analyze_simplicity_bias(engine, metric, samples=500, L=64, T=64, subset_size=(16, 16)):
    """
    Global Simplicity Bias: Correlation between complexity and probability of occurrence 
    of patterns sampled from random initial conditions.
    """
    print(f"Sampling {samples} random initial conditions...")
    patterns = []
    
    for _ in range(samples):
        # Random initial state
        init = np.random.randint(0, 2, (L, T), dtype=np.uint8)
        # Run for a few steps to get "natural" GOL patterns, or just use patches from history
        history = engine.run(init, steps=10) # Run for 10 steps to stabilize
        final_state = history[-1]
        
        # Sample random sub-patches
        for _ in range(5): # 5 patches per run
            r = np.random.randint(0, L - subset_size[0])
            c = np.random.randint(0, T - subset_size[1])
            patch = final_state[r:r+subset_size[0], c:c+subset_size[1]]
            patterns.append(patch)

    # Count frequencies
    hashes = [p.tobytes() for p in patterns]
    counts = Counter(hashes)
    
    unique_hashes = list(counts.keys())
    complexities = []
    log_freqs = []
    
    for h in unique_hashes:
        idx = hashes.index(h)
        comp = metric.calculate(patterns[idx])
        complexities.append(comp)
        log_freqs.append(np.log(counts[h] / len(patterns)))

    slope, intercept, r_value, p_value, std_err = linregress(complexities, log_freqs)
    
    return complexities, log_freqs, slope, r_value**2, p_value

def analyze_conditional_simplicity_bias(engine, metric, samples=200, steps=20, L=64, T=64):
    """
    Conditional Simplicity Bias: Given an initial state A', 
    what is the conditional complexity K(B|A') vs its probability P(B|A')?
    """
    print("Analyzing Conditional Simplicity Bias...")
    
    # We use a conditional metric specifically
    cond_metric = ZlibConditionalComplexity()
    
    all_complexities = []
    all_log_probs = []

    for _ in range(samples):
        # Start from a "natural" GOL state
        base_state = np.random.randint(0, 2, (L, T), dtype=np.uint8)
        for _ in range(20): base_state = engine.step(base_state)
        
        # Perturb and observe transitions
        transitions = []
        perturbed_states = []
        for _ in range(100): # 100 perturbations
            perturbed = base_state.copy()
            # Flip 1% of bits
            idx = np.random.choice(L*T, size=int(0.01*L*T), replace=False)
            perturbed.flat[idx] = 1 - perturbed.flat[idx]
            
            perturbed_states.append(perturbed)
            next_state = engine.step(perturbed)
            transitions.append(next_state.tobytes())
            
        counts = Counter(transitions)
        t_total = len(transitions)
        
        for h, count in counts.items():
            # Find an example B and its corresponding A' that led to it
            idx = transitions.index(h)
            img_b = np.frombuffer(h, dtype=np.uint8).reshape(L, T)
            img_ap = perturbed_states[idx]
            
            # K(B|A') = K(B, A') - K(A')
            comp = cond_metric.calculate(img_ap, img_b)
            prob = count / t_total
            
            all_complexities.append(comp)
            all_log_probs.append(np.log(prob))

    slope, intercept, r_value, p_value, std_err = linregress(all_complexities, all_log_probs)
    return all_complexities, all_log_probs, slope, r_value**2, p_value

def main():
    L, T = 64, 64
    gol = GameOfLife(size=(L, T))
    metric = ZlibComplexity()
    
    # 1. Global Simplicity Bias
    comps, logs, slope, r2, p = analyze_simplicity_bias(gol, metric, L=L, T=T)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(comps, logs, alpha=0.5)
    plt.title(f"Global Simplicity Bias (GOL)\nSlope: {slope:.3f}, R2: {r2:.3f}")
    plt.xlabel("Complexity")
    plt.ylabel("log(Probability)")
    
    # 2. Conditional Simplicity Bias
    c_comps, c_logs, c_slope, c_r2, c_p = analyze_conditional_simplicity_bias(gol, metric, L=L, T=T)
    
    plt.subplot(1, 2, 2)
    plt.scatter(c_comps, c_logs, alpha=0.3, color='orange')
    plt.title(f"Conditional Simplicity Bias (GOL)\nSlope: {c_slope:.3f}, R2: {c_r2:.3f}")
    plt.xlabel("Complexity of Next State")
    plt.ylabel("log(P(B|A'))")
    
    plt.tight_layout()
    os.makedirs("Analysis/Saved Figures", exist_ok=True)
    plt.savefig("Analysis/Saved Figures/gol_simplicity_bias.png")
    print("Results saved to Analysis/Saved Figures/gol_simplicity_bias.png")
    plt.show()

if __name__ == "__main__":
    main()
