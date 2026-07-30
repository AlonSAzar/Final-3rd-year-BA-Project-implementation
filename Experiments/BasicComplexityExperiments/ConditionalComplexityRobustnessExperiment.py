import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import os

from Experiments.experiments import BaseExperiment, compute_ncc
from Core.ComplexityMeasures.complexity import ComplexityMetric
from Core.ComplexityMeasures.conditional_complexity import ZlibConditionalComplexity

class ConditionalComplexityRobustnessExperiment(BaseExperiment):
    """
    Tests the relationship between Phenotype Complexity and Conditional Complexity K(mutant | base).
    Higher K(mutant | base) means the mutation led to a more 'surprising' or different phenotype.
    Lower K(mutant | base) indicates high robustness (the mutant is easily described given the base).
    """

    def __init__(self, engine, metric: ComplexityMetric, cond_metric=None, random_iterations: int = 20):
        super().__init__(engine, metric)
        self.cond_metric = cond_metric if cond_metric else ZlibConditionalComplexity()
        self.random_iterations = random_iterations

    def run(self, strategy, num_seeds=10, mut_engine=None):
        """
        Similar to RobustnessExperiment, but computes K(mut_img | base_img).
        """
        if mut_engine is None:
            mut_engine = self.engine

        print(f"--- Conditional Complexity Robustness: {strategy.name()} ---")
        
        results = [] # List of (base_complexity, avg_cond_complexity)

        seeds = [self.engine.generate_seed() for _ in range(num_seeds)]

        for rule in tqdm(range(256), desc="Rules"):
            for seed in seeds:
                base_img = self.engine.run(rule, seed)[1:]
                base_k = self.metric.calculate(base_img)

                num_vars = strategy.get_variations_count(self.engine, rule, seed, self.random_iterations)
                cond_complexities = []

                for i in range(num_vars):
                    m_rule, m_seed = strategy.apply(self.engine, rule, seed, i)
                    mut_img = mut_engine.run(m_rule, m_seed)[1:]

                    # Calculate K(mut | base)
                    k_cond = self.cond_metric.calculate(base_img, mut_img)
                    cond_complexities.append(k_cond)
                
                if cond_complexities:
                    avg_cond_k = np.mean(cond_complexities)
                    results.append((base_k, avg_cond_k))

        self.results = np.array(results)
        return self.results

    def plot(self, save_path="Analysis/Saved Figures/conditional_complexity_robustness.png"):
        if not hasattr(self, 'results'):
            print("No results to plot. Run the experiment first.")
            return

        base_ks = self.results[:, 0]
        cond_ks = self.results[:, 1]

        # Correlations
        p_corr, p_val = pearsonr(base_ks, cond_ks)
        s_corr, s_val = spearmanr(base_ks, cond_ks)

        plt.figure(figsize=(10, 6))
        plt.scatter(base_ks, cond_ks, alpha=0.4, color='teal', s=10)
        
        plt.title(f"Complexity vs. Mutant Surprise (Conditional Complexity)\nPearson: {p_corr:.3f}, Spearman: {s_corr:.3f}")
        plt.xlabel("Base Phenotype Complexity K(x)")
        plt.ylabel("Avg Conditional Complexity K(mutant | base)")
        plt.grid(True, alpha=0.3)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        plt.show()

    def analyze(self):
        """Standard analyze implementation for template method."""
        raw_data = self.results
        return raw_data

if __name__ == "__main__":
    from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
    from Core.ComplexityMeasures.complexity import ZlibComplexity
    from Core.strategies import BitFlipRuleStrategy

    # Default settings for a quick run from file
    L, T = 100, 100
    engine = ElementaryCA(L=L, T=T)
    metric = ZlibComplexity()
    
    # Initialize the experiment
    exp = ConditionalComplexityRobustnessExperiment(engine, metric, random_iterations=8)
    
    # Use Rule Bit-Flip strategy as default
    strategy = BitFlipRuleStrategy()
    
    # Run a subset of rules for speed if executed directly
    print("Executing experiment directly...")
    exp.run(strategy, num_seeds=100)
    exp.plot()
