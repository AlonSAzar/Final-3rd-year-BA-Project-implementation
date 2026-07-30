import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

# Import your existing engine
# from matrix_map import MatrixMapEngine  # Or ElementaryCA

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA


class EvolutionaryExperiment:
    def __init__(self, engine, pop_size=1000, mutation_rate=0.2):
        self.engine = engine
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate  # Probability per bit
        self.L = engine.L

    def find_targets(self, n_samples=10000):
        """
        Pre-run: Samples the map to find a 'Frequent' and a 'Rare' phenotype.
        """
        print("Sampling map to define fitness landscape...")
        counts = Counter()
        cache = {}

        # Random sampling
        for _ in range(n_samples):
            g = self.engine.generate_seed()
            p = self.engine.run(0, g)  # rule_id 0 for matrix
            h = p.tobytes()
            counts[h] += 1
            cache[h] = p

        # Sort by frequency
        sorted_phenos = counts.most_common()

        # Target A: The most frequent (The "Trap")
        self.target_A_hash = sorted_phenos[0][0]
        self.target_A_img = cache[self.target_A_hash]
        freq_A = sorted_phenos[0][1]

        # Target B: A rare one (The "Global Optima")
        # We pick one that appeared just once or twice
        self.target_B_hash = sorted_phenos[-1][0]
        self.target_B_img = cache[self.target_B_hash]
        freq_B = sorted_phenos[-1][1]

        print(f"Target A (Trap):   Freq={freq_A}/{n_samples} (Fitness=1.99)")
        print(f"Target B (Optimum): Freq={freq_B}/{n_samples} (Fitness=2.0)")
        return self.target_A_img, self.target_B_img

    def calculate_fitness(self, phenotype_hash):
        """
        The Two-Peak Landscape
        """
        if phenotype_hash == self.target_B_hash:
            return 2.0  # Global Maximum (Rare)
        elif phenotype_hash == self.target_A_hash:
            return 1.99  # Local Maximum (Frequent)
        else:
            return 0.1  # Background/Dead

    def run_evolution(self, generations=100):
        # 1. Initialize Population (Random Genotypes)
        population = [self.engine.generate_seed() for _ in range(self.pop_size)]

        history_A = []
        history_B = []
        avg_fitness = []

        for gen in tqdm(range(generations), desc="Evolving"):
            # A. Evaluate Fitness
            fitness_scores = []
            count_A = 0
            count_B = 0

            # Cache phenotypes for next generation selection
            phenotypes = []

            for genotype in population:
                pheno = self.engine.run(0, genotype)
                h = pheno.tobytes()

                fit = self.calculate_fitness(h)
                fitness_scores.append(fit)
                phenotypes.append(h)

                if h == self.target_A_hash: count_A += 1
                if h == self.target_B_hash: count_B += 1

            # Record stats
            history_A.append(count_A / self.pop_size)
            history_B.append(count_B / self.pop_size)
            avg_fitness.append(np.mean(fitness_scores))

            # B. Selection (Roulette Wheel)
            fitness_scores = np.array(fitness_scores)
            total_fit = np.sum(fitness_scores)
            if total_fit == 0:
                probs = np.ones(self.pop_size) / self.pop_size
            else:
                probs = fitness_scores / total_fit

            # Select indices of parents for next generation
            parent_indices = np.random.choice(
                np.arange(self.pop_size),
                size=self.pop_size,
                p=probs
            )
            parents = [population[i] for i in parent_indices]

            # C. Reproduction with Mutation
            new_population = []
            for parent in parents:
                child = parent.copy()
                # Apply mutation to each bit
                mask = np.random.random(self.L) < self.mutation_rate
                child[mask] = 1 - child[mask]  # Flip bits
                new_population.append(child)

            population = new_population

        return history_A, history_B, avg_fitness


def main():
    # Use Matrix Map (L=15) or CA (L=10)
    # Smaller L is better to ensure Target B is actually reachable
    eng = ElementaryCA(L=10, T=64)

    evo = EvolutionaryExperiment(eng, pop_size=200, mutation_rate=0.1)

    # 1. Setup Landscape
    img_A, img_B = evo.find_targets()

    # 2. Run
    hist_A, hist_B, fit = evo.run_evolution(generations=500)

    # 3. Plot
    plt.figure(figsize=(10, 5))
    plt.plot(hist_A, label=f"Freq Phenotype (Fit=1.99)", color='orange', linewidth=2)
    plt.plot(hist_B, label=f"Rare Phenotype (Fit=2.0)", color='blue', linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Population Fraction")
    plt.title("Arrival of the Frequent: Bias vs Selection")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()