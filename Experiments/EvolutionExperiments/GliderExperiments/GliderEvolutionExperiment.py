import numpy as np
import matplotlib.pyplot as plt
import zlib
from tqdm import tqdm
from copy import deepcopy


class GliderEvolutionExperiment:
    def __init__(self, engine, pop_size=100, mutation_rate=0.02, complexity_bonus=True):
        self.engine = engine
        self.L = engine.L
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.complexity_bonus = complexity_bonus

        # Results storage
        self.rule_fitness_map = {}  # {rule_id: max_fitness}
        self.best_phenotypes = {}  # {rule_id: (seed, history)}

    def _zlib_complexity(self, data: np.ndarray) -> int:
        return len(zlib.compress(data.tobytes()))

    def calculate_fitness(self, rule_id, genotype):
        """
        Multi-objective fitness:
        1. Survival (Not empty)
        2. Velocity (Right-most pixel)
        3. Economy (Low Density)
        4. Optional: Emergence (High Phenotype Complexity / Low Genotype Complexity)
        """
        history = self.engine.run(rule_id, genotype)
        final_row = history[-1]

        # 1. Survival Check
        if np.sum(final_row) == 0:
            return 0.0

        # 2. Velocity (Right-most active cell)
        # We assume the seed starts near the left or center, so higher index = movement
        active_indices = np.where(final_row == 1)[0]
        right_most_index = np.max(active_indices)

        # 3. Density Penalty (Avoid Chaos/Floods)
        density = np.mean(final_row)
        if density > 0.65:
            return 0.0  # Instant death for chaotic floods

        base_fitness = right_most_index

        # 4. Complexity Bonus (The "Efficient Replicator" Reward)
        # We want: Simple Seed -> Complex Structure
        if self.complexity_bonus:
            k_genotype = self._zlib_complexity(genotype)
            k_phenotype = self._zlib_complexity(history)

            # Ratio: How much complexity did the system generate from the seed?
            # Higher is better.
            emergence_ratio = k_phenotype / k_genotype

            # Scale it so it affects fitness meaningfully
            # (e.g. if ratio is 5.0, add 50 points)
            base_fitness += (emergence_ratio * 10)

        return max(0.0, base_fitness)

    def run_for_rule(self, rule_id, generations=50):
        """Runs the GA for a single rule."""
        # Initialize Population (Random Seeds)
        population = [self.engine.generate_seed() for _ in range(self.pop_size)]

        best_genotype = None
        max_fitness = -1

        for gen in range(generations):
            fitness_scores = []

            # Evaluate
            for individual in population:
                fit = self.calculate_fitness(rule_id, individual)
                fitness_scores.append(fit)

                if fit > max_fitness:
                    max_fitness = fit
                    best_genotype = individual.copy()

            # If everything died, restart with new randoms (rare but possible in Class 1)
            if max_fitness == 0 and gen < 5:
                population = [self.engine.generate_seed() for _ in range(self.pop_size)]
                continue
            elif max_fitness == 0:
                break  # Give up

            # Selection (Tournament)
            new_population = []
            fitness_scores = np.array(fitness_scores)

            # Elitism: Keep the absolute best
            new_population.append(best_genotype)

            while len(new_population) < self.pop_size:
                # Pick 2 random parents
                candidates = np.random.randint(0, self.pop_size, 2)
                p1, p2 = population[candidates[0]], population[candidates[1]]
                f1, f2 = fitness_scores[candidates[0]], fitness_scores[candidates[1]]

                # Winner takes all
                parent = p1 if f1 > f2 else p2

                # Mutation
                child = parent.copy()
                mask = np.random.random(self.L) < self.mutation_rate
                child[mask] = 1 - child[mask]  # Flip bits
                new_population.append(child)

            population = new_population

        return max_fitness, best_genotype

    def scan_all_rules(self, generations_per_rule=30):
        """Iterates 0-255 to find the best Glider/Replicator rules."""
        print(f"Scanning all 256 rules for 'Smart Gliders'...")

        for rule in tqdm(range(256)):
            max_fit, best_gene = self.run_for_rule(rule, generations_per_rule)

            self.rule_fitness_map[rule] = max_fit

            if max_fit > 0:
                # Save the history for visualization later
                history = self.engine.run(rule, best_gene)
                self.best_phenotypes[rule] = (best_gene, history)

    def plot_distribution(self):
        """Maps which rules are capable of this behavior."""
        fitnesses = list(self.rule_fitness_map.values())

        plt.figure(figsize=(10, 6))
        plt.bar(self.rule_fitness_map.keys(), fitnesses, color='teal')
        plt.xlabel("Rule ID")
        plt.ylabel("Max Fitness Achieved")
        plt.title("Distribution of Evolutionary Potential across CA Space")
        plt.grid(True, alpha=0.3)
        plt.show()

    def visualize_top_rules(self, top_n=5):
        """Shows the actual spacetime patterns of the winners."""
        # Sort rules by fitness
        sorted_rules = sorted(self.rule_fitness_map.items(), key=lambda x: x[1], reverse=True)
        top_rules = sorted_rules[:top_n]

        print(f"Top {top_n} Rules for Efficient Replication/Gliders:")

        for rule, fit in top_rules:
            if fit == 0: continue

            seed, history = self.best_phenotypes[rule]

            # Calculate metrics for title
            k_in = self._zlib_complexity(seed)
            k_out = self._zlib_complexity(history)

            plt.figure(figsize=(10, 4))
            plt.imshow(history, cmap='magma', interpolation='nearest')
            plt.title(f"Rule {rule} | Fitness: {fit:.1f}\nSeed Complexity: {k_in} -> Pheno Complexity: {k_out}")
            plt.xlabel("Space")
            plt.ylabel("Time")
            plt.show()