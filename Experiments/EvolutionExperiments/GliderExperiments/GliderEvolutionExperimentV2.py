import numpy as np
import matplotlib.pyplot as plt
import zlib
from tqdm import tqdm
from copy import deepcopy

# TODO it's pretty bad practice to have the same function names

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
        history = self.engine.run(rule_id, genotype)

        # 1. Survival Check (Must not be empty)
        if np.sum(history[-1]) == 0: return 0.0

        # 2. Density Check (Kill Chaos)
        # Gliders are usually sparse. Chaos is usually ~50% dense.
        # Strict penalty if density > 20%
        avg_density = np.mean(history)
        if avg_density > 0.20: return 0.0

        # 3. Coherence Check (The "Glider" Detector)
        # We check if row T is similar to row T-1 shifted by S

        max_coherence = 0
        best_velocity = 0

        # Check T=50 to T=100 (Stable phase)
        t_start = len(history) // 2
        t_end = len(history) - 1

        # We test shifts of -1 (left), 0 (still), +1 (right)
        for shift in [-1, 0, 1]:
            matches = 0
            total_checks = 0

            for t in range(t_start, t_end):
                row_t = history[t]
                row_next = history[t + 1]

                # Roll row_t by 'shift' and compare to row_next
                expected_next = np.roll(row_t, shift)

                # Hamming similarity
                similarity = np.mean(expected_next == row_next)
                matches += similarity
                total_checks += 1

            avg_coherence = matches / total_checks

            if avg_coherence > max_coherence:
                max_coherence = avg_coherence
                best_velocity = shift

        # 4. The "Conveyor Belt" Penalty (NEW)
        # Check if the glider is just the seed shifted
        # We compare the first row (seed) with the middle row (shifted back)

        mid_row = history[len(history) // 2]
        seed_shifted = np.roll(genotype, best_velocity * (len(history) // 2))

        # If the seed is identical to the glider, it's a boring Class 2 rule
        similarity_to_seed = np.mean(mid_row == seed_shifted)

        if similarity_to_seed > 0.9:
            return 0.0  # Kill the trivial shifters

        # 5. Complexity Multiplier
        # We want the glider to be MORE complex than the seed
        k_seed = self._zlib_complexity(genotype)
        k_glider = self._zlib_complexity(mid_row)

        emergence_factor = k_glider / (k_seed + 1)

        if best_velocity != 0:
            return max_coherence * 50 * emergence_factor  # Boost complex gliders
        else:
            return max_coherence * 5
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