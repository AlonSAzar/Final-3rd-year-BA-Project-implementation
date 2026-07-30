import numpy as np
import matplotlib.pyplot as plt
import zlib
from tqdm import tqdm


class EvolutionaryDynamicsAnalyzer:
    def __init__(self, experiment_instance):
        """
        experiment_instance: An instance of GliderEvolutionExperiment
                             (must have .engine and .calculate_fitness)
        """
        self.exp = experiment_instance
        self.engine = experiment_instance.engine

    def _get_complexity(self, data):
        return len(zlib.compress(data.tobytes()))

    def run_tracking(self, rule_id, generations=50):
        """
        Runs evolution for one rule, tracking metrics per generation.
        Returns: Dict of history lists
        """
        # Initialize Population
        population = [self.engine.generate_seed() for _ in range(self.exp.pop_size)]

        # History Logs
        history = {
            'gen': [],
            'max_fitness': [],
            'avg_fitness': [],
            'best_seed_k': [],
            'best_pheno_k': []
        }

        best_genotype_overall = None
        max_fitness_overall = -1

        for gen in range(generations):
            # 1. Evaluate Population
            fitness_scores = []
            best_in_gen_idx = 0
            max_fit_in_gen = -1

            for i, individual in enumerate(population):
                fit = self.exp.calculate_fitness(rule_id, individual)
                fitness_scores.append(fit)

                if fit > max_fit_in_gen:
                    max_fit_in_gen = fit
                    best_in_gen_idx = i

                # Keep track of global best
                if fit > max_fitness_overall:
                    max_fitness_overall = fit
                    best_genotype_overall = individual.copy()

            # 2. Record Metrics for the Best Individual of this Generation
            best_ind = population[best_in_gen_idx]

            # Calculate Complexities
            seed_k = self._get_complexity(best_ind)
            # Run simulation to get phenotype complexity
            pheno_hist = self.engine.run(rule_id, best_ind)
            pheno_k = self._get_complexity(pheno_hist)

            history['gen'].append(gen)
            history['max_fitness'].append(max_fit_in_gen)
            history['avg_fitness'].append(np.mean(fitness_scores))
            history['best_seed_k'].append(seed_k)
            history['best_pheno_k'].append(pheno_k)

            # 3. Selection & Mutation (Standard GA Logic)
            new_population = []
            new_population.append(best_genotype_overall)  # Elitism

            fitness_scores = np.array(fitness_scores)

            # Simple Tournament Selection
            while len(new_population) < self.exp.pop_size:
                candidates = np.random.randint(0, self.exp.pop_size, 2)
                idx1, idx2 = candidates
                parent = population[idx1] if fitness_scores[idx1] > fitness_scores[idx2] else population[idx2]

                child = parent.copy()
                mask = np.random.random(self.exp.L) < self.exp.mutation_rate
                child[mask] = 1 - child[mask]
                new_population.append(child)

            population = new_population

        return history

    def compare_top_rules(self, top_rules_list, generations=50):
        """
        Runs the tracking for a list of specific rules and plots them side-by-side.
        top_rules_list: list of integers [169, 73, 192, ...]
        """
        results = {}

        print(f"Tracking dynamics for rules: {top_rules_list}")
        for rule in tqdm(top_rules_list):
            results[rule] = self.run_tracking(rule, generations)

        # --- Plotting ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Fitness Trajectory
        ax = axes[0]
        for rule, data in results.items():
            ax.plot(data['gen'], data['max_fitness'], label=f"Rule {rule}", linewidth=2)
        ax.set_title("Evolution of Fitness")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Max Fitness")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Seed Complexity (Genotypic Entropy)
        ax = axes[1]
        for rule, data in results.items():
            ax.plot(data['gen'], data['best_seed_k'], label=f"Rule {rule}", linestyle='--')
        ax.set_title("Genotype Complexity (Seed)")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Zlib Complexity (Bytes)")
        ax.grid(True, alpha=0.3)

        # Plot 3: Phenotype Complexity (Emergence)
        ax = axes[2]
        for rule, data in results.items():
            ax.plot(data['gen'], data['best_pheno_k'], label=f"Rule {rule}")
        ax.set_title("Phenotype Complexity (Structure)")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Zlib Complexity (Bytes)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()