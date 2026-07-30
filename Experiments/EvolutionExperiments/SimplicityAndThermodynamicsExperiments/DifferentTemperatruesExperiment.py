import numpy as np
import matplotlib.pyplot as plt
import zlib
from tqdm import tqdm
from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA


class TemperatureExperiment:
    def __init__(self, rule, seed, L=128, max_steps=1000):
        self.rule = rule
        self.L = L
        self.max_steps = max_steps
        self.engine = ElementaryCA(L, max_steps)
        self.seed = seed

    def measure_fitness(self, genotype):
        """Fitness = Lifetime (Transient Length)."""
        history = self.engine.run(self.rule, genotype)
        seen_states = {}

        for t, row in enumerate(history):
            h = row.tobytes()
            if np.sum(row) == 0:
                return 0  # Death
            if h in seen_states:
                return seen_states[h]  # Cycle
            seen_states[h] = t

        return self.max_steps  # Pseudo-infinite

    def mutate(self, genotype, mutation_rate=0.03):
        child = genotype.copy()
        mask = np.random.random(self.L) < mutation_rate
        child[mask] = 1 - child[mask]
        return child

    def run_at_temperature(self, temp, generations=300):
        current_geno = self.seed.copy()
        current_fit = self.measure_fitness(current_geno)

        fits = []
        seed_complexities = []

        for _ in range(generations):
            candidate = self.mutate(current_geno)
            cand_fit = self.measure_fitness(candidate)

            if cand_fit >= current_fit:
                accept = True
            else:
                if temp <= 0:
                    prob = 0
                else:
                    delta = cand_fit - current_fit
                    prob = np.exp(delta / temp)
                accept = np.random.random() < prob

            if accept:
                current_geno = candidate
                current_fit = cand_fit

            fits.append(current_fit)
            seed_k = len(zlib.compress(current_geno.tobytes()))
            seed_complexities.append(seed_k)

        return np.mean(fits), np.max(fits), np.mean(seed_complexities)


def scan_temperatures(rule=110, n_seeds=15):
    temps = [0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0]

    avg_fitnesses = []
    max_fitnesses = []
    avg_seed_k = []

    print(f"Scanning Temperatures for Rule {rule} (averaged over {n_seeds} seeds)...")

    base_engine = ElementaryCA(L=128, T=1000)

    for T in tqdm(temps):
        seed_avg_fit = []
        seed_max_fit = []
        seed_avg_k = []

        # Generate 15 independent seeds
        seeds = [base_engine.generate_seed() for _ in range(n_seeds)]

        for seed in seeds:
            exp = TemperatureExperiment(rule, seed)
            avg_f, max_f, avg_k = exp.run_at_temperature(T)
            seed_avg_fit.append(avg_f)
            seed_max_fit.append(max_f)
            seed_avg_k.append(avg_k)

        avg_fitnesses.append(np.mean(seed_avg_fit))
        max_fitnesses.append(np.mean(seed_max_fit))
        avg_seed_k.append(np.mean(seed_avg_k))

    # ---- Plotting ----
    fig, ax1 = plt.subplots(figsize=(10, 6))

    plot_temps = [t if t > 0 else 0.01 for t in temps]

    color = 'tab:red'
    ax1.set_xlabel('Temperature (Log Scale)')
    ax1.set_ylabel('Average Fitness (Lifetime)', color=color, fontweight='bold')
    ax1.plot(plot_temps, avg_fitnesses, color=color, marker='o', label="Avg Fitness")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')

    best_T_idx = np.argmax(avg_fitnesses)
    ax1.axvline(plot_temps[best_T_idx], color='grey', linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Seed Complexity (Zlib Bytes)', color=color, fontweight='bold')
    ax2.plot(plot_temps, avg_seed_k, color=color, marker='s', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(
        f"Evolutionary Edge of Chaos (Rule {rule})\nAveraged over {n_seeds} initial seeds"
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    scan_temperatures(110, n_seeds=15)
