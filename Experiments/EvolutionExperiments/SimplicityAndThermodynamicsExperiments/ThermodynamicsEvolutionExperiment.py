import numpy as np
import matplotlib.pyplot as plt
from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
import zlib


class MetropolisEvolution:
    def __init__(self, rule, L=256, max_steps=1000):
        self.rule = rule
        self.L = L
        self.max_steps = max_steps
        self.engine = ElementaryCA(L, max_steps)

    def measure_fitness(self, genotype):
        """
        Fitness = Transient Length (Lifetime).
        We cap 'Infinite' (-1) at max_steps for numerical stability.
        """
        history = self.engine.run(self.rule, genotype)
        seen_states = {}

        for t, row in enumerate(history):
            h = row.tobytes()

            # Case 1: Extinction (Zero fitness)
            if np.sum(row) == 0:
                return 0

            # Case 2: Cycle Found
            if h in seen_states:
                return seen_states[h]  # Transient length

            seen_states[h] = t

        # Case 3: "Infinite" / Complex
        return self.max_steps

    def mutate(self, genotype, rate_left, rate_right):
        child = genotype.copy()
        mid = self.L // 2

        # Left mutation
        mask_l = np.random.random(mid) < rate_left
        child[:mid][mask_l] = 1 - child[:mid][mask_l]

        # Right mutation
        mask_r = np.random.random(self.L - mid) < rate_right
        child[mid:][mask_r] = 1 - child[mid:][mask_r]

        return child

    def run_metropolis(self, generations=1000, rates=(0.01, 0.05), temperature=10.0):
        # 1. Initialize
        current_geno = self.engine.generate_seed()
        current_fit = self.measure_fitness(current_geno)

        history_fit = []
        history_complexity = []  # To track "Simplicity Bias"

        for i in range(generations):
            # 2. Mutate
            rate_l, rate_r = rates
            candidate = self.mutate(current_geno, rate_l, rate_r)
            candidate_fit = self.measure_fitness(candidate)

            # 3. Metropolis Acceptance
            if candidate_fit >= current_fit:
                # Always accept better/equal
                accept = True
            else:
                # Probabilistically accept worse (The "Knock Off")
                delta = candidate_fit - current_fit
                prob = np.exp(delta / temperature)
                accept = np.random.random() < prob

            if accept:
                current_geno = candidate
                current_fit = candidate_fit

            # 4. Log Data
            history_fit.append(current_fit)
            # Log complexity of the genotype (Seed)
            # Hypothesis: When fitness drops, Complexity drops (falling into simple basin)
            c_bytes = len(zlib.compress(current_geno.tobytes()))
            history_complexity.append(c_bytes)

        return history_fit, history_complexity


def visualize_arrival_of_frequent(rule=110):
    # Setup
    # L=100 so cycles aren't instant
    model = MetropolisEvolution(rule, L=128, max_steps=500)

    # Run simulation
    # Temperature=20: Allows dropping ~20 steps of lifetime fairly easily
    print("Running Metropolis Evolution...")
    fits, comps = model.run_metropolis(generations=200, rates=(0.01, 0.05), temperature=20.0)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness (Lifetime)', color=color)
    ax1.plot(fits, color=color, alpha=0.6, label="Fitness")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title(
        f"Dynamic Evolution (Metropolis): Rule {rule}\nNote the 'Collapse' events where system falls off a peak")

    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Genotype Complexity (Zlib)', color=color)  # we already handled the x-label with ax1
    ax2.plot(comps, color=color, alpha=0.6, linestyle='--', label="Complexity")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


if __name__ == "__main__":
    visualize_arrival_of_frequent(110)