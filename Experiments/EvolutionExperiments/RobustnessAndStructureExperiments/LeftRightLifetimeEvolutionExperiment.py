import numpy as np
import matplotlib.pyplot as plt
from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA


class LifetimeEvolution:
    def __init__(self, rule, L=64, max_steps=1000):
        self.rule = rule
        self.L = L
        self.max_steps = max_steps
        self.engine = ElementaryCA(L, max_steps)

    def measure_transient_length(self, genotype):
        history = self.engine.run(self.rule, genotype)
        seen_states = {}

        for t, row in enumerate(history):
            h = row.tobytes()
            if h in seen_states:
                return seen_states[h]  # Found a cycle! Return start time

            # If we hit an empty board, that's a short life (Bad)
            if np.sum(row) == 0:
                return t

            seen_states[h] = t

        # FIX: If we run out of time, that's GOOD! It means complex behavior.
        # Return a score higher than any cycle found within the limit.
        return self.max_steps + 100
    def mutate_split(self, genotype, rate_left, rate_right):
        """
        Applies different mutation rates to the two halves of the seed.
        """
        child = genotype.copy()
        mid = self.L // 2

        # Left Half Mutation
        mask_left = np.random.random(mid) < rate_left
        child[:mid][mask_left] = 1 - child[:mid][mask_left]

        # Right Half Mutation
        mask_right = np.random.random(self.L - mid) < rate_right
        child[mid:][mask_right] = 1 - child[mid:][mask_right]

        return child

    def run_evolution(self, generations=100, rates=(0, 0.0005)):
        """
        Wolfram-style 1+1 Evolution (Hill Climber).
        rates: (rate_left, rate_right)
        """
        rate_l, rate_r = rates

        # Start with random seed
        current_geno = self.engine.generate_seed()
        current_lifetime = self.measure_transient_length(current_geno)

        # If we started with "Infinite" (Chaos), retry until we get finite
        while current_lifetime == -1:
            current_geno = self.engine.generate_seed()
            current_lifetime = self.measure_transient_length(current_geno)

        history_lifetime = []
        hamming_left = []
        hamming_right = []

        original_geno = current_geno.copy()

        for g in range(generations):
            # 1. Mutate
            candidate = self.mutate_split(current_geno, rate_l, rate_r)

            # 2. Measure
            candidate_lifetime = self.measure_transient_length(candidate)

            # 3. Selection (Wolfram's Rule)
            # Accept if:
            # a. Not Infinite (-1)
            # b. Lifetime is >= Current Lifetime
            if candidate_lifetime != -1 and candidate_lifetime >= current_lifetime:
                # Accept
                current_geno = candidate
                current_lifetime = candidate_lifetime

            # 4. Record Metrics
            history_lifetime.append(current_lifetime)

            # Calculate distance from ORIGINAL ancestor to see where drift happened
            diff = (current_geno != original_geno)
            mid = self.L // 2
            hamming_left.append(np.sum(diff[:mid]))
            hamming_right.append(np.sum(diff[mid:]))

        return history_lifetime, hamming_left, hamming_right, current_geno


def compare_mutation_strategies(rule=110, L=128, gens=2000):
    exp = LifetimeEvolution(rule, L, max_steps=L * 8)

    print(f"Running Control (Uniform Mutation) for Rule {rule}...")
    # Control: Uniform 5% mutation
    hist_c, hamm_l_c, hamm_r_c, best_c = exp.run_evolution(gens, rates=(0.01, 0.01))

    print(f"Running Experiment (Split Mutation) for Rule {rule}...")
    # Exp: Left=1% (Conserved), Right=10% (Hypervariable)
    hist_e, hamm_l_e, hamm_r_e, best_e = exp.run_evolution(gens, rates=(0.001, 0.01))

    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Lifetime Growth
    ax = axes[0, 0]
    ax.plot(hist_c, label="Control (Uniform)", alpha=0.7)
    ax.plot(hist_e, label="Experiment (Split)", alpha=0.7)
    ax.set_title("Evolution of Lifetime (Transient Length)")
    ax.set_ylabel("Lifetime (Steps)")
    ax.legend()

    # 2. Hamming Distance (Accumulated Change)
    ax = axes[0, 1]
    ax.plot(hamm_l_e, label="Exp: Left Half (Low Mut)", linestyle="--")
    ax.plot(hamm_r_e, label="Exp: Right Half (High Mut)")
    ax.set_title("Genotypic Drift (Experiment)")
    ax.set_ylabel("Bits changed from Ancestor")
    ax.legend()

    # 3. Visualizing the Best Phenotypes
    # Re-run to get the image
    img_c = exp.engine.run(rule, best_c)
    img_e = exp.engine.run(rule, best_e)

    axes[1, 0].imshow(img_c, cmap='magma')
    axes[1, 0].set_title(f"Control Best (Lifetime: {hist_c[-1]})")

    axes[1, 1].imshow(img_e, cmap='magma')
    axes[1, 1].set_title(f"Experiment Best (Lifetime: {hist_e[-1]})")

    plt.tight_layout()
    plt.show()


# Recommended Rules:
# Rule 110 (Complex)
# Rule 30 (Chaos - expect difficult results)
# Rule 54 (Class 4)
if __name__ == "__main__":
    compare_mutation_strategies(110)