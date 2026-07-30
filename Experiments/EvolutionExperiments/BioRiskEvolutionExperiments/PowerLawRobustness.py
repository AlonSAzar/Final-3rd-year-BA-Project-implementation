import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA


class PowerLawRobustness:
    def __init__(self, engine):
        self.engine = engine
        self.L = engine.L

    def generate_power_law_mask(self, alpha=2.0):
        """
        Generates a damage mask where the size of the damage block
        follows a Power Law (Pareto distribution).
        """
        # 1. Sample size from Pareto
        # Mode is 1 (minimum 1 pixel). Long tail determines max size.
        # Alpha=1.16 gives 80/20 rule. Alpha=2.0 is standard.
        size = int(np.random.pareto(alpha) + 1)

        # Clip size to grid length (can't destroy more than 100%)
        size = min(size, self.L)

        # 2. Pick random location
        start = np.random.randint(0, self.L)

        # 3. Create Mask
        mask = np.zeros(self.L, dtype=bool)

        # Handle wrap-around indices
        indices = [(start + i) % self.L for i in range(size)]
        mask[indices] = True

        return mask, size

    def run_stress_test(self, rule_id, num_trials=1000, alpha=2.0):
        """
        Runs trials with varying damage sizes.
        Returns: Scatter data (Damage_Size, Recovery_Score)
        """
        damage_sizes = []
        recovery_scores = []

        for _ in range(num_trials):
            # A. Generate healthy history
            seed = self.engine.generate_seed()
            history_clean = self.engine.run(rule_id, seed)
            final_clean = history_clean[-1]

            # B. Generate Damage
            mask, size = self.generate_power_law_mask(alpha)

            # C. Apply Damage to Seed (Constitutive Knockout)
            # We enforce 0s in the damaged area (Mutation)
            seed_damaged = seed.copy()
            seed_damaged[mask] = 0

            # D. Run Damaged History
            history_damaged = self.engine.run(rule_id, seed_damaged)
            final_damaged = history_damaged[-1]

            # E. Measure "Healing"
            # How different is the result?
            # 0.0 = Totally different, 1.0 = Perfectly healed
            hamming_diff = np.mean(final_clean != final_damaged)
            similarity = 1.0 - hamming_diff

            damage_sizes.append(size)
            recovery_scores.append(similarity)

        return damage_sizes, recovery_scores

    def compare_rules(self, rules=[30, 110, 168, 192], alpha=1.5):
        plt.figure(figsize=(10, 6))

        for rule in rules:
            print(f"Testing Rule {rule}...")
            sizes, scores = self.run_stress_test(rule, alpha=alpha)

            # Bin the data to get average recovery per size
            # Logarithmic binning makes sense for Power Laws
            bins = np.logspace(0, np.log10(self.engine.L), 10)
            digitized = np.digitize(sizes, bins)

            mean_scores = []
            bin_centers = []

            for i in range(1, len(bins)):
                # Get all scores for this size bin
                bin_data = np.array(scores)[digitized == i]
                if len(bin_data) > 0:
                    mean_scores.append(np.mean(bin_data))
                    bin_centers.append(bins[i - 1])

            plt.plot(bin_centers, mean_scores, marker='o', label=f"Rule {rule}", linewidth=2)

        plt.xscale('log')
        plt.xlabel("Perturbation Size (Log Scale)")
        plt.ylabel("Recovery Score (1.0 = Perfect Healing)")
        plt.title(f"Antifragility Stress Test (Power Law alpha={alpha})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


# Usage
def main():
    engine = ElementaryCA(L=256, T=256) # Use large L to allow massive shocks
    exp = PowerLawRobustness(engine)
    exp.compare_rules([30, 110, 192, 169])

if __name__ == "__main__":
    main()