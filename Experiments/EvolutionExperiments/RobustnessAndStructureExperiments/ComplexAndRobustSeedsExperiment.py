import zlib
import numpy as np


def calculate_fitness_structural_integrity(self, genotype):
    # 1. Run "Clean" history
    history_clean = self.engine.run(0, genotype)
    final_clean = history_clean[-1]

    # --- CHECK 1: Is it boring? (Variance Check) ---
    # If the variance is too low (all 0s or all 1s), kill it immediately.
    # TODO won't it be better to replace with kolmogorov?
    variance = np.var(final_clean)
    if variance < 0.05:
        return 0.0  # "The Rock" penalty (Stable but dead)

    # 2. Run "Damaged" history
    # Inject damage at T=30 (Kill a block)
    damaged_state = history_clean[30].copy()
    L = len(damaged_state)
    mid = L // 2
    # Wipe out 10% of the center
    gap = max(1, L // 10)
    damaged_state[mid - gap: mid + gap] = 0

    # Run the rest of the simulation from the damaged state
    # TODO (Assuming you added run_from_state to your engine)
    remaining_steps = self.engine.T - 30
    history_damaged = self.engine.run_from_state(0, damaged_state, remaining_steps)
    final_damaged = history_damaged[-1]

    # --- CHECK 2: Stability (Hamming Similarity) ---
    # TODO use NCC I think
    # Fraction of cells that match the original timeline
    similarity = np.mean(final_clean == final_damaged)

    # --- CHECK 3: Complexity Boost ---
    # High variance = not empty.

    # Total Fitness:
    # We reward Stability, but scaled by how "alive" the pattern is.
    fitness = similarity * variance * 10

    return fitness