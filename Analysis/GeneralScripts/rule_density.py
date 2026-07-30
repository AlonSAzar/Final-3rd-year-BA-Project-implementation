import numpy as np


def run_eca(rule_number, L=1000, T=500):
    """
    Simulates a 1D Elementary Cellular Automaton using periodic boundary conditions.
    """
    rule_bin = np.array([(rule_number >> i) & 1 for i in range(8)], dtype=np.uint8)

    # Random initial seed (50% density start)
    state = np.random.randint(0, 2, size=L, dtype=np.uint8)
    grid = np.zeros((T, L), dtype=np.uint8)
    grid[0] = state

    for t in range(1, T):
        left = np.roll(state, 1)
        right = np.roll(state, -1)
        # Vectorized lookup index from 3-cell neighborhood (0 to 7)
        neighborhood = (left << 2) | (state << 1) | right
        state = rule_bin[neighborhood]
        grid[t] = state

    return grid


def measure_rule_density(rule_number, L=1000, T=500, transient_steps=100, num_trials=20):
    """
    Measures the stationary spatial density of 1s across multiple trials.

    Parameters:
        rule_number (int): ECA rule (0-255)
        L (int): Spatial width of the grid
        T (int): Time steps to simulate per trial
        transient_steps (int): Number of initial time steps to discard
        num_trials (int): Number of random initial conditions to average over
    """
    trial_densities = []

    for _ in range(num_trials):
        grid = run_eca(rule_number, L=L, T=T)
        # Discard the transient warm-up rows
        stationary_grid = grid[transient_steps:]
        trial_densities.append(np.mean(stationary_grid))

    mean_density = np.mean(trial_densities)
    std_density = np.std(trial_densities)

    return mean_density, std_density


if __name__ == "__main__":
    # Test rules of interest
    rules_to_test = [22, 30, 45, 90, 146]

    print(f"{'Rule':<8} | {'Mean Density':<15} | {'Std Dev':<10} | {'Status for CG (Majority >50%)'}")
    print("-" * 72)

    for rule in rules_to_test:
        mean_d, std_d = measure_rule_density(
            rule_number=rule,
            L=1000,
            T=600,
            transient_steps=100,
            num_trials=20
        )

        # Check if density is too far from 0.50 for majority coarse-graining
        cg_status = "Safe (~50%)" if abs(mean_d - 0.5) < 0.05 else "Will Collapse (Needs adaptive threshold)"

        print(f"Rule {rule:<3} | {mean_d:<15.4f} | {std_d:<10.4f} | {cg_status}")