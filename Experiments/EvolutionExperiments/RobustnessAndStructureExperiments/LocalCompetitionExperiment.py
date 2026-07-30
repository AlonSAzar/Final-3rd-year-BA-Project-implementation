import numpy as np
from tqdm import tqdm


def run_spatial_evolution(engine, grid_size=30, generations=100):
    # 1. Initialize a Grid of Seeds (Genotypes)
    # Shape: (Grid_H, Grid_W, Seed_Length)
    population_grid = np.random.randint(0, 2, (grid_size, grid_size, engine.L), dtype=np.uint8)

    history_fitness = []

    for gen in tqdm(range(generations)):
        # Calculate Fitness Map
        fitness_map = np.zeros((grid_size, grid_size))
        for r in range(grid_size):
            for c in range(grid_size):
                # Run the CA for this specific individual
                seed = population_grid[r, c]
                pheno = engine.run(0, seed)
                # Use your landscape fitness (Trap vs Optimum)
                fitness_map[r, c] = calculate_fitness_landscape(pheno.tobytes())

        history_fitness.append(np.mean(fitness_map))

        # Spatial Tournament Selection
        new_grid = population_grid.copy()

        for r in range(grid_size):
            for c in range(grid_size):
                # Look at Moore Neighborhood (3x3)
                r_min, r_max = max(0, r - 1), min(grid_size, r + 2)
                c_min, c_max = max(0, c - 1), min(grid_size, c + 2)

                # Find best neighbor
                local_fitness = fitness_map[r_min:r_max, c_min:c_max]
                best_idx = np.unravel_index(np.argmax(local_fitness), local_fitness.shape)

                # Convert local index back to global
                best_r = r_min + best_idx[0]
                best_c = c_min + best_idx[1]

                # Overwrite current cell with best neighbor's genes
                winner_genome = population_grid[best_r, best_c]

                # Apply Mutation
                if np.random.random() < 0.01:
                    mut_idx = np.random.randint(0, engine.L)
                    winner_genome[mut_idx] = 1 - winner_genome[mut_idx]

                new_grid[r, c] = winner_genome

        population_grid = new_grid

    return history_fitness