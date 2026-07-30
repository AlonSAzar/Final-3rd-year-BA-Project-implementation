import numpy as np
import matplotlib.pyplot as plt
import zlib
from tqdm import tqdm

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA

# TODO I might want to use the graph from here somehow

def calculate_structural_info(L=128, T=128, samples_per_rule=50):
    engine = ElementaryCA(L, T)

    densities = []
    structural_infos = []
    rule_labels = []

    print("Calculating Structural Information for all rules...")
    for rule in tqdm(range(256)):
        for _ in range(samples_per_rule):
            seed = engine.generate_seed()
            # Get the final 50 rows (the attractor)
            img = engine.run(rule, seed)[-50:]

            # 1. Calculate Density (X-axis)
            density = np.mean(img)

            # 2. Calculate K_raw
            bytes_raw = img.tobytes()
            k_raw = len(zlib.compress(bytes_raw))

            # 3. Calculate K_shuffled
            # Flatten, shuffle, and reshape to destroy local spatial grammar
            flat_img = img.flatten()
            np.random.shuffle(flat_img)
            shuffled_img = flat_img.reshape(img.shape)

            bytes_shuf = shuffled_img.tobytes()
            k_shuf = len(zlib.compress(bytes_shuf))

            # 4. Structural Information (Y-axis)
            # How much compression was lost when we destroyed the spatial grammar?
            delta_k = k_shuf - k_raw

            densities.append(density)
            structural_infos.append(delta_k)
            rule_labels.append(rule)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    # Plot the CA data
    plt.scatter(densities, structural_infos, alpha=0.5, c='purple', edgecolor='k', s=20, label='CA Phenotypes')

    # Plot the Shuffled Baseline (By definition, it is 0)
    plt.axhline(0, color='yellow', linestyle='--', linewidth=3, label='Maximum Entropy Baseline (Shuffled)')

    # Optional: Highlight specific rules
    highlight_rules = [30, 110, 192]
    colors = ['orange', 'green', 'red']
    for i, r in enumerate(highlight_rules):
        idx = [j for j, label in enumerate(rule_labels) if label == r]
        d_subset = [densities[j] for j in idx]
        si_subset = [structural_infos[j] for j in idx]
        plt.scatter(d_subset, si_subset, c=colors[i], s=80, edgecolor='k', label=f'Rule {r}')

    plt.title("Structural Information vs. Density\nIsolating Spatial Grammar from Shannon Entropy")
    plt.xlabel("Phenotype Density (Fraction of 1s)")
    plt.ylabel("Structural Information $\Delta K$ (Bytes)\n(Higher = More complex spatial patterns)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Calculate and print the average Structural Information value
    avg_si = np.mean(structural_infos)
    print(f"\n--- Statistics ---")
    print(f"Average Structural Information (Delta K): {avg_si:.2f} bytes")

    plt.show()


if __name__ == "__main__":
    calculate_structural_info()