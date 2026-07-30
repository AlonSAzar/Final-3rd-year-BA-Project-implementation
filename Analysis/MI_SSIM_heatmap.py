import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import sys

from sklearn.metrics import mutual_info_score
from skimage.metrics import structural_similarity as ssim

# Add the project root to sys.path to resolve imports properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.strategies import *


from sklearn.metrics import normalized_mutual_info_score

def compute_mutual_information(img1, img2):
    return normalized_mutual_info_score(
        img1.flatten(),
        img2.flatten()
    )


def compute_ssim(img1, img2):
    """
    Computes SSIM between two binary CA images.
    """

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Ensure window size fits small CA images
    min_dim = min(img1.shape)

    win_size = min(7, min_dim)

    # SSIM requires odd window size
    if win_size % 2 == 0:
        win_size -= 1

    return ssim(
        img1,
        img2,
        data_range=1.0,
        win_size=win_size
    )


def calculate_robustness(L,
                         T,
                         rules=[30],
                         num_seeds=100,
                         metric="mi"):
    """
    Calculates robustness using one of:
        metric = "mi"
        metric = "ssim"
    """

    engine = ElementaryCA(L=L, T=T)
    strategy = BitFlipSeedStrategy()

    if rules is None:
        rules = range(256)

    total_robustness = 0

    for rule in rules:

        rule_score = 0

        seeds = [engine.generate_seed() for _ in range(num_seeds)]

        for seed in seeds:

            base_img = engine.run(rule, seed)[1:]

            num_vars = strategy.get_variations_count(engine, rule, seed, 0)
            sample_size = min(num_vars, 20)

            mutant_scores = []

            mutation_indices = np.random.choice(
                range(num_vars),
                sample_size,
                replace=False
            )

            for i in mutation_indices:

                m_rule, m_seed = strategy.apply(engine, rule, seed, i)
                mut_img = engine.run(m_rule, m_seed)[1:]

                if metric.lower() == "mi":
                    score = compute_mutual_information(base_img, mut_img)

                elif metric.lower() == "ssim":
                    score = compute_ssim(base_img, mut_img)

                else:
                    raise ValueError("metric must be 'mi' or 'ssim'")

                mutant_scores.append(score)

            rule_score += np.mean(mutant_scores)

        total_robustness += rule_score / num_seeds

    return total_robustness / len(rules), strategy.name()


def main():

    # Choose metric here
    metric = "ssim"       # "mi" or "ssim"

    # Grid Sizes (L) and Time Steps (T)
    L_values = [10, 50, 100, 250, 500]
    T_values = [10, 50, 100, 250, 500]

    robustness_matrix = np.zeros((len(T_values), len(L_values)))
    strategy_name = ""

    print(f"Starting Robustness Experiment using {metric.upper()}...")

    all_rules = range(256)
    num_seeds = 1

    for i, T in enumerate(tqdm(T_values, desc="T loop")):
        for j, L in enumerate(L_values):

            score, s_name = calculate_robustness(
                L,
                T,
                rules=[30],          # replace with all_rules if desired
                num_seeds=num_seeds,
                metric=metric
            )

            robustness_matrix[i, j] = score
            strategy_name = s_name

            print(f"L={L}, T={T} -> {metric.upper()}: {score:.4f}")

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        robustness_matrix,
        annot=True,
        fmt=".3f",
        xticklabels=L_values,
        yticklabels=T_values,
        cmap="viridis"
    )

    plt.title(
        f"Robustness ({metric.upper()}) Heatmap ({strategy_name})"
    )
    plt.xlabel("Grid Size (L)")
    plt.ylabel("Time Steps (T)")

    output_dir = "Saved Figures"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"Robustness_{metric.upper()}_L_vs_T_Heatmap.png"

    plt.savefig(os.path.join(output_dir, filename))

    print(f"Heatmap saved to {os.path.join(output_dir, filename)}")

    plt.show()


if __name__ == "__main__":
    main()