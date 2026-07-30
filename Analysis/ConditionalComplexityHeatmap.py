import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.ComplexityMeasures.conditional_complexity import ZlibConditionalComplexity
from Core.strategies import *


def calculate_conditional_complexity(
    L,
    T,
    rules=[30],
    num_seeds=100,
):
    """
    Calculates the average conditional complexity K(y|x)
    between the original phenotype and bit-flipped seed mutants.
    """
    engine = ElementaryCA(L=L, T=T)
    strategy = BitFlipSeedStrategy()
    metric = ZlibConditionalComplexity()

    if rules is None:
        rules = range(256)

    total_complexity = 0.0

    for rule in rules:
        rule_score = 0.0

        seeds = [engine.generate_seed() for _ in range(num_seeds)]

        for seed in seeds:
            # Original phenotype
            x = engine.run(rule, seed)[1:]

            num_vars = strategy.get_variations_count(engine, rule, seed, 0)

            # Sample at most 20 mutations (same behavior as original script)
            sample_size = min(num_vars, 20)

            mutation_indices = np.random.choice(
                range(num_vars),
                sample_size,
                replace=False,
            )

            mutant_scores = []

            for i in mutation_indices:
                m_rule, m_seed = strategy.apply(
                    engine,
                    rule,
                    seed,
                    i,
                )

                # Mutated phenotype
                y = engine.run(m_rule, m_seed)[1:]

                k_cond = metric.calculate(x, y)
                mutant_scores.append(k_cond)

            rule_score += np.mean(mutant_scores)

        total_complexity += rule_score / num_seeds

    return total_complexity / len(rules), strategy.name()


def main():
    # Grid Sizes (L) and Time Steps (T)
    L_values = [10, 50, 100, 250, 500]
    T_values = [10, 50, 100, 250, 500]

    complexity_matrix = np.zeros((len(T_values), len(L_values)))
    strategy_name = ""

    print("Starting Grid Size vs Time Steps Conditional Complexity Experiment...")

    # Change this to range(256) if desired
    all_rules = [30]

    num_seeds = 1

    for i, T in enumerate(tqdm(T_values, desc="T loop")):
        for j, L in enumerate(L_values):

            score, s_name = calculate_conditional_complexity(
                L=L,
                T=T,
                rules=all_rules,
                num_seeds=num_seeds,
            )

            complexity_matrix[i, j] = score
            strategy_name = s_name

            print(
                f"L={L}, T={T} -> Average Conditional Complexity: {score:.3f}"
            )

    # Plot heatmap
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        complexity_matrix,
        annot=True,
        fmt=".2f",
        xticklabels=L_values,
        yticklabels=T_values,
        cmap="viridis",
    )

    plt.title(
        f"Average Conditional Complexity Heatmap ({strategy_name})\n"
        "Bit-Flip Seed Perturbation"
    )
    plt.xlabel("Grid Size (L)")
    plt.ylabel("Time Steps (T)")

    output_dir = "Saved Figures"
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(
        output_dir,
        "ConditionalComplexity_L_vs_T_Heatmap.png",
    )

    plt.savefig(save_path)
    print(f"Heatmap saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    main()