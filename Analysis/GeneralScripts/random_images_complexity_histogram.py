import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Core.ComplexityMeasures.complexity import ZlibComplexity
from Experiments.experiments import shuffle_space_time
import os


def run_random_image_complexity_experiment(L=12, T=64, num_samples=12800, shuffle_control=False):
    """
    Measures the probability distribution of complexities for purely random binary images.

    Parameters:
    - L, T: Height and width dimensions of the generated random image (L x T).
    - num_samples: Total number of random images to generate.
    - shuffle_control: If True, shuffles the generated images across space and time.
    """
    metric = ZlibComplexity()
    complexities = []

    shuffle_str = " (Shuffled Control)" if shuffle_control else ""
    print(f"Running Random Image experiment{shuffle_str}: L={L}, T={T}, {num_samples} samples...")

    for _ in tqdm(range(num_samples), desc=f"Generating Random Images{shuffle_str}"):
        # Generate a random binary image matching shape (L, T) with equal probability (p=0.5)
        random_image = np.random.choice([0, 1], size=(L, T))

        if shuffle_control:
            random_image = shuffle_space_time(random_image)

        k = metric.calculate(random_image)
        complexities.append(k)

    return np.array(complexities)


def plot_complexity_bar_linear(complexities, title="Complexity Probability for Random Images (Linear Scale)"):
    """
    Plots a bar plot of complexity probabilities with linear axes.
    """
    plt.figure(figsize=(12, 6))

    unique_ks, counts = np.unique(complexities, return_counts=True)
    order = np.argsort(unique_ks)
    unique_ks = unique_ks[order]

    # Calculate empirical probabilities P(K)
    probabilities = counts[order] / len(complexities)

    plt.bar(unique_ks, probabilities, color='coral', edgecolor='black', alpha=0.7)

    plt.xscale('linear')
    plt.yscale('linear')

    plt.xlabel("Estimated Kolmogorov Complexity K (Zlib Bytes)", fontsize=14)
    plt.ylabel("Probability P(K)", fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)

    stats_text = (
        f"Mean K: {np.mean(complexities):.2f}\n"
        f"Median K: {np.median(complexities):.1f}\n"
        f"Total Samples: {len(complexities)}"
    )
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    save_dir = os.path.join("Analysis", "GeneralScripts", "Saved Figures")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "random_image_complexity_probability_linear.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()


def plot_complexity_bar_semilog(complexities, title="Complexity Probability for Random Images (Semi-Log Scale)"):
    """
    Plots a bar plot of complexity probabilities where ONLY the Y-axis is log.
    """
    plt.figure(figsize=(12, 6))

    unique_ks, counts = np.unique(complexities, return_counts=True)
    order = np.argsort(unique_ks)
    unique_ks = unique_ks[order]

    # Calculate empirical probabilities P(K)
    probabilities = counts[order] / len(complexities)

    plt.bar(unique_ks, probabilities, color='mediumpurple', edgecolor='black', alpha=0.7)

    plt.xscale('linear')
    plt.yscale('log')

    plt.xlabel("Complexity K (Zlib) [Linear]", fontsize=14)
    plt.ylabel("Probability P(K) [Log Scale]", fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()

    save_dir = os.path.join("Analysis", "GeneralScripts", "Saved Figures")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "random_image_complexity_probability_semilog.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    SHUFFLE_CONTROL = False
    L = 12
    T = 63

    # 256 * 50 = 12800 samples to match CA experiment scale
    data = run_random_image_complexity_experiment(L=L, T=T, num_samples=12800, shuffle_control=SHUFFLE_CONTROL)

    title_suffix = " (Shuffled Control)" if SHUFFLE_CONTROL else ""
    plot_complexity_bar_linear(data, title=f"Complexity Probability for Random Images{title_suffix}, L={L}, T={T + 1}")
    plot_complexity_bar_semilog(data,
                                title=f"Complexity Probability for Random Images (Semi-Log Scale){title_suffix}, L={L}, T={T + 1}")