from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, linregress
from sklearn.linear_model import QuantileRegressor

class UpperBoundPlotter:

    def __init__(self, metric, engine, num_seeds_used):
        self.metric = metric
        self.engine = engine
        self.num_seeds_used = num_seeds_used

    """This function specifically was written by AI, since it's a plotting function"""

    def plot(self,  freq_map, complexity_map, title, **kwargs):

        """
        Visualizes the Simplicity Bias with Upper Bound Fit and Stats.
        """
        Ks = []  # Complexities
        log_probs = []  # log(Probability)
        total = sum(freq_map.values())

        for h, count in freq_map.items():
            Ks.append(complexity_map[h])
            log_probs.append(np.log10(count / total))

        # Convert to numpy for calculations
        Ks = np.array(Ks)
        log_probs = np.array(log_probs)

        # ---------------- CORRELATIONS ----------------
        if len(Ks) > 1:
            pearson_corr, _ = pearsonr(Ks, log_probs)
            spearman_corr, _ = spearmanr(Ks, log_probs)
        else:
            pearson_corr = spearman_corr = 0

        # ---------------- UPPER BOUND FITTING ----------------
        # 1. Find max probability for each unique complexity value
        unique_ks = np.unique(Ks)
        max_log_probs = []
        for k in unique_ks:
            # Get all log_probs that have complexity == k
            max_val = np.max(log_probs[Ks == k])
            max_log_probs.append(max_val)

        unique_ks = np.array(unique_ks)
        max_log_probs = np.array(max_log_probs)

        # 2. Fit quantile regression to the upper bound (y = mx + c)
        if len(unique_ks) > 1:
            qr = QuantileRegressor(quantile=0.95, alpha=0)
            qr.fit(unique_ks.reshape(-1, 1), max_log_probs)
            slope = qr.coef_[0]
            intercept = qr.intercept_
        else:
            slope, intercept = 0, 0

        # 3. Convert to form P(p) = 2^(-aK - b)
        # We fitted log10(P) = slope * K + intercept
        # a = -slope * log2(10), b = -intercept * log2(10)
        log2_10 = np.log2(10)
        a_param = -slope * log2_10
        b_param = -intercept * log2_10

        # ---------------- PLOTTING ----------------
        # Use a larger figure size to match annotated plotter for seamless transitions
        plt.figure(figsize=(14, 8))
        ax = plt.gca()
        
        # Ensure teal color and alpha match the annotated version
        scatter_label = 'Phenotypes'
        plt.scatter(Ks, log_probs, alpha=0.2, c='teal', label=scatter_label)

        if 'xlim' in kwargs:
            ax.set_xlim(kwargs['xlim'])
        if 'ylim' in kwargs:
            ax.set_ylim(kwargs['ylim'])

        # Plot Fitted Line
        x_line = np.linspace(min(Ks), max(Ks), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, color='red', linestyle='--', linewidth=2, label='Upper Bound Fit')

        # Labels
        plt.xlabel("Estimated Kolmogorov Complexity (Zlib Bytes)")
        plt.ylabel("Log10 Probability")
        plt.title(f"{title} (L={self.engine.L}, T={self.engine.T})")

        # Info Box / Legend
        if not kwargs.get('hide_stats', False):
            stats_text = (
                f"Simulation Parameters:\n"
                f"  N_seeds = {self.num_seeds_used}\n\n"
                f"Correlations:\n"
                f"  Spearman = {spearman_corr:.3f}\n"
                f"  Pearson = {pearson_corr:.3f}\n\n"
                f"Fit P = 2^(-aK - b):\n"
                f"  a = {a_param:.3f}\n"
                f"  b = {b_param:.3f}"
            )

            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
                     verticalalignment='top', horizontalalignment='right', bbox=props, zorder=10)
        else:
            # When stats are hidden (as in prepare_presentation.py), 
            # put Spearman in a separate small box at top-right
            corr_text = f"Spearman Correlation: {spearman_corr:.3f}"
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            plt.text(0.95, 0.95, corr_text, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', bbox=props, zorder=10)

        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower left')
        # Apply larger spacing on the right even in standard plot for consistency
        plt.subplots_adjust(right=0.55, left=0.1)
        plt.show()
