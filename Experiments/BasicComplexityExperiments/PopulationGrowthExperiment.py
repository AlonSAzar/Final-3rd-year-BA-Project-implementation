from collections import Counter

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from sklearn.linear_model import QuantileRegressor

from Experiments.experiments import *

"""This experiment was done close to deadline, so mostly AI-generated"""


class PopulationGrowthExperiment(BaseExperiment):
    """
    Analyzes the population dynamics (growth/shrinkage) of CA.
    Maps population change time-series to complexity.
    """

    def __init__(self, engine: ElementaryCA, metric: ComplexityMetric):
        self.engine = engine
        self.metric = metric
        self.results = {}  # rule -> list of binary sequences
        self.num_seeds_used = 0
        self.shuffle_control = False
        self.iterate_all_seeds = False

    def run(
        self,
        num_seeds: int,
        rules=range(256),
        shuffle_control: bool = False,
        iterate_all_seeds: bool = False,
    ):
        """Generates population change sequences."""
        self.shuffle_control = shuffle_control
        self.iterate_all_seeds = iterate_all_seeds

        if iterate_all_seeds:
            seeds = [
                np.array(
                    list(np.binary_repr(i, width=self.engine.L)),
                    dtype=np.uint8
                )
                for i in range(2 ** self.engine.L)
            ]
            self.num_seeds_used = len(seeds)
        else:
            seeds = [self.engine.generate_seed() for _ in range(num_seeds)]
            self.num_seeds_used = num_seeds

        for rule in tqdm(rules, desc="Simulating Population Rules"):
            binary_sequences = []

            for seed in seeds:
                # Run simulation and skip the seed (t=0) to focus on the rule's dynamics
                img = self.engine.run(rule, seed)[1:]

                if self.shuffle_control:
                    from Experiments.experiments import shuffle_space_time
                    img = shuffle_space_time(img)

                # 1. Calculate population at each step
                population = np.sum(img, axis=1).astype(int)

                # 2. Calculate differences between steps
                diffs = np.diff(population)

                # 3. Create binary sequence: 1 if grew/same, 0 if shrank
                binary_seq = (diffs >= 0).astype(np.uint8)

                binary_sequences.append(binary_seq)

            self.results[rule] = binary_sequences

        self.analyze_and_plot()

    def analyze_and_plot(self):
        """
        Calculates complexity, correlations, and fits the upper bound curve.
        """
        freq_map = Counter()
        complexity_map = {}

        # Flatten all sequences
        all_seqs = [
            seq
            for sublist in self.results.values()
            for seq in sublist
        ]

        for seq in tqdm(all_seqs, desc="Analyzing Population Complexity"):
            h = seq.tobytes()
            freq_map[h] += 1

            if h not in complexity_map:
                complexity_map[h] = self.metric.calculate(seq)

        # ---------------- PREPARE DATA ----------------
        Ks = []
        log_probs = []
        total = sum(freq_map.values())

        for h, count in freq_map.items():
            Ks.append(complexity_map[h])
            log_probs.append(np.log10(count / total))

        Ks = np.array(Ks)
        log_probs = np.array(log_probs)

        # ---------------- CORRELATIONS ----------------
        if len(Ks) > 1:
            pearson_corr, _ = pearsonr(Ks, log_probs)
            spearman_corr, _ = spearmanr(Ks, log_probs)
        else:
            pearson_corr = spearman_corr = 0

        # ---------------- UPPER BOUND FITTING ----------------
        unique_ks = np.unique(Ks)
        max_log_probs = []

        for k in unique_ks:
            max_val = np.max(log_probs[Ks == k])
            max_log_probs.append(max_val)

        unique_ks = np.array(unique_ks)
        max_log_probs = np.array(max_log_probs)

        if len(unique_ks) > 1:
            qr = QuantileRegressor(quantile=0.95, alpha=0)
            qr.fit(unique_ks.reshape(-1, 1), max_log_probs)
            slope = qr.coef_[0]
            intercept = qr.intercept_
        else:
            slope, intercept = 0, 0

        # Convert to P = 2^(-aK - b)
        log2_10 = np.log2(10)
        a_param = -slope * log2_10
        b_param = -intercept * log2_10

        # ---------------- PLOTTING ----------------
        plt.figure(figsize=(10, 7))

        plt.scatter(
            Ks,
            log_probs,
            alpha=0.4,
            c="purple",
            label="Phenotypes",
            edgecolors="none",
        )

        x_line = np.linspace(min(Ks), max(Ks), 100)
        y_line = slope * x_line + intercept

        plt.plot(
            x_line,
            y_line,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Upper Bound Fit",
        )

        shuffle_str = " (Shuffled Control)" if self.shuffle_control else ""

        plt.xlabel(f"Complexity ({self.metric.name()}){shuffle_str}")
        plt.ylabel("Log10 Probability")
        plt.title(f"Simplicity Bias in Population Dynamics{shuffle_str}, L={self.engine.L}, T={self.engine.T}")

        stats_text = (
            f"Simulation Parameters:\n"
            # f"  L = {self.engine.L}\n"
            # f"  T = {self.engine.T}\n"
            f"  N_seeds = {self.num_seeds_used}"
            f"{' (all possible)' if self.iterate_all_seeds else ''}\n"
            f"  Shuffle Control = {self.shuffle_control}\n\n"
            f"Correlations:\n"
            f"  Spearman = {spearman_corr:.3f}\n"
            f"  Pearson = {pearson_corr:.3f}\n\n"
            f"Fit P = 2^(-aK - b):\n"
            f"  a = {a_param:.3f}\n"
            f"  b = {b_param:.3f}"
        )

        props = dict(boxstyle="round", facecolor="white", alpha=0.8)

        plt.text(
            0.95,
            0.95,
            stats_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=props,
        )

        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower left")
        plt.show()

    def plot(self):
        pass

    def analyze(self):
        pass