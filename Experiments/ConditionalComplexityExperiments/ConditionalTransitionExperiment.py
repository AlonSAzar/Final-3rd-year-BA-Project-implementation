from collections import Counter

from tqdm import tqdm

from Core import strategies
from Core.ComplexityMeasures.conditional_complexity import ZlibConditionalComplexity, ConditionalComplexityMetric
from Experiments.experiments import *
from Core.strategies import *
from scipy.stats import pearsonr, spearmanr, linregress
from sklearn.linear_model import QuantileRegressor

# TODO implement abstract class
class ConditionalTransitionExperiment():
    def __init__(self, engine, metric: ConditionalComplexityMetric, last_row_only: bool = True):
        self.engine = engine
        self.metric = metric
        self.last_row_only = last_row_only

    def run(self, rules=range(256), num_parents=50, strategy=strategies.BitFlipSeedStrategy(), shuffle_control=False, **kwargs):
        """
        Maps transitions P(x -> y) vs Conditional Complexity K(y|x).
        """
        self.strategy_name = strategy.name()
        
        if not kwargs.get('skip_simulation', False):
            self.transitions = Counter()  # Counts of specific (hash(x), hash(y))
            self.parent_creations = Counter()  # Counts of unique PARENT creations
            self.creations = Counter()         # Counts of ALL creations (parent + child)
            self.parent_cache = {}    # Stores ONLY parent images by hash
            self.phenotype_cache = {} # Stores ALL images by hash
            self.num_seeds_used = num_parents  # Store for the plot stats
        
        if not kwargs.get('skip_simulation', False):
            if kwargs.get('use_all_seeds', False):
                L = self.engine.L
                parent_seeds = []
                for i in range(2**L):
                    seed_bits = [(i >> bit) & 1 for bit in range(L - 1, -1, -1)]
                    parent_seeds.append(np.array(seed_bits, dtype=np.uint8))
                self.num_seeds_used = len(parent_seeds)
            else:
                parent_seeds = [self.engine.generate_seed() for _ in range(num_parents)]
                self.num_seeds_used = num_parents

            for rule in tqdm(rules, desc="Mapping Transitions: "):
                for p_seed in parent_seeds:
                    # get parent phenotype
                    full_history = self.engine.run(rule, p_seed)
                    if self.last_row_only:
                        img_x = full_history[-1:]  # Slice to keep 2D shape (1, L)
                    else:
                        img_x = full_history       # Keep full (T, L)
                    h_x = img_x.tobytes()
                    self.phenotype_cache[h_x] = img_x
                    self.parent_cache[h_x] = img_x
                    self.creations[h_x] += 1
                    self.parent_creations[h_x] += 1

                    # Generate mutants (1-bit flip neighbors)
                    # Assuming seed length is engine.L
                    for bit in range(self.engine.L):
                        # TODO also this needs refinement in accordance with the strategies,
                        # such that we'll be able to use different strategies more easily
                        m_rule, m_seed = strategy.apply(engine=self.engine, rule=rule, seed=p_seed, bit_index=bit)

                        # Get Mutant Phenotype (y)
                        img_y = self.engine.run(m_rule, m_seed)
                        if self.last_row_only:
                            img_y = img_y[-1:]
                        if shuffle_control:
                            img_y = shuffle_space_time(img_y)
                        # else: img_y = img_y (full)
                        h_y = img_y.tobytes()
                        self.phenotype_cache[h_y] = img_y
                        self.creations[h_y] += 1

                        # Record Transition
                        self.transitions[(h_x, h_y)] += 1

        if not kwargs.get('just_distribute', False):
            self.analyze_conditional_simplicity_bias(strategy, shuffle_control, **kwargs)

    def return_phenotype_cache(self):
        return self.phenotype_cache

    def return_parent_cache(self):
        return self.parent_cache

    def return_transitions(self):
        return self.transitions

    def return_creations(self):
        return self.creations

    def return_parent_creations(self):
        return self.parent_creations

    # Analyze Data points
    def analyze_conditional_simplicity_bias(self, strategy, shuffle_control, **kwargs):
        plot_data_k = []
        plot_data_prob = []
        transition_info = []

        total_mutations_per_parent = Counter()
        for (h_x, h_y), count in self.transitions.items():
            total_mutations_per_parent[h_x] += count

        print("Calculating Conditional Complexities...")
        for (h_x, h_y), count in self.transitions.items():
            img_x = self.phenotype_cache[h_x]
            img_y = self.phenotype_cache[h_y]

            # Calculate K(y|x)
            k_cond = self.metric.calculate(img_x, img_y)

            # NORMALIZED PROBABILITY: P(x -> y)
            prob = count / total_mutations_per_parent[h_x]

            plot_data_k.append(k_cond)
            plot_data_prob.append(np.log10(prob))
            transition_info.append((h_x, h_y))
        
        # Determine title and plotting logic
        if kwargs.get('annotated', False):
            from Visualizations.annotated_plotter import AnnotatedSimplicityBiasPlotter
            plotter = AnnotatedSimplicityBiasPlotter(self.metric, self.engine, self.num_seeds_used)
            
            # Create a mapping that shows transition pairs
            transition_map = {}
            for i, (h_x, h_y) in enumerate(transition_info):
                transition_map[i] = (self.phenotype_cache[h_x], self.phenotype_cache[h_y])
            
            # Call specialized plotter
            plotter.plot_csb(plot_data_k, plot_data_prob, transition_map, strategy, shuffle_control, **kwargs)
        else:
            self.plot_results(plot_data_k, plot_data_prob, strategy, shuffle_control)

    """This function specifically was written by AI, since it's a plotting function"""

    # Combined both into one clean method and removed the messy duplicated one
    def plot_results(self, Ks, log_probs, strategy, shuffle_control, title_add: str = ""):
        """
        Visualizes the Conditional Simplicity Bias with Upper Bound Fit and Stats.
        Ks: List/Array of conditional complexities K(y|x)
        log_probs: List/Array of log10(P(x -> y))
        """
        import numpy as np
        import matplotlib.pyplot as plt

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
        log2_10 = np.log2(10)
        a_param = -slope * log2_10
        b_param = -intercept * log2_10

        # ---------------- PLOTTING ----------------
        plt.figure(figsize=(10, 7))

        # Scatter Plot
        plt.scatter(Ks, log_probs, alpha=0.5, label='Transitions (x -> y)', s=20)

        # Plot Fitted Line
        if len(Ks) > 0:
            x_line = np.linspace(min(Ks), max(Ks), 100)
            y_line = slope * x_line + intercept
            plt.plot(x_line, y_line, color='red', linestyle='--', linewidth=2, label='Upper Bound Fit')

        shuffle_str = ""
        if shuffle_control:
            shuffle_str = " Shuffled Control."

        # Labels
        plt.xlabel(f"Conditional Complexity K(y|x) ({self.metric.name()}, {strategy.name()}.{shuffle_str})")
        plt.ylabel("Log10 Transition Probability P(x → y)")
        plt.title(f"Conditional Simplicity Bias (Standard CA, L={self.engine.L}, T={self.engine.T})\n" + title_add)

        # Info Box (Matching your template)
        stats_text = (
            f"Simulation Parameters:\n"
            f"  N_parent_seeds = {self.num_seeds_used}\n"
            # f"  Seed Length L = {self.engine.L}\n"
            f"  Last Row Only = {self.last_row_only}\n\n"
            f"Correlations:\n"
            f"  Spearman = {spearman_corr:.3f}\n"
            f"  Pearson = {pearson_corr:.3f}\n\n"
            f"Fit P = 2^(-aK - b):\n"
            f"  a = {a_param:.3f}\n"
            f"  b = {b_param:.3f}"
        )

        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.show()

    def return_phenotype_cache(self):
        return self.phenotype_cache

    def return_transitions(self):
        return self.transitions
