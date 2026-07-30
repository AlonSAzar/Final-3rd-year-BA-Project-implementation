from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
from sklearn.isotonic import IsotonicRegression
import statsmodels.api as sm
from tqdm import tqdm

from Experiments.experiments import *


class RobustnessExperiment(BaseExperiment):
    """
    Tests the relationship between Complexity and Evolutionary Robustness.
    """

    def __init__(self, engine: ElementaryCA, metric: ComplexityMetric, random_iterations: int = 20):
        super().__init__(engine, metric)
        self.random_iterations = random_iterations

    def run(self, strategy, num_seeds=20, shuffle_control=False, mut_engine=None):
        """
        Returns dictionaries {rule_id: robustness_score}, {rule_id, phenotype_complexity}.
        Robustness = Average NCC between Rule(seed) and MutantRule(seed).
        """

        print(f"--- Robustness: {strategy.name()} ---")
        # Setup dicts
        rule_robustness_scores = {}
        rule_images_dict = {}
        seeds = [self.engine.generate_seed() for _ in range(num_seeds)]

        if mut_engine is None:
            mut_engine = self.engine

        for rule in tqdm(range(256), desc="Evolutionary Robustness"):
            rule_score = 0
            rule_images_list = []

            for seed in seeds:
                base_img = self.engine.run(rule, seed)[1:]
                rule_images_list.append(base_img)

                # Check all 8 1-bit neighbors
                mutant_scores = []

                num_vars = strategy.get_variations_count(self.engine, rule, seed, self.random_iterations)

                for i in range(num_vars):
                    m_rule, m_seed = strategy.apply(self.engine, rule, seed, i)

                    mut_img = mut_engine.run(m_rule, m_seed)[1:]

                    # Optional: Shuffle image to test control
                    if shuffle_control:
                        mut_img = shuffle_space_time(mut_img)

                    # Compute NCC (Normalized Cross Correlation)
                    ncc = compute_ncc(base_img, mut_img)
                    mutant_scores.append(ncc)
                rule_score += np.mean(mutant_scores)

            # Calculate average across seeds, per rule
            rule_robustness_scores[rule] = rule_score / len(seeds)
            rule_images_dict[rule] = rule_images_list

        rule_phenotype_complexity_scores = self.mean_phenotype_complexity(rule_images_dict)

        # Skip internal plotting if we are just collecting results
        # self.plot_results(rule_robustness_scores, rule_phenotype_complexity_scores, strategy, mut_engine, shuffle_control)

        # New model fitting and plotting
        self.plot_with_fits(rule_robustness_scores, rule_phenotype_complexity_scores, strategy)

        return rule_robustness_scores, rule_phenotype_complexity_scores

    # TODO change name to plot like in abstract class
    """ This function is AI generated. """

    def plot_results(self, robustness_scores: dict, phenotype_complexities: dict, mutation_type, mut_engine, shuffle):
        """
        Plots Robustness vs Complexity and calculates correlations.
        """
        # 1. Align data (ensure we match the same rule for X and Y)
        # We sort by rule ID to ensure lists correspond index-for-index
        rules = sorted(robustness_scores.keys())

        # X = Complexity, Y = Robustness
        Xs = [phenotype_complexities[r] for r in rules]
        Ys = [robustness_scores[r] for r in rules]

        # 2. Calculate Correlations
        # TODO see if the warning is a problem
        pearson_val, _ = pearsonr(Xs, Ys)
        spearman_val, _ = spearmanr(Xs, Ys)

        # 3. Plotting
        plt.figure(figsize=(10, 7))
        plt.scatter(Xs, Ys, alpha=0.6, c='teal', edgecolors='black', linewidth=0.5)

        plt.xlabel(f"Phenotype Complexity ({self.metric.name()})")
        plt.ylabel("Robustness (Avg NCC)")
        shuffle_str = ""
        if shuffle:
            shuffle_str = " Shuffled Control."

        # Pull L and T from the engine instance
        L = self.engine.L
        T = self.engine.T
        plt.title(
            f"1D CA Robustness VS Complexity, {mutation_type.name()}, {mut_engine.name()}. L={L}, T={T}.{shuffle_str}")

        # 4. Info Box with Stats
        stats_text = (
            f"Correlations:\n"
            f"  Spearman = {spearman_val:.3f}\n"
            f"  Pearson = {pearson_val:.3f}"
        )

        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_with_fits(self, robustness_scores: dict, phenotype_complexities: dict, strategy):
        """
        Generates separate graphs for each fitting model.
        """
        rules = sorted(robustness_scores.keys())
        Xs = np.array([phenotype_complexities[r] for r in rules])
        Ys = np.array([robustness_scores[r] for r in rules])

        # Sort for smooth line plotting
        sort_idx = np.argsort(Xs)
        Xs_sorted = Xs[sort_idx]
        Ys_sorted = Ys[sort_idx]

        models = ["Exponential", "Quantile Log-Linear (80th)", "Quantile Log-Linear (90th)",
                  "Quantile Log-Linear (95th)", "Logistic Decay", "Isotonic"]

        for model in models:
            plt.figure(figsize=(10, 7))
            plt.scatter(Xs, Ys, alpha=0.4, c='gray', label='Data')

            try:
                if model == "Exponential":
                    # y = a * exp(b * x)
                    def exp_func(x, a, b):
                        return a * np.exp(b * x)

                    popt, _ = curve_fit(exp_func, Xs, Ys, p0=(1, -0.01), maxfev=5000)
                    plt.plot(Xs_sorted, exp_func(Xs_sorted, *popt), 'r-', lw=2,
                             label=f'Fit: {popt[0]:.2f}*exp({popt[1]:.2f}*x)')

                elif model.startswith("Quantile Log-Linear"):
                    # Quantile regression on log(Y) ~ X, then exponentiate back
                    import pandas as pd
                    import statsmodels.formula.api as smf
                    # Shift Y if non-positive
                    min_y = np.min(Ys)
                    shift = 0
                    if min_y <= 0:
                        shift = abs(min_y) + 1e-5
                    log_Ys = np.log(Ys + shift)
                    df = pd.DataFrame({'X': Xs, 'logY': log_Ys})
                    # Determine quantile from model name (80/90/95)
                    if '95' in model:
                        q = 0.95
                    elif '90' in model:
                        q = 0.9
                    elif '80' in model:
                        q = 0.8
                    else:
                        q = 0.95
                    res = smf.quantreg('logY ~ X', df).fit(q=q)
                    pred_log = res.predict(pd.DataFrame({'X': Xs_sorted}))
                    pred_y = np.exp(pred_log) - shift
                    plt.plot(Xs_sorted, pred_y, 'b-', lw=2, label=f'Quantile {int(q * 100)}th (log-linear)')
                    try:
                        intercept = res.params['Intercept']
                        slope = res.params['X']
                        plt.title(
                            f"Quantile Log-Linear ({int(q * 100)}th) - Intercept: {intercept:.2f}, Slope: {slope:.2f}")
                    except Exception:
                        pass

                elif model == "Logistic Decay":
                    # y = L / (1 + exp(-k(x-x0)))
                    def logistic_func(x, L, k, x0):
                        return L / (1 + np.exp(-k * (x - x0)))

                    # Heuristic p0: L=max(Y), k=-0.1 (decay), x0=mean(X)
                    p0 = [np.max(Ys), -0.1, np.mean(Xs)]
                    popt, _ = curve_fit(logistic_func, Xs, Ys, p0=p0, maxfev=5000)
                    plt.plot(Xs_sorted, logistic_func(Xs_sorted, *popt), 'm-', lw=2, label='Logistic Decay')

                elif model == "Isotonic":
                    # Force a decreasing isotonic regression
                    ir = IsotonicRegression(increasing=False, out_of_bounds='clip')
                    # Fit on original data then predict on sorted X for plotting
                    ir.fit(Xs, Ys)
                    plt.plot(Xs_sorted, ir.predict(Xs_sorted), 'orange', lw=2, label='Isotonic (decreasing)')

            except Exception as e:
                plt.title(f"{model} Fit Failed: {str(e)}")

            plt.xlabel(f"Phenotype Complexity ({self.metric.name()})")
            plt.ylabel("Robustness (Avg NCC)")
            plt.title(f"Robustness vs Complexity: {model} model\n{strategy.name()}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

    def compute_rule_complexity(self, rule: int):
        """Compute the complexity of the array represented by the rule itself."""
        binary_rule = int_to_binary_array_numpy(rule)
        return self.metric.calculate(binary_rule)

    def mean_phenotype_complexity(self, results):
        """Compute mean phenotype complexity per rule over all seeds."""
        mean_complexities = {}
        for rule, images in results.items():
            complexities = [self.metric.calculate(img) for img in images]
            mean_complexities[rule] = np.mean(complexities)
        return mean_complexities

    # TODO I can implement, but I think we're past the point of relevance
    def plot(self):
        pass

    def analyze(self):
        pass
