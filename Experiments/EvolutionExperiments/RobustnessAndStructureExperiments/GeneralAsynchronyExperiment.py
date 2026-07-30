import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
from sklearn.isotonic import IsotonicRegression
import statsmodels.api as sm
import pandas as pd
import statsmodels.formula.api as smf
from Core.ComplexityMeasures.complexity import ComplexityMetric
from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA, _rule_to_lut
from Core.Engines.OneDimensionEngines.OneDimensionAsynchronousEngine import AsynchronousCA, _simulate_1d_numba_async
from Core import strategies
from Core.Engines.OneDimensionEngines.OneDimensionFrozenCellsEngine import FrozenCellsCA
from Core.Engines.OneDimensionEngines.OneDimensionNoiseEngine import RandomNoiseCA
from Experiments.experiments import BaseExperiment, compute_ncc


class GeneralAsynchronyExperiment(BaseExperiment):
    """
    Generalization of the Synchronization Cliff experiment.
    Compares a 'reliable' baseline against an 'asynchronous' mutant engine,
    mapping how baseline complexity relates to structural robustness under asynchrony.
    """

    def __init__(self, engine: ElementaryCA, metric: ComplexityMetric, random_iterations: int = 20):
        super().__init__(engine, metric)
        self.random_iterations = random_iterations

    def run(self,
            strategy=strategies.SameSeedAndRuleStrategy(),
            mut_engine=RandomNoiseCA(0.003, 128, 128),
            num_seeds=20,
            rules=range(256),
            use_numba_async_default=False):
        """
        Runs the comparison between control (self.engine) and async (mut_engine).
        
        Args:
            strategy: Mutation strategy (defaults to SameSeedAndRule for basic async tests).
            mut_engine: The async engine to test. If None and use_numba_async_default=True, 
                        uses _simulate_1d_numba_async via a temporary AsynchronousCA.
            num_seeds: Number of seeds per rule.
            rules: Range of rules to scan.
            use_numba_async_default: If True and mut_engine is None, defaults to the 
                                     standard random-sequential async update.
        """
        print(f"--- General Asynchrony Cliff: {strategy.name()} ---")

        # Determine the asynchronous engine to compare against
        if mut_engine is None:
            if use_numba_async_default:
                mut_engine = AsynchronousCA(self.engine.L, self.engine.T)
            else:
                raise ValueError("Must provide a mut_engine or set use_numba_async_default=True")

        baseline_complexities = []
        async_complexities = []
        robustness_scores = []
        rule_labels = []

        # Generate seeds once for consistency across rules? Or per rule?
        # Following RobustnessExperiment pattern of generating seeds upfront.
        seeds = [self.engine.generate_seed() for _ in range(num_seeds)]

        for rule in tqdm(rules, desc="Scanning Async Robustness"):
            rule_base_k = []
            rule_async_k = []
            rule_robustness = []

            for seed in seeds:
                # 1. Control: Perfect Clock (Standard CA)
                # We use the base engine run. 
                # Note: RobustnessExperiment skips first row [1:], we follow suit for consistency
                sync_history = self.engine.run(rule, seed)[1:]

                # 2. Async Mutant: Run the strategy and async engine
                # For basic asynchrony, strategy might just return the same rule/seed
                # but we support the general pattern.
                m_rule, m_seed = strategy.apply(self.engine, rule, seed, 0)  # Use first variation
                async_history = mut_engine.run(m_rule, m_seed)[1:]

                # 3. Measure Complexity
                # Calculate complexity of the baseline and the async version
                sync_k = self.metric.calculate(sync_history)
                async_k = self.metric.calculate(async_history)

                # 4. Measure Robustness (Deviation from Baseline)
                # 1.0 = Structure maintained, 0.0 = Totally collapsed/changed
                # Using relative difference in complexity as a proxy for structural maintenance
                deviation = abs(sync_k - async_k) / max(sync_k, 1e-5)
                robustness = max(0.0, 1.0 - deviation)

                # Alternative: Use NCC if histories are expected to be similar
                # ncc = compute_ncc(sync_history, async_history)
                # robustness = ncc

                rule_base_k.append(sync_k)
                rule_async_k.append(async_k)
                rule_robustness.append(robustness)

            baseline_complexities.append(np.mean(rule_base_k))
            async_complexities.append(np.mean(rule_async_k))
            robustness_scores.append(np.mean(rule_robustness))
            rule_labels.append(rule)

        results = {
            'complexities': np.array(baseline_complexities),
            'async_complexities': np.array(async_complexities),
            'robustness': np.array(robustness_scores),
            'rules': np.array(rule_labels)
        }

        self.plot_synchronization_cliff(results, strategy, mut_engine)
        self.plot_transition_matrix(results, strategy, mut_engine)
        self.plot_with_fits(results, strategy)
        return results

    def plot_synchronization_cliff(self, results, strategy, mut_engine):
        """
        Plots Robustness vs Complexity and highlights special rules.
        Also generates an interactive HTML plot with hover details.
        """
        import pandas as pd
        import plotly.express as px
        import os

        Xs = results['complexities']
        Ys = results['robustness']
        rules = results['rules']
        async_ks = results['async_complexities']

        # 1. Matplotlib Static Plot
        plt.figure(figsize=(10, 7))
        plt.scatter(Xs, Ys, alpha=0.6, c='royalblue', edgecolors='black', linewidth=0.5, label='Rules')

        plt.xlabel(f"Control Phenotype Complexity ({self.metric.name()})")
        plt.ylabel("Robustness (1.0 = Metric Maintained)")
        plt.title(f"Cliff: {strategy.name()}\nControl: {self.engine.name()} | Experiment: {mut_engine.name()}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # 2. Plotly Interactive HTML
        df = pd.DataFrame({
            'Control_Complexity': Xs,
            'Robustness': Ys,
            'Rule': rules,
            'Async_Complexity': async_ks
        })

        fig = px.scatter(
            df,
            x='Control_Complexity',
            y='Robustness',
            hover_data=['Rule', 'Control_Complexity', 'Async_Complexity', 'Robustness'],
            title=f"Interactive Cliff: {strategy.name()} (Control: {self.engine.name()})",
            template="plotly_white"
        )

        fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))

        save_dir = os.path.join("Saved Figures", "HTML")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filename = os.path.join(save_dir, f"AsyncCliff_{strategy.name().replace(' ', '_')}.html")
        fig.write_html(filename)
        print(f"Interactive HTML saved to: {filename}")

    def plot_transition_matrix(self, results, strategy, mut_engine, bins=10):
        """
        Plots a transition matrix showing the probability of moving from 
        a given Control Complexity Bin to an Asynchronous Complexity Bin.
        """
        import seaborn as sns

        sync_k = results['complexities']
        async_k = results['async_complexities']

        # Define bin edges based on the min/max of both sets to ensure square matrix
        min_k = min(sync_k.min(), async_k.min())
        max_k = max(sync_k.max(), async_k.max())

        bin_edges = np.linspace(min_k, max_k, bins + 1)

        # Create a 2D histogram
        matrix, xedges, yedges = np.histogram2d(sync_k, async_k, bins=bin_edges)

        # Normalize by row (Sync K bins) heavily so rows sum to 1 (transition probabilities)
        # Avoid divide-by-zero
        row_sums = matrix.sum(axis=1)
        # If a row sum is 0 (no data in that bin), just leave it as 0
        norm_matrix = np.divide(matrix.T, row_sums, out=np.zeros_like(matrix.T), where=row_sums != 0).T

        plt.figure(figsize=(8, 6))
        sns.heatmap(norm_matrix.T, annot=True, fmt=".2f", cmap="YlOrBr",
                    xticklabels=np.round((xedges[:-1] + xedges[1:]) / 2, 1),
                    yticklabels=np.round((yedges[:-1] + yedges[1:]) / 2, 1))

        plt.gca().invert_yaxis()  # Origin at bottom-left
        plt.xlabel("Control Complexity")
        plt.ylabel("Asynchronous Complexity")
        plt.title(f"Transition Matrix: Control vs Async ({strategy.name()})\nProb(Async | Control)")
        plt.tight_layout()
        plt.show()

    def plot_with_fits(self, results, strategy):
        """
        Generates separate graphs for each fitting model.
        """
        Xs = results['complexities']
        Ys = results['robustness']

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
                    plt.title(
                        f"Quantile Log-Linear ({int(q * 100)}th) - Slope: {res.params['X']:.2f}, Mutation: {strategy.name()}")

                elif model == "Logistic Decay":
                    # y = L / (1 + exp(-k(x-x0)))
                    def logistic_func(x, L, k, x0):
                        return L / (1 + np.exp(-k * (x - x0)))

                    p0 = [np.max(Ys), -0.1, np.mean(Xs)]
                    popt, _ = curve_fit(logistic_func, Xs, Ys, p0=p0, maxfev=5000)
                    plt.plot(Xs_sorted, logistic_func(Xs_sorted, *popt), 'm-', lw=2, label='Logistic Decay')

                elif model == "Isotonic":
                    ir = IsotonicRegression(increasing=False, out_of_bounds='clip')
                    ir.fit(Xs, Ys)
                    plt.plot(Xs_sorted, ir.predict(Xs_sorted), 'orange', lw=2, label='Isotonic (decreasing)')

            except Exception as e:
                plt.title(f"{model} Fit Failed: {str(e)}")

            plt.xlabel(f"Phenotype Complexity ({self.metric.name()})")
            plt.ylabel("Async Robustness")
            plt.title(f"Robustness vs Complexity: {model} model\n{strategy.name()}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

    # Abstract methods from BaseExperiment
    def plot(self, data):
        pass

    def analyze(self):
        pass


if __name__ == "__main__":
    from Core.ComplexityMeasures.complexity import ZlibComplexity
    from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
    from Core import strategies
    from Experiments.EvolutionExperiments.RobustnessAndStructureExperiments.GeneralAsynchronyExperiment import \
        GeneralAsynchronyExperiment

    engine = ElementaryCA(L=128, T=128)
    metric = ZlibComplexity()
    exp = GeneralAsynchronyExperiment(engine, metric)

    # Runs with default _simulate_1d_numba_async and SameSeedAndRuleStrategy
    exp.run(num_seeds=5)
