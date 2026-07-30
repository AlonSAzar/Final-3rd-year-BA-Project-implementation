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
    # Global storage to compare multiple runs
    session_results = []

    def __init__(self, engine: ElementaryCA, metric: ComplexityMetric, random_iterations: int = 20):
        super().__init__(engine, metric)
        self.random_iterations = random_iterations

    def run(self, strategy, num_seeds=20, shuffle_control=False, mut_engine=None, use_phenotypes=False, **kwargs):
        """
        Returns dictionaries {rule_id: robustness_score}, {rule_id, phenotype_complexity}.
        Robustness = Average NCC between Rule(seed) and MutantRule(seed).

        If use_phenotypes is True, the plot will show each individual (seed, rule) pair as a dot.
        Otherwise, it will show the average per rule (default).
        """

        if mut_engine is None:
            mut_engine = self.engine

        print(f"--- Robustness: {strategy.name()}, {mut_engine.name()} ---")
        # Setup dicts
        rule_robustness_scores = {}
        rule_images_dict = {}
        
        # For phenotype-level data
        all_phenotype_complexities = []
        all_phenotype_robustness = []

        seeds = [self.engine.generate_seed() for _ in range(num_seeds)]

        # if mut_engine is None: # Moved up
        #    mut_engine = self.engine

        for rule in tqdm(range(256), desc="Evolutionary Robustness"):
            rule_score = 0
            rule_images_list = []

            for seed in seeds:
                base_img = self.engine.run(rule, seed)[1:]
                rule_images_list.append(base_img)
                
                # Calculate phenotype complexity for this specific seed
                pheno_k = self.metric.calculate(base_img)

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
                
                seed_avg_robustness = np.mean(mutant_scores)
                rule_score += seed_avg_robustness
                
                all_phenotype_complexities.append(pheno_k)
                all_phenotype_robustness.append(seed_avg_robustness)

            # Calculate average across seeds, per rule
            rule_robustness_scores[rule] = rule_score / len(seeds)
            rule_images_dict[rule] = rule_images_list

        rule_phenotype_complexity_scores = self.mean_phenotype_complexity(rule_images_dict)

        # Determine what to plot
        if use_phenotypes:
            Xs_plot = np.array(all_phenotype_complexities)
            Ys_plot = np.array(all_phenotype_robustness)
            plot_type_str = "Phenotype-lvl"
        else:
            Xs_plot = np.array([rule_phenotype_complexity_scores[r] for r in sorted(rule_robustness_scores.keys())])
            Ys_plot = np.array([rule_robustness_scores[r] for r in sorted(rule_robustness_scores.keys())])
            plot_type_str = "Rule-lvl"

        # Store results for session comparison (always use rule-level for session consistency)
        # Handle parameters for different engines
        engine_params = ""
        if hasattr(mut_engine, 'noise_level'):
            engine_params = f"p={mut_engine.noise_level}"
        elif hasattr(mut_engine, 'locked_fraction'):
            engine_params = f"frac={mut_engine.locked_fraction}"

        RobustnessExperiment.session_results.append({
            'strategy_name': strategy.name(),
            'mut_engine_name': mut_engine.name() if mut_engine else self.engine.name(),
            'L': self.engine.L,
            'T': self.engine.T,
            'shuffle_control': shuffle_control,
            'plot_type': plot_type_str,
            'engine_params': engine_params,
            'Xs': np.array([rule_phenotype_complexity_scores[r] for r in sorted(rule_robustness_scores.keys())]),
            'Ys': np.array([rule_robustness_scores[r] for r in sorted(rule_robustness_scores.keys())])
        })

        # New model fitting and plotting
        # Identify engine parameters for the plot title
        engine_params_str = ""
        if hasattr(mut_engine, 'noise_level'):
            engine_params_str = f", p={mut_engine.noise_level}"
        elif hasattr(mut_engine, 'locked_fraction'):
            engine_params_str = f", frac={mut_engine.locked_fraction}"
        
        if shuffle_control:
            engine_params_str += " (Shuffled)"
            
        self.plot_with_fits(Xs_plot, Ys_plot, strategy, f"{mut_engine.name()}{engine_params_str}", plot_type_str, **kwargs)

        # Build image list for interactive plot
        if use_phenotypes:
            # We need the direct phenotype images in the same order as all_phenotype_complexities
            # Note: We can retrieve them during the loop or rebuild them. 
            # Rebuilding for simplicity of the edit:
            pass # See the loop change below
        else:
            # Rule images are harder to map specifically to the mean, but we can take the first seed's image
            sorted_rules = sorted(rule_robustness_scores.keys())
            interactive_imgs = [rule_images_dict[r][0] for r in sorted_rules]
            from Visualizations.interactive_utils import plot_interactive_robustness
            plot_interactive_robustness(Xs_plot, Ys_plot, interactive_imgs, rules_ids=sorted_rules, 
                                       title=f"Robustness_{strategy.name()}_{plot_type_str}")

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
        pearson_val, _ = pearsonr(Xs, Ys)
        spearman_val, _ = spearmanr(Xs, Ys)

        # 3. Plotting
        plt.figure(figsize=(14, 10))
        plt.scatter(Xs, Ys, alpha=0.6, c='teal', edgecolors='black', linewidth=0.5, s=40)

        plt.xlabel(f"Phenotype Complexity ({self.metric.name()})", fontsize=24)
        plt.ylabel("Robustness (Avg NCC)", fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        shuffle_str = ""
        if shuffle:
            shuffle_str = " Shuffled Control."

        # Pull L and T from the engine instance
        L = self.engine.L
        T = self.engine.T
        plt.title(f"1D CA Robustness VS Complexity, {mutation_type.name()}, {mut_engine.name()}.\nL={L}, T={T}.{shuffle_str}", fontsize=22, pad=10)

        # 4. Info Box with Stats
        stats_text = (
            f"Spearman = {spearman_val:.3f}\n"
            f"Pearson = {pearson_val:.3f}"
        )

        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=18,
                 verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_with_fits(self, Xs, Ys, strategy, mut_img_name, plot_type_str="", **kwargs):
        """
        Generates a plot with the Isotonic fit and correlation statistics.
        """
        import matplotlib.pyplot as plt
        from sklearn.isotonic import IsotonicRegression
        from scipy.stats import pearsonr, spearmanr

        show_fit = kwargs.get('show_fit', False)

        # Filter out NaNs
        mask = ~np.isnan(Xs) & ~np.isnan(Ys)
        Xs, Ys = Xs[mask], Ys[mask]

        if len(Xs) == 0:
            print("No valid data to plot.")
            return

        # Sort for smooth line plotting
        sort_idx = np.argsort(Xs)
        Xs_sorted = Xs[sort_idx]
        Ys_sorted = Ys[sort_idx]

        # Calculate correlations
        p_corr, p_pval = pearsonr(Xs, Ys)
        s_corr, s_pval = spearmanr(Xs, Ys)

        plt.figure(figsize=(14, 10))
        plt.scatter(Xs, Ys, alpha=0.5, c='dodgerblue', edgecolors='navy', linewidth=0.5, label='Rule', s=75)

        # Isotonic Regression (decreasing)
        if show_fit:
            try:
                ir = IsotonicRegression(increasing=False, out_of_bounds='clip')
                ir.fit(Xs, Ys)
                plt.plot(Xs_sorted, ir.predict(Xs_sorted), color='red', lw=4, label=f'Isotonic Fit: {strategy.name()}')
            except Exception as e:
                plt.title(f"Isotonic Fit Failed: {str(e)}")

        plt.xlabel(f"Complexity ({self.metric.name()})", fontsize=24)
        plt.ylabel("Robustness (Avg NCC)", fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        
        # Pull L and T from the engine instance
        L = self.engine.L
        T = self.engine.T
        
        plt.title(f"1D CA Robustness VS Complexity: {strategy.name()}\n"
                  f"L={L}, T={T}, {mut_img_name}. {plot_type_str}", fontsize=22, pad=10)
        
        # Info Box with Stats (matching the requested style)
        stats_text = (
            f"Spearman = {s_corr:.3f}\n"
            f"Pearson = {p_corr:.3f}"
        )

        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=18,
                 verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.legend(fontsize=18, loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.show()

    @classmethod
    def plot_session_comparison(cls):
        """Plots the isotonic fits from all runs in the session for comparison."""
        import matplotlib.pyplot as plt
        from sklearn.isotonic import IsotonicRegression
        
        if not cls.session_results:
            print("No session results to compare.")
            return

        plt.figure(figsize=(12, 8))
        
        # Use a qualitative colormap for distinct colors
        cmap = plt.get_cmap('tab10')
        
        for i, res in enumerate(cls.session_results):
            name = f"{res['strategy_name']}"
            if res['mut_engine_name'] and res['mut_engine_name'] != "Elementary CA":
                name += f" ({res['mut_engine_name']}"
                if res.get('engine_params'):
                    name += f": {res['engine_params']}"
                name += ")"
            
            if res.get('shuffle_control'):
                name += " [SHUFFLED]"
            
            # Include L, T and plot level in the legend
            name += f" [{res.get('plot_type', 'Rule-lvl')}]"

            Xs = res['Xs']
            Ys = res['Ys']
            
            # Filter NaNs
            mask = ~np.isnan(Xs) & ~np.isnan(Ys)
            Xs, Ys = Xs[mask], Ys[mask]
            
            if len(Xs) == 0: continue

            # Sort for fitting
            idx = np.argsort(Xs)
            Xs_fit = Xs[idx]
            Ys_fit = Ys[idx]
            
            # Isotonic fit
            try:
                ir = IsotonicRegression(increasing=False, out_of_bounds='clip')
                y_ir = ir.fit_transform(Xs_fit, Ys_fit)
                plt.plot(Xs_fit, y_ir, label=name, linewidth=2, color=cmap(i % 10))
            except:
                continue

        plt.xlabel('Mean Phenotype Complexity')
        plt.ylabel('Robustness Score (Avg NCC)')
        
        # Pull L and T from the first result (assuming consistency as requested)
        L_first = cls.session_results[0].get('L', '?')
        T_first = cls.session_results[0].get('T', '?')
        plt.title(f'Comparison of Isotonic Robustness Fits\nL={L_first}, T={T_first}')
        
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
    def plot(self): pass
    def analyze(self): pass
