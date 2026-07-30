from collections import Counter

from tqdm import tqdm

from Experiments.experiments import *
from Visualizations import utils


class SimplicityBiasExperiment(BaseExperiment):
    """
    Tests the AIT hypothesis: P(x) ~ 2^-K(x).
    Generates distribution plots.
    """

    def __init__(self, engine: ElementaryCA, metric: ComplexityMetric):
        super().__init__(engine, metric)
        self.results = {}  # rule -> list of images
        self.num_seeds_used = 0

    def run(self, num_seeds: int, rules=range(256), shuffle_control=False, **kwargs):
        """Generates data."""
        self.num_seeds_used = num_seeds
        
        if not kwargs.get('skip_simulation', False):
            # If we want to run on ALL possible seeds precisely once
            if kwargs.get('use_all_seeds', False):
                L = self.engine.L
                seeds = []
                for i in range(2**L):
                    # Convert integer i to binary array of length L
                    seed_bits = [(i >> bit) & 1 for bit in range(L - 1, -1, -1)]
                    seeds.append(np.array(seed_bits, dtype=np.uint8))
                self.num_seeds_used = len(seeds)
            else:
                seeds = [self.engine.generate_seed() for _ in range(num_seeds)]

            # tqdm displays a progress bar in the console
            for rule in tqdm(rules, desc="Simulating Rules"):
                phenotypes = []
                for seed in seeds:
                    # Discard t=0 (seed) usually
                    img = self.engine.run(rule, seed)[1:]
                    phenotypes.append(img)
                self.results[rule] = phenotypes

        if kwargs.get('just_distribute', False):
            return

        freqs, comps, hash_to_img = self.analyze(shuffle_control)
        
        # Determine title based on shuffle state
        title_suffix = " (Shuffled Control)" if shuffle_control else ""
        base_title = f"Simplicity Bias Experiment{title_suffix}"

        if kwargs.get('annotated', False):
            from Visualizations.annotated_plotter import AnnotatedSimplicityBiasPlotter
            plotter = AnnotatedSimplicityBiasPlotter(self.metric, self.engine, num_seeds)
            plotter.plot(freqs, comps, hash_to_img, title=base_title, **kwargs)
        else:
            upr_bound_pltr = utils.UpperBoundPlotter(self.metric, self.engine, num_seeds)
            upr_bound_pltr.plot(freqs, comps, title=base_title, **kwargs)

    def analyze(self, shuffle_control=False):
        """
        Hashes phenotypes and calculates complexity.
        returns: freq_map, complexity_map, hash_to_img
            freq_map: Hash of image -> count
            complexity_map: Dict hash of image -> complexity score
            hash_to_img: Dict hash of image -> actual image array
            """
        # Dict that will store the count of each unique phenotype.
        # Hash of image -> count
        freq_map = Counter()
        # Dict hash of image -> complexity score
        complexity_map = {}
        # Dict hash of image -> actual image array
        hash_to_img = {}

        # Flatten all results (which are a list of lists) into one big pool of phenotypes
        all_images = [img for sublist in self.results.values() for img in sublist]

        for img in tqdm(all_images, desc="Analyzing Complexity"):
            # Optional: Shuffle image to test control
            if shuffle_control:
                img = shuffle_space_time(img)

            h = img.tobytes()  # Hash
            freq_map[h] += 1

            if h not in complexity_map:
                complexity_map[h] = self.metric.calculate(img)
                hash_to_img[h] = img

        return freq_map, complexity_map, hash_to_img

    # TODO maybe delete from the abstract class
    # In this case, the plotting is handled by utilities.UpperBoundPlotter
    def plot(self): pass