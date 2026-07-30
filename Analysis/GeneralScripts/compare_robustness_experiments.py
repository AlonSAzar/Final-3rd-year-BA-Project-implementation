"""
Run multiple RobustnessExperiment configurations and plot Robustness vs Complexity
for all runs on a single unified scatter, color-coded per experiment.

Usage (from repo root):
    python Analysis/compare_robustness_experiments.py --num_seeds 20 --L 12 --T 64 --out combined.png
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.Engines.OneDimensionEngines.OneDimensionFrozenCellsEngine import FrozenCellsCA
from Core.Engines.OneDimensionEngines.OneDimensionNoiseEngine import RandomNoiseCA
from Core.Engines.OneDimensionEngines.OneDimensionAsynchronousEngine import AsynchronousCA
from Core import strategies
from Core.ComplexityMeasures.complexity import ZlibComplexity
from Experiments.BasicComplexityExperiments.RobustnessExperiment import RobustnessExperiment


def run_and_collect(engine, metric, strategy, num_seeds=20, shuffle_control=False, mut_engine=None, random_iterations=20):
    rob = RobustnessExperiment(engine, metric, random_iterations)
    scores, complexities = rob.run(strategy=strategy, num_seeds=num_seeds, shuffle_control=shuffle_control, mut_engine=mut_engine)
    return scores, complexities


def align_xy(robustness_scores, phenotype_complexities):
    # Align by sorted rule id
    rules = sorted(robustness_scores.keys())
    Xs = np.array([phenotype_complexities[r] for r in rules], dtype=float)
    Ys = np.array([robustness_scores[r] for r in rules], dtype=float)
    return Xs, Ys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--T", type=int, default=6)
    parser.add_argument("--num_seeds", type=int, default=80)
    parser.add_argument("--random_iterations", type=int, default=20)
    parser.add_argument("--out", type=str, default=None, help="Path to save combined plot")
    parser.add_argument("--smoothing", type=float, default=None, help="Smoothing factor for spline fits (larger = smoother). If omitted a heuristic is used.")
    args = parser.parse_args()

    # Engines
    engine = ElementaryCA(args.L, args.T)
    frozen_engine = FrozenCellsCA(0.1, args.L, args.T)
    random_noise_engine = RandomNoiseCA(0.1, args.L, args.T)
    asynchronous_engine = AsynchronousCA(args.L, args.T)

    metric = ZlibComplexity()

    # Define experiment configurations (label, strategy_instance, mut_engine, shuffle_control)
    configs = [
        ("SameParams - Frozen", strategies.SameSeedAndRuleStrategy(), frozen_engine, False),
        ("SameParams - Noise", strategies.SameSeedAndRuleStrategy(), random_noise_engine, False),
        ("SameParams - Async", strategies.SameSeedAndRuleStrategy(), asynchronous_engine, False),
        ("Rule BitFlip", strategies.BitFlipRuleStrategy(), None, False),
        ("Rule BitFlip (Shuffled)", strategies.BitFlipRuleStrategy(), None, True),
        ("Seed BitFlip", strategies.BitFlipSeedStrategy(), None, False),
        ("Random Seed", strategies.RandomSeedStrategy(), None, False),
        ("Random Rule", strategies.RandomRuleStrategy(), None, False),
        ("Random Seed & Rule", strategies.RandomSeedAndRuleStrategy(), None, False),
    ]

    all_results = []

    # 1. Run all experiments one after another and collect data
    for label, strat, mut_eng, shuffle_flag in configs:
        print(f"Running: {label}")
        # Run but suppress the internal plot from RobustnessExperiment
        # Note: We need a way to prevent RobustnessExperiment from calling plt.show()
        # For now we'll just collect the data. If the experiment class forces show(), 
        # we might need to patch it or just accept the popups.
        scores, comps = run_and_collect(engine, metric, strat, 
                                       num_seeds=args.num_seeds, 
                                       shuffle_control=shuffle_flag, 
                                       mut_engine=mut_eng, 
                                       random_iterations=args.random_iterations)
        Xs, Ys = align_xy(scores, comps)
        
        # correlations
        try:
            p_val, _ = pearsonr(Xs, Ys)
        except Exception:
            p_val = np.nan
        try:
            s_val, _ = spearmanr(Xs, Ys)
        except Exception:
            s_val = np.nan
            
        all_results.append({
            'label': label,
            'Xs': Xs,
            'Ys': Ys,
            'pearson': p_val,
            'spearman': s_val
        })

    # 2. Only when it's done, make one unified plot
    print("Creating unified plot...")
    plt.figure(figsize=(12, 8))
    cmap = plt.get_cmap('tab10')

    for idx, res in enumerate(all_results):
        color = cmap(idx % 10)
        plt.scatter(res['Xs'], res['Ys'], alpha=0.5, color=color, 
                    label=f"{res['label']} (ρ={res['pearson']:.2f}, σ={res['spearman']:.2f})")

    plt.xlabel(f"Phenotype Complexity ({metric.name()})")
    plt.ylabel("Robustness (Avg NCC)")
    plt.title(f"Comparison of Perturbations: Robustness vs Complexity\n(L={args.L}, T={args.T}, seeds={args.num_seeds})")
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    
    if args.out:
        plt.savefig(args.out, bbox_inches='tight')
        print(f"Saved combined plot to {args.out}")
    
    plt.show()

    # --- Second figure: smoothed trendlines only (no scatter) ---
    print("Creating smoothed trendlines (no dots)...")
    try:
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
    except ImportError:
        SVR = None

    plt.figure(figsize=(12, 8))
    for idx, res in enumerate(all_results):
        Xs = np.array(res['Xs']).reshape(-1, 1)
        Ys = np.array(res['Ys'])
        
        # 1. Basic filtering: remove NaNs/Infs
        mask = np.isfinite(Xs.flatten()) & np.isfinite(Ys)
        xs_clean = Xs[mask]
        ys_clean = Ys[mask]
        
        # New: Remove the leftmost data point (lowest complexity) if it's considered an outlier/artifact
        if len(xs_clean) > 5:
            # Re-sort to find the true leftmost point
            sort_idx = np.argsort(xs_clean.flatten())
            # Drop the first 10 indexes (lowest x)
            xs_clean = xs_clean[sort_idx][1:]
            ys_clean = ys_clean[sort_idx][1:]
        
        if len(xs_clean) < 5:
            print(f"Skipping {res['label']}: too few points ({len(xs_clean)})")
            continue

        color = cmap(idx % 10)

        # 2. Grid for evaluation - Use original data range only
        x_min, x_max = xs_clean.min(), xs_clean.max()
        x_grid = np.linspace(x_min, x_max, 500)

        # 3. Fitting
        # Simply use a 3rd degree polynomial as requested.
        try:
            coeffs = np.polyfit(xs_clean.flatten(), ys_clean, 3)
            poly = np.poly1d(coeffs)
            y_grid = poly(x_grid)
            
            # Clip y_grid to [0, 1] as NCC/Robustness is logically bounded
            y_grid = np.clip(y_grid, 0, 1)
            
            plt.plot(x_grid, y_grid, color=color, linewidth=3, label=f"{res['label']} (ρ={res['pearson']:.2f})")
            
        except Exception as e:
            print(f"Polynomial fit failed for {res['label']}: {e}")
            continue

    plt.xlabel(f"Phenotype Complexity ({metric.name()})")
    plt.ylabel("Robustness (Avg NCC)")
    plt.title(f"Comparison of Perturbations (ML-Smoothed Trendlines)\n(L={args.L}, T={args.T}, num_seeds={args.num_seeds})")
    plt.grid(True, alpha=0.3)
    
    if plt.gca().get_lines():
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        if args.out:
            base, ext = os.path.splitext(args.out)
            smooth_out = f"{base}_smooth{ext if ext else '.png'}"
            plt.savefig(smooth_out, bbox_inches='tight')
            print(f"Saved ML-trendline plot to {smooth_out}")
        plt.show()
    else:
        print("Warning: No lines were generated for the trendline plot.")
        plt.close()


if __name__ == "__main__":

    main()
