import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, HPacker, TextArea
from sklearn.linear_model import QuantileRegressor
from scipy.stats import linregress
import os

class AnnotatedSimplicityBiasPlotter:
    def __init__(self, metric, engine, num_seeds_used):
        self.metric = metric
        self.engine = engine
        self.num_seeds_used = num_seeds_used

    def plot(self, freq_map, complexity_map, hash_to_img, title, num_annotations=3, **kwargs):
        """
        Visualizes the Simplicity Bias with annotated phenotype examples.
        """
        Ks = []
        log_probs = []
        hashes = []
        total = sum(freq_map.values())

        for h, count in freq_map.items():
            Ks.append(complexity_map[h])
            log_probs.append(np.log10(count / total))
            hashes.append(h)

        Ks = np.array(Ks)
        log_probs = np.array(log_probs)

        # Plot setup
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Calculate correlations for legend
        from scipy.stats import spearmanr
        spearman_label = 'Phenotypes'
        if len(Ks) > 1:
            spearman_corr, _ = spearmanr(Ks, log_probs)
            spearman_label += f' (Spearman: {spearman_corr:.3f})'

        # Consistent markers and colors with the standard plotter
        ax.scatter(Ks, log_probs, alpha=0.2, c='teal', label='Phenotypes')

        # Fit upper bound with Quantile Regression (0.95 quantile)
        unique_ks = np.unique(Ks)
        max_log_probs = np.array([np.max(log_probs[Ks == k]) for k in unique_ks])
        slope, intercept = 0, 0
        if len(unique_ks) > 1:
            qr = QuantileRegressor(quantile=0.95, alpha=0)
            qr.fit(unique_ks.reshape(-1, 1), max_log_probs)
            slope = qr.coef_[0]
            intercept = qr.intercept_
            x_line = np.linspace(min(Ks), max(Ks), 100)
            ax.plot(x_line, slope * x_line + intercept, color='red', linestyle='--', linewidth=2, label='Upper Bound Fit')

        # Calculate a/b parameters for the stats box
        log2_10 = np.log2(10)
        a_param = -slope * log2_10
        b_param = -intercept * log2_10

        # Set consistent axes if provided
        if 'xlim' in kwargs:
            ax.set_xlim(kwargs['xlim'])
        if 'ylim' in kwargs:
            ax.set_ylim(kwargs['ylim'])

        # Info Box / Legend
        from scipy.stats import pearsonr, spearmanr
        p_corr, _ = pearsonr(Ks, log_probs) if len(Ks) > 1 else (0, 0)
        s_corr, _ = spearmanr(Ks, log_probs) if len(Ks) > 1 else (0, 0)

        if not kwargs.get('hide_stats', False):
            stats_text = (
                f"Simulation Parameters:\n"
                f"  N_seeds = {self.num_seeds_used}\n\n"
                f"Correlations:\n"
                f"  Spearman = {s_corr:.3f}\n"
                f"  Pearson = {p_corr:.3f}\n\n"
                f"Fit P = 2^(-aK - b):\n"
                f"  a = {a_param:.3f}\n"
                f"  b = {b_param:.3f}"
            )
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            # Position it at 0.95 relative to the axes
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=16,
                     verticalalignment='top', horizontalalignment='right', bbox=props, zorder=10)
        else:
            # For presentation, put Spearman Correlation in the top right
            corr_text = f"Spearman Correlation: {s_corr:.3f}"
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(0.95, 0.95, corr_text, transform=ax.transAxes, fontsize=20,
                     verticalalignment='top', horizontalalignment='right', bbox=props, zorder=10)

        # Legend at top right, below correlation values
        ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.8))

        # Annotation Logic: Pick phenotypes at extreme and middle complexities
        # For the middle (blue) point, we want something above minimum frequency
        min_lp = np.min(log_probs)
        interesting_indices = np.where(log_probs > min_lp)[0]
        
        # Determine sampling indices
        # Red/Green can come from entire set, Blue should be from interesting_indices if possible
        all_sorted_indices = np.argsort(Ks)
        
        if len(all_sorted_indices) >= num_annotations:
            # 1. Red (Simple): Lowest complexity from everything
            idx_red = all_sorted_indices[0]
            
            # 2. Green (Complex): Highest complexity from everything
            idx_green = all_sorted_indices[-1]
            
            # 3. Blue (Middle): Median complexity specifically from things ABOVE minimum frequency
            if len(interesting_indices) > 0:
                interesting_sorted = interesting_indices[np.argsort(Ks[interesting_indices])]
                idx_blue = interesting_sorted[len(interesting_sorted) // 2]
            else:
                idx_blue = all_sorted_indices[len(all_sorted_indices) // 2]
                
            annotation_indices = [idx_red, idx_blue, idx_green]
        else:
            annotation_indices = all_sorted_indices

        # Define marker and arrow colors for annotations to avoid confusion with the red fit line
        annotation_colors = ['darkorange', 'blue', 'green', 'purple', 'brown']
        
        # Adjust subplot to make the plot itself smaller and create much more room on the right
        plt.subplots_adjust(right=0.55, left=0.1)
        
        for i, idx in enumerate(annotation_indices):
            k = Ks[idx]
            lp = log_probs[idx]
            h = hashes[idx]
            img = hash_to_img[h]
            color = annotation_colors[i % len(annotation_colors)]

            # Highlight the point
            ax.scatter(k, lp, edgecolors=color, facecolors='none', s=200, lw=3, zorder=5)

            # Save individual images for animation/presentation if requested
            if kwargs.get('save_examples', False):
                os.makedirs("Presentation_Assets", exist_ok=True)
                plt.imsave(f"Presentation_Assets/phenotype_{i}_K{k:.1f}.png", img, cmap='binary')

            # Prepare image: rotate
            img_rotated = np.rot90(img, k=1) 
            
            # Use zoom to make images significantly bigger for poster
            image_box = OffsetImage(img_rotated, cmap='binary', zoom=5.0)
            
            # Adjust connection radius to be straighter or arc differently to avoid text
            if i == 0:
                rad = -0.1
            else:
                rad = -0.1
            
            ab = AnnotationBbox(image_box, (k, lp),
                                xybox=(1.5, 0.82 - i*0.35),
                                xycoords='data',
                                boxcoords="axes fraction",
                                pad=0.3,
                                arrowprops=dict(arrowstyle="->", color=color, lw=2.5, 
                                               connectionstyle=f"arc3,rad={rad}"))
            ax.add_artist(ab)
            
            # Label the complexity/probability near the image
            ax.annotate(f"K={k:.1f}\nlogP={lp:.2f}", 
                        xy=(1.15, 0.92 - i*0.35), 
                        xycoords='axes fraction',
                        color=color, fontweight='bold', fontsize=20,
                        va='bottom', ha='left')

        ax.set_xlabel("Estimated Kolmogorov Complexity (Zlib Bytes)")
        ax.set_ylabel("Log10 Probability")
        ax.set_title(f"{title} (L={self.engine.L}, T={self.engine.T})", pad=20)
        ax.grid(True, alpha=0.3)
        # plt.tight_layout() is removed to preserve subplots_adjust parameters
        plt.show()

    def plot_csb(self, Ks, log_probs, transition_map, strategy, shuffle_control, **kwargs):
        """
        Specialized plotter for Conditional Simplicity Bias that shows PAIRS of phenotypes (x -> y).
        """
        Ks = np.array(Ks)
        log_probs = np.array(log_probs)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.scatter(Ks, log_probs, alpha=0.2, c='darkcyan', label='Transitions')

        # Upper bound fit
        unique_ks = np.unique(Ks)
        max_log_probs = np.array([np.max(log_probs[Ks == k]) for k in unique_ks])
        if len(unique_ks) > 1:
            qr = QuantileRegressor(quantile=0.95, alpha=0)
            qr.fit(unique_ks.reshape(-1, 1), max_log_probs)
            slope, intercept = qr.coef_[0], qr.intercept_
            x_line = np.linspace(min(Ks), max(Ks), 100)
            ax.plot(x_line, slope * x_line + intercept, color='red', linestyle='--', lw=2, label='Upper Bound Fit')

        if 'xlim' in kwargs: ax.set_xlim(kwargs['xlim'])
        if 'ylim' in kwargs: ax.set_ylim(kwargs['ylim'])

        # Info Box / Legend
        from scipy.stats import spearmanr
        s_corr, _ = spearmanr(Ks, log_probs) if len(Ks) > 1 else (0, 0)
        
        # When annotations are hidden (clean plot for presentation), 
        # put Spearman in a separate small box at top-right
        if kwargs.get('num_annotations') == 0:
            corr_text = f"Spearman Correlation: {s_corr:.3f}"
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(0.95, 0.95, corr_text, transform=ax.transAxes, fontsize=20,
                     verticalalignment='top', horizontalalignment='right', bbox=props, zorder=10)

        # Legend at top right, below correlation values
        ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.8))

        # Annotations
        num_annotations = kwargs.get('num_annotations', 3)
        if num_annotations == 0:
            indices = []
        else:
            all_sorted = np.argsort(Ks)
            if len(all_sorted) >= num_annotations:
                indices = [all_sorted[0], all_sorted[len(all_sorted)//2], all_sorted[-1]]
            else:
                indices = all_sorted

        plt.subplots_adjust(right=0.55, left=0.1)
        colors = ['darkorange', 'blue', 'green']
        
        for i, idx in enumerate(indices):
            k, lp = Ks[idx], log_probs[idx]
            img_x, img_y = transition_map[idx]
            color = colors[i % len(colors)]
            
            ax.scatter(k, lp, edgecolors=color, facecolors='none', s=250, lw=4, zorder=10)
            
            # Create a combined image (Parent on top, Mutant on bottom)
            # Both need to be rotated for consistency
            img_x_rot = np.rot90(img_x, k=1)
            img_y_rot = np.rot90(img_y, k=1)
            
            h, w = img_x_rot.shape
            
            # Start with an all-white RGB background
            parent_rgb = np.ones((h, w, 3))
            parent_rgb[img_x_rot == 1] = [0, 0, 0]          # Parent black pixels
            
            mutant_rgb = np.ones((h, w, 3))
            mutant_rgb[img_y_rot == 1] = [0, 0, 0]          # Mutant black pixels
            
            # Create individual box contents for parent and mutant
            # Note: We use OffsetImage alone inside HPacker. 
            # Decorating it with ANOTHER AnnotationBbox inside an HPacker is sometimes ignored or causes blank renders in some Matplotlib versions.
            # Instead, we create a tight RGB border manually for the parent and mutant.
            
            # Parent with border
            p_h, p_w = parent_rgb.shape[:2]
            parent_with_border = np.zeros((p_h+2, p_w+2, 3)) 
            parent_with_border[:, :, 0] = 1.0 # Red channel = 1
            parent_with_border[:, :, 1] = 0.0 # Green channel = 0
            parent_with_border[:, :, 2] = 0.0 # Blue channel = 0
            parent_with_border[1:-1, 1:-1] = parent_rgb # white/black CA content
            
            # Mutant with border
            m_h, m_w = mutant_rgb.shape[:2]
            mutant_with_border = np.zeros((m_h+2, m_w+2, 3))
            mutant_with_border[:, :, 0] = 1.0 # Red
            mutant_with_border[:, :, 1] = 0.0
            mutant_with_border[:, :, 2] = 0.0
            mutant_with_border[1:-1, 1:-1] = mutant_rgb # white/black CA content
            
            ib_parent = OffsetImage(parent_with_border, zoom=5.5)
            ib_mutant = OffsetImage(mutant_with_border, zoom=5.5)
            
            # Create a right arrow
            arrow_txt = TextArea(" $\\rightarrow$ ", textprops=dict(fontsize=32, color='black', fontweight='bold'))
            
            # Pack them horizontally: [Parent] -> [Mutant]
            pack = HPacker(children=[ib_parent, arrow_txt, ib_mutant], align="center", pad=2, sep=5)
            
            # Final outer container
            ab = AnnotationBbox(pack, (k, lp), xybox=(1.6, 0.90 - i*0.35),
                                xycoords='data', boxcoords="axes fraction",
                                pad=0.1,
                                bboxprops=dict(alpha=0), # Outer box is transparent
                                arrowprops=dict(arrowstyle="->", color=color, lw=2.5, connectionstyle="arc3,rad=-0.1"))
            ax.add_artist(ab)
            
            # Adjusted vertical position (0.94 -> 0.98) and alignment to move text above the arrows
            ax.annotate(f"Transition {i+1}\nK(y|x)={k:.1f}\nlogP={lp:.2f}", 
                        xy=(1.44, 0.97 - i*0.35), xycoords='axes fraction',
                        color=color, fontweight='bold', fontsize=20, va='bottom')

        ax.set_xlabel("Conditional Complexity K(y|x) (Zlib Bytes)")
        ax.set_ylabel("Log10 P(x → y)")
        title_suff = " (Shuffled)" if shuffle_control else ""
        
        # Use a smaller pad for Title in CSB plots specifically as requested
        ax.set_title(f"Conditional Simplicity Bias{title_suff}\nStrategy: {strategy.name()}", pad=kwargs.get('title_pad', 40))
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left')
        plt.show()
