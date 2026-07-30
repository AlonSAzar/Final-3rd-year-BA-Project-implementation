import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.ComplexityMeasures.conditional_complexity import ZlibConditionalComplexity
from Experiments.ConditionalComplexityExperiments.ConditionalTransitionExperiment import ConditionalTransitionExperiment
from Core import strategies
from Visualizations.interactive_utils import array_to_ascii
import pandas as pd
import plotly.express as px

def create_interactive_csb_plot():
    """
    Runs a seed bit-flip CSB experiment (L=T=6) across all rules
    and generates an interactive Plotly plot showing parent-child patterns on hover.
    """
    L, T = 6, 6
    NUM_PARENTS = 2**L # Exhaustive search for L=6
    
    engine = ElementaryCA(L, T)
    metric = ZlibConditionalComplexity()
    # Strategy specified: Seed bit-flip
    strategy = strategies.BitFlipSeedStrategy()
    
    print(f"Running Interactive CSB (Seed-Flip, L={L}, T={T})...")
    csb_exp = ConditionalTransitionExperiment(engine, metric, last_row_only=False)
    
    # Run simulation
    csb_exp.run(rules=range(256), num_parents=NUM_PARENTS, strategy=strategy, use_all_seeds=True)
    
    # Process results for Plotly
    data = []
    
    # Helper to map parent/child indices to rule IDs
    # Since we iterated rules then seeds, we can't easily recover rule from transition Counter keys
    # Instead, let's rebuild the data specifically for this plotting task
    
    print("Pre-processing data for Plotly...")
    total_mutations_per_parent = {}
    for (h_x, h_y), count in csb_exp.transitions.items():
        total_mutations_per_parent[h_x] = total_mutations_per_parent.get(h_x, 0) + count

    # We want a plot of K(y|x) vs log10(P(y|x))
    # Note: Multiple transitions might have the same K and P, so we group them.
    # However, for 'Rule Number' visibility, we might want to keep more info.
    
    for (h_x, h_y), count in csb_exp.transitions.items():
        img_x = csb_exp.phenotype_cache[h_x]
        img_y = csb_exp.phenotype_cache[h_y]
        
        k_val = metric.calculate(img_x, img_y)
        prob = count / total_mutations_per_parent[h_x]
        
        # Prepare visuals
        parent_visual = array_to_ascii(np.rot90(img_x, k=1), width=T)
        child_visual = array_to_ascii(np.rot90(img_y, k=1), width=T)
        
        data.append({
            "Conditional Complexity K(y|x)": k_val,
            "Log10 Probability": np.log10(prob),
            "Parent Pattern": parent_visual,
            "Child Pattern": child_visual,
            "Raw Count": count
        })

    df = pd.DataFrame(data)
    
    title = f"Interactive CSB (Seed-Flip, L={L}, T={T})"
    fig = px.scatter(
        df,
        x="Conditional Complexity K(y|x)",
        y="Log10 Probability",
        hover_data={
            "Parent Pattern": True, 
            "Child Pattern": True, 
            "Log10 Probability": ":.3f"
        },
        title=title,
        color="Log10 Probability",
        color_continuous_scale="Viridis",
        template="plotly_white",
        width=1100,
        height=850
    )

    fig.update_layout(
        hoverlabel=dict(
            align="left",
            font_size=10,
            font_family="monospace",
            namelength=-1
        )
    )

    filename = "Interactive_CSB_SeedFlip.html"
    fig.write_html(filename)
    print(f"Interactive plot saved to: {filename}")
    
    # Also show correlations in terminal
    from scipy.stats import spearmanr
    s_corr, _ = spearmanr(df["Conditional Complexity K(y|x)"], df["Log10 Probability"])
    print(f"Finished. Spearman Correlation: {s_corr:.4f}")

if __name__ == "__main__":
    create_interactive_csb_plot()
