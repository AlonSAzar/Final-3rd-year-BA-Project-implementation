import pandas as pd
import plotly.express as px
import numpy as np


def array_to_ascii(arr, width=50):
    """
    Converts a binary array to a string.
    Injects <br> (HTML line breaks) every 'width' characters
    to prevent Plotly from truncating the tooltip.
    """
    arr = arr.flatten()
    # 1. Create the full string
    s = "".join(["#" if x else "." for x in arr])

    # 2. Split into chunks if it's too long (e.g., L=64 or L=128)
    if len(s) > width:
        # Python magic to split string into chunks
        chunks = [s[i:i + width] for i in range(0, len(s), width)]
        return "<br>".join(chunks)

    return s


def plot_interactive_scatter(phenotype_cache, transitions, coords, title="Interactive Plot", creations=None):
    print(f"Generating interactive plot for: {title}...")

    unique_hashes = list(phenotype_cache.keys())

    # 1. Calculate Frequency
    if creations is not None:
        frequencies = creations
    else:
        frequencies = {h: 0 for h in unique_hashes}
        for (h_parent, h_child), count in transitions.items():
            frequencies[h_parent] += count
            frequencies[h_child] += count

    total_freq = 0

    # 2. Build DataFrame
    data = []
    for i, h in enumerate(unique_hashes):
        img = phenotype_cache[h]
        freq = frequencies[h]
        total_freq += freq
        log_freq = np.log10(freq + 1)

        # Use the NEW function with line breaks
        visual = array_to_ascii(img, width=40)

        density = np.sum(img) / img.size

        data.append({
            "X": coords[i, 0],
            "Y": coords[i, 1],
            "LogFreq": log_freq,
            "Pattern": visual,
            "Density": f"{density:.2f}",
            "Raw_Freq": freq
        })

    df = pd.DataFrame(data)

    # 3. Create Plot
    fig = px.scatter(
        df,
        x="X",
        y="Y",
        color="LogFreq",
        # Add the data we want to hover over
        hover_data={"Pattern": True, "Density": True, "Raw_Freq": True, "X": False, "Y": False},
        title=title,
        color_continuous_scale="Viridis",
        template="plotly_white",
        width=1000,
        height=800
    )

    # 4. KEY FIX: Style the layout to allow long labels
    fig.update_layout(
        hoverlabel=dict(
            align="left",  # Align text to left
            font_size=12,  # Keep font readable
            namelength=-1  # Prevent truncation of field names
        )
    )

    fig.update_traces(marker=dict(size=6, opacity=0.7, line=dict(width=0.5, color='DarkSlateGrey')))
    if creations is not None:
        print(f"Total frequencies are {total_freq}")
    
    filename = f"{title.replace(' ', '_')}.html"
    fig.write_html(filename)
    print(f"Saved: {filename}")


def plot_interactive_robustness(Xs, Ys, imgs, rules_ids=None, title="Interactive Robustness Plot"):
    """
    Creates an interactive Plotly scatter plot for Robustness vs Complexity.
    Shows the phenotype pattern on hover.
    """
    import pandas as pd
    import plotly.express as px
    
    # Xs: Complexity, Ys: Robustness, imgs: List of phenotype images
    data = []
    for i in range(len(Xs)):
        img = imgs[i]
        # Use existing utility to convert image to ASCII for tooltip
        visual = array_to_ascii(img, width=40)
        
        row = {
            "Complexity": Xs[i],
            "Robustness": Ys[i],
            "Pattern": visual,
        }
        if rules_ids is not None:
            row["Rule"] = rules_ids[i]
            
        data.append(row)
        
    df = pd.DataFrame(data)
    
    hover_dict = {"Pattern": True, "Complexity": ":.2f", "Robustness": ":.3f"}
    if rules_ids is not None:
        hover_dict["Rule"] = True

    fig = px.scatter(
        df,
        x="Complexity",
        y="Robustness",
        hover_data=hover_dict,
        title=title,
        color="Robustness",
        color_continuous_scale="Viridis",
        template="plotly_white",
        width=1000,
        height=800
    )
    
    fig.update_layout(
        hoverlabel=dict(
            align="left",
            font_size=12,
            font_family="monospace",
            namelength=-1
        )
    )
    
    filename = f"Interactive_Robustness_{title.replace(' ', '_')}.html"
    fig.write_html(filename)
    print(f"Saved interactive plot to: {filename}")
