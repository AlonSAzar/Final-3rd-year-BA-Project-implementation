import numpy as np
import matplotlib.pyplot as plt
import zlib
import pandas as pd
import plotly.express as px
from tqdm import tqdm
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Visualizations.interactive_utils import array_to_ascii

def calculate_structural_info(L=128, T=128, samples_per_rule=5, interactive=True):
    engine = ElementaryCA(L, T)

    data = []

    print("Calculating Structural Information for all rules...")
    for rule in tqdm(range(256)):
        for _ in range(samples_per_rule):
            seed = engine.generate_seed()
            # Get the final 50 rows (the attractor)
            img = engine.run(rule, seed)[-50:]

            # 1. Calculate Density (X-axis)
            density = np.mean(img)

            # 2. Calculate K_raw
            bytes_raw = img.tobytes()
            k_raw = len(zlib.compress(bytes_raw, level=9))

            # 3. Calculate K_shuffled
            # Flatten to destroy local spatial grammar
            flat_img = img.flatten()
            np.random.shuffle(flat_img)
            shuffled_img = flat_img.reshape(img.shape)

            bytes_shuf = shuffled_img.tobytes()
            k_shuf = len(zlib.compress(bytes_shuf, level=9))

            # 4. Structural Information (Y-axis)
            # How much compression was lost when we destroyed the spatial grammar?
            delta_k = k_shuf - k_raw

            data.append({
                "Rule": rule,
                "Density": float(density),
                "Structural_Info": int(delta_k),
                "K_raw": int(k_raw),
                "K_shuffled": int(k_shuf),
                "Pattern": array_to_ascii(img, width=L)
            })

    df = pd.DataFrame(data)

    if interactive:
        fig = px.scatter(
            df,
            x="Density",
            y="Structural_Info",
            color="Rule",
            hover_data={
                "Rule": True,
                "Density": ":.3f",
                "Structural_Info": True,
                "K_raw": True,
                "K_shuffled": True,
                "Pattern": True
            },
            title=f"Interactive Structural Information vs. Density (L={L}, T={T})",
            template="plotly_white",
            labels={"Structural_Info": "Delta K (Structural Info)"},
            color_continuous_scale="Viridis"
        )

        fig.update_layout(
            hoverlabel=dict(
                align="left",
                font_size=10,
                namelength=-1,
                bgcolor="white"
            )
        )

        output_filename = "Structural_Information_Interactive.html"
        fig.write_html(output_filename)
        print(f"Saved interactive plot to {output_filename}")

        # Summary statistics
        avg_si = df["Structural_Info"].mean()
        print(f"\nAverage Structural Information (Delta K): {avg_si:.2f} bytes")

        # Try to show if in interactive environment
        try:
            fig.show()
        except:
            pass
    else:
        # --- Static Matplotlib Plotting ---
        plt.figure(figsize=(12, 7))
        sc = plt.scatter(df["Density"], df["Structural_Info"], alpha=0.5, c=df["Rule"], cmap='viridis', edgecolor='k', s=20, label='CA Phenotypes')
        plt.axhline(0, color='red', linestyle='--', linewidth=2, label='Max Entropy Baseline')
        plt.title("Structural Information vs. Density")
        plt.xlabel("Phenotype Density")
        plt.ylabel("Structural Information Delta K (Bytes)")
        plt.colorbar(sc, label="Rule ID")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    calculate_structural_info(interactive=True)
