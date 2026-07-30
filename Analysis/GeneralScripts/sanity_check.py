import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path to resolve core imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import your optimized CA engine
from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA


def generate_and_show_ca(rule: int, width: int = 64, time_steps: int = 64, seed_type: str = "random",
                         save_fig: bool = True):
    """
    Generates a space-time diagram for a given ECA rule and displays it.

    Parameters:
        rule (int): ECA Rule Number (0-255).
        width (int): Spatial width (L).
        time_steps (int): Time duration (T).
        seed_type (str): 'random', 'center', or 'zero'.
        save_fig (bool): Whether to save the image to disk.
    """
    # 1. Initialize Engine
    engine = ElementaryCA(L=width, T=time_steps)

    # 2. Generate Seed & Run Simulation
    seed = engine.generate_seed(seed_type=seed_type)
    phenotype = engine.run(rule, seed)

    # 3. Plot Space-Time Diagram
    plt.figure(figsize=(6, 6))

    # 'gray_r' renders 1s as black cells and 0s as white cells
    plt.imshow(phenotype, cmap='gray_r', interpolation='nearest', origin='upper')

    plt.title(f"Elementary CA: Rule {rule} ({width}x{time_steps})", fontsize=12, pad=10)
    plt.xlabel("Space (L)", fontsize=10)
    plt.ylabel("Time (T)", fontsize=10)
    plt.tight_layout()

    # 4. Save Output
    if save_fig:
        save_dir = os.path.join("Analysis", "GeneralScripts", "Saved Figures")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"rule_{rule}_grid_{width}x{time_steps}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image successfully saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    # Parameters
    RULE_TO_VIEW = 30  # Change to any rule (0-255)
    L = 64
    T = 64
    SEED_TYPE = "random"  # Options: "random", "center"

    generate_and_show_ca(
        rule=RULE_TO_VIEW,
        width=L,
        time_steps=T,
        seed_type=SEED_TYPE,
        save_fig=True
    )