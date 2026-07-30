import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.strategies import BitFlipSeedStrategy
from Experiments.experiments import compute_ncc

def generate_ncc_examples(L=12, T=64, rules=range(256)):
    """
    Generates 3 examples of seed perturbations and their NCC scores:
    High NCC (~1.0), Medium NCC (~0.5), and Low NCC (~0.0).
    Iterates through all provided rules to find the best examples.
    """
    engine = ElementaryCA(L=L, T=T)
    strategy = BitFlipSeedStrategy()
    
    print(f"Generating NCC examples across {len(rules)} rules...")
    
    # We'll search for good examples
    examples = {
        'High (~1.0)': None,
        'Medium (~0.5)': None,
        'Low (~0.0)': None
    }
    
    found_count = 0
    
    # Randomize rules to avoid bias from low-numbered rules
    shuffled_rules = list(rules)
    np.random.shuffle(shuffled_rules)

    for rule in shuffled_rules:
        if found_count == 3: break
        
        # Check a few seeds per rule
        for _ in range(20):
            if found_count == 3: break
            
            seed = engine.generate_seed()
            base_img = engine.run(rule, seed)[1:]
            
            # Try a few variations for this seed
            num_vars = strategy.get_variations_count(engine, rule, seed, 0)
            for i in range(min(num_vars, 5)):
                m_rule, m_seed = strategy.apply(engine, rule, seed, i)
                mut_img = engine.run(m_rule, m_seed)[1:]
                
                ncc = compute_ncc(base_img, mut_img)
                
                if examples['High (~1.0)'] is None and 0.98 < ncc < 0.999: # Avoiding exact 1.0 (trivial)
                    examples['High (~1.0)'] = (base_img, mut_img, ncc, rule)
                    found_count += 1
                    print(f"Found High NCC (Rule {rule}): {ncc:.4f}")
                elif examples['Medium (~0.5)'] is None and 0.45 < ncc < 0.55:
                    examples['Medium (~0.5)'] = (base_img, mut_img, ncc, rule)
                    found_count += 1
                    print(f"Found Medium NCC (Rule {rule}): {ncc:.4f}")
                elif examples['Low (~0.0)'] is None and -0.01 < ncc < 0.01:
                    examples['Low (~0.0)'] = (base_img, mut_img, ncc, rule)
                    found_count += 1
                    print(f"Found Low NCC (Rule {rule}): {ncc:.4f}")
                
                if found_count == 3: break

    # Plotting
    # Filter out None values
    valid_examples = {k: v for k, v in examples.items() if v is not None}
    
    # Define a light blue background color similar to the attached image
    bg_color = '#ADD8E6'  # Light Blue

    for i, (label, data) in enumerate(valid_examples.items()):
        base, mut, ncc, rule_id = data
        
        # Create a new figure for each example
        # Tall figure for vertical stack
        fig = plt.figure(figsize=(8, 12), facecolor=bg_color)
        
        # Create a main axes with a border
        ax_main = fig.add_axes([0.05, 0.05, 0.9, 0.9])
        ax_main.set_facecolor(bg_color)
        # Add a gray border around the figure region
        for spine in ax_main.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(3)
        
        ax_main.set_xticks([])
        ax_main.set_yticks([])

        # Display Father (Top)
        # Position [left, bottom, width, height]
        ax_base = fig.add_axes([0.15, 0.52, 0.7, 0.38])
        ax_base.imshow(np.rot90(base, k=1), cmap='binary', interpolation='nearest')
        ax_base.axis('off')
        
        # Display Mutant (Bottom)
        ax_mut = fig.add_axes([0.15, 0.12, 0.7, 0.38])
        ax_mut.imshow(np.rot90(mut, k=1), cmap='binary', interpolation='nearest')
        ax_mut.axis('off')

        # NCC Text at the very bottom center
        fig.text(0.5, 0.06, f"NCC = {ncc:.4f}", ha='center', va='center', 
                 fontsize=32, fontfamily='serif', fontweight='bold')
        
        # Label the Rule and Type at the top
        fig.text(0.5, 0.93, f"Rule {rule_id}: {label.split(' ')[0]} Robustness", 
                 ha='center', fontsize=20, color='#333333', fontweight='bold', fontfamily='serif')

        output_path = f"ncc_example_{i}_{label.split(' ')[0]}.png"
        plt.savefig(output_path, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved visualization to {output_path}")
    
    plt.show()

if __name__ == "__main__":
    generate_ncc_examples()
