import numpy as np
import matplotlib.pyplot as plt
from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA, _rule_to_lut

def test_rule_30():
    L = 101
    T = 50
    eca = ElementaryCA(L, T)
    
    # Rule 30
    rule = 30
    lut = _rule_to_lut(rule)
    print(f"Rule {rule} LUT: {lut}")
    
    # Center seed
    seed = eca.generate_seed(seed_type="center")
    
    # Run simulation
    output = eca.run(rule, seed)
    
    # Plot
    plt.imshow(output, cmap='binary', interpolation='nearest')
    plt.title(f"Rule {rule} Simulation")
    plt.xlabel("Space")
    plt.ylabel("Time")
    plt.savefig("rule_30_test.png")
    print("Saved rule_30_test.png")

if __name__ == "__main__":
    test_rule_30()
