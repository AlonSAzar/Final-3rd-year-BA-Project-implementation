# Amit Dynamics: Cellular Automata Complexity & Robustness

This project explores the relationship between structural complexity and evolutionary robustness in 1D Cellular Automata (CA). It provides a framework for simulating various CA engines, measuring phenotype complexity, and running evolutionary experiments.

## 📁 Project Structure

### [Core](Core) - The engine room
*   **[Engines/](Core/Engines)**: Implementation of various CA simulators:
    *   `ElementaryCA`: Standard Rules 0-255 using Numba optimization.
    *   `AsynchronousCA`: Non-deterministic cell updates.
    *   `FrozenCellsCA`, `RandomNoiseCA`: Specialized dynamics.
*   **[ComplexityMeasures/](Core/ComplexityMeasures)**: Metrics for quantifying CA phenotypes:
    *   `ZlibComplexity`: Compression-based complexity.
    *   `RLEComplexity`: Run-length encoding transitions.
    *   `PatchComplexity2D`: Unique NxN patch counting.
*   **[strategies.py](Core/strategies.py)**: Strategy Pattern implementation for mutations (Bit-flips, Random Rule/Seed).

### [Experiments](Experiments) - Scientific workflows
Organized by research focus, all experiments inherit from a common [BaseExperiment](Experiments/experiments.py).
*   **[BasicComplexityExperiments/](Experiments/BasicComplexityExperiments)**: Core research on Robustness vs. Complexity and Simplicity Bias.
*   **[EvolutionExperiments/](Experiments/EvolutionExperiments)**: Long-term dynamics including Glider evolution and thermodynamics.
*   **[ConditionalComplexityExperiments/](Experiments/ConditionalComplexityExperiments)**: Transitions and conditional entropy analysis.

### [Analysis](Analysis) & [Visualizations](Visualizations)
*   **[PCA/](Analysis/PCA)** & **[TSNE/](Analysis/TSNE)**: Dimensionality reduction tools for the "Morphospace" of CA rules.
*   **[Interactive HTMLs](Visualizations)**: Dynamic plots for exploring rule clusters (Morphospace, Texture).

## 🚀 Getting Started

### Prerequisites
*   Python 3.8+
*   Dependencies: `numpy`, `matplotlib`, `scipy`, `scikit-learn`, `numba`, `tqdm`, `statsmodels`.

### Running an Experiment
The project uses a standard `execute()` workflow for experiments:
```python
from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.ComplexityMeasures.complexity import ZlibComplexity
from Experiments.BasicComplexityExperiments.RobustnessExperiment import RobustnessExperiment

# 1. Setup Engine and Metric
engine = ElementaryCA(L=100, T=100)
metric = ZlibComplexity()

# 2. Initialize and Run Experiment
exp = RobustnessExperiment(engine, metric)
exp.execute()
```

## 🛠️ Design Patterns
This project follows professional OOP principles:
*   **Strategy Pattern**: Used in [strategies.py](Core/strategies.py) to decouple mutation logic from experiment execution.
*   **Template Method**: [BaseExperiment](Experiments/experiments.py) defines a rigid `run -> analyze -> plot` sequence that all subclasses follow.
*   **Abstract Base Classes**: Used across Engines, Metrics, and Experiments to ensure consistent interfaces.

## 📝 Ongoing Research (TODOs)
*   Refining the abstraction layers for 1D vs 2D CA.
*   Integrating UMAP for better morphospace visualization.
*   Optimizing RLE metrics for varied lattice sizes.
