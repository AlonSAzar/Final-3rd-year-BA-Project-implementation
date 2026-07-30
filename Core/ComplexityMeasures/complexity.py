from abc import ABC, abstractmethod
import numpy as np
import zlib
from scipy.stats import wasserstein_distance
from sklearn.metrics import mutual_info_score


class ComplexityMetric(ABC):
    """Abstract Base Class for all complexity measurements."""

    @abstractmethod
    def calculate(self, image: np.ndarray) -> float:
        """Returns a complexity score (higher = more complex)."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Returns the display name of the metric."""
        pass


# --- Concrete Implementations ---

class ZlibComplexity(ComplexityMetric):
    def calculate(self, image: np.ndarray) -> float:
        # Implementation: Compress byte string
        byte_string = image.tobytes()
        return len(zlib.compress(byte_string, level=9))

    def name(self) -> str:
        return "Zlib Bytes"


class RLEComplexity(ComplexityMetric):
    """
    Run-Length Encoding Complexity.
    Perfect for small L (e.g., L=12 to 50).
    "00001111" -> 2 runs -> Complexity 2
    "01010101" -> 8 runs -> Complexity 8
    """
    def calculate(self, image: np.ndarray) -> float:
        # Flatten to 1D if it isn't already
        flat = image.flatten()
        if len(flat) == 0: return 0

        # Count number of times value changes
        # e.g., 0->1 or 1->0
        changes = np.sum(flat[1:] != flat[:-1])

        # Complexity is transitions + 1
        return float(changes + 1)

    def name(self) -> str:
        return "RLE (Transitions)"


class PatchComplexity2D(ComplexityMetric):
    """
    Counts the number of unique 2D patches in the image.
    """

    def __init__(self, patch_size=4):
        self.patch_size = patch_size

    def count_unique_patches(self, image: np.ndarray, stride=1) -> float:
        # Implementation: Count unique N x N patches
        # Returns the number of rows and cols
        rows, cols = image.shape
        seen_patches = set()

        for i in range(0, rows - self.patch_size + 1, stride):
            for j in range(0, cols - self.patch_size + 1, stride):
                # Slice rows i to i+patch_size, etc.
                patch = image[i:i + self.patch_size, j:j + self.patch_size]
                # .flatten() turns it from 2D to 1D, since keys have to be immutable
                # and 2D Numpy shapes are mutable, as opposed to tuples
                key = tuple(patch.flatten())  # flatten to hashable key
                seen_patches.add(key)

        return len(seen_patches)

    def calculate(self, image: np.ndarray) -> float:
        # Normalize to binary representation
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        return self.count_unique_patches(image, stride=1)

    def name(self) -> str:
        return f"LZ 2D Patch ({self.patch_size}x{self.patch_size})"


class MutualInfoComplexity(ComplexityMetric):
    def calculate(self, image: np.ndarray) -> float:
        # Implementation: 1 - Avg(Mutual Info between rows)
        rows, cols = image.shape
        mi_values = []

        for t in range(rows - 1):
            row1 = image[t]
            row2 = image[t + 1]

            mi = mutual_info_score(row1, row2)
            mi_values.append(mi)

        avg_mi = np.mean(mi_values)

        # Normalize
        normalized_mi = avg_mi / 1.0
        return 1.0 - normalized_mi  # higher output = more complexity

    def name(self) -> str:
        return "1 - Mutual Information"