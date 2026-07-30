from abc import ABC, abstractmethod
import numpy as np
import zlib

class  ConditionalComplexityMetric(ABC):
    """Abstract Base Class for all conditional complexity measurements.
    K(y|x) ~ K(xy) - K(x) """

    @abstractmethod
    def calculate(self, img_x: np.ndarray, img_y: np.ndarray) -> float:
        """Returns a conditional complexity score (higher = more complex)."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Returns the display name of the metric."""
        pass

class ZlibConditionalComplexity(ConditionalComplexityMetric):

    def calculate(self, img_x: np.ndarray, img_y: np.ndarray) -> float:
        # Compress x alone
        bytes_x = img_x.tobytes()
        k_x = len(zlib.compress(bytes_x))

        # Compress x and y
        bytes_y = img_y.tobytes()
        k_xy = len(zlib.compress(bytes_x+bytes_y))

        # Conditional complexity
        k_cond = k_xy - k_x

        # K(y|x) cannot be negative, but zlib overhead might cause it
        return max(0.0, k_cond)

    def name(self) -> str:
        return "Zlib Cond. Complexity, Bytes"