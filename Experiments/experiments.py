from abc import abstractmethod, ABC

import numpy as np

# Import from other modules
from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import ElementaryCA
from Core.ComplexityMeasures.complexity import ComplexityMetric

class BaseExperiment(ABC):
    def __init__(self, engine: ElementaryCA, metric: ComplexityMetric):
        self.engine = engine
        self.metric = metric

    @abstractmethod
    def run(self, **kwargs):
        """Execute the simulation phase."""
        pass

    @abstractmethod
    def analyze(self):
        """Process raw results into statistics."""
        pass

    @abstractmethod
    def plot(self):
        """Visualize the analysis."""
        pass

    def execute(self):
        """The Template Method: Defines the standard workflow."""
        print(f"Starting {self.__class__.__name__}...")
        self.run()
        data = self.analyze()
        self.plot(data)

def shuffle_space_time(image: np.ndarray, rng=None) -> np.ndarray:
    """
    Shuffle pixels across the entire space-time image (T, L) so that
    spatial and temporal correlations are destroyed while preserving
    the global pixel value distribution.

    Args:
        image: numpy array of shape (T, L)
        rng: optional random generator with a `shuffle` method (e.g., np.random.RandomState)
    Returns:
        shuffled image with same shape as input
    """
    flat = image.copy().ravel()
    if rng is None:
        np.random.shuffle(flat)
    else:
        rng.shuffle(flat)
    return flat.reshape(image.shape)


def compute_ncc(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Normalized Cross-Correlation (NCC) between two images.
    Images must be the same shape.
    Returns value in [-1, 1], where 1 mean perfect match.
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape for NCC.")

    # Convert to float32 for accurate floating
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Calculate average value of pixels
    img1_mean = img1.mean()
    img2_mean = img2.mean()

    # Scale the images, and multiply each pixel by the value of the equivalent.
    # If 2 pixels have opposite values compared to the mean, the result will be neg.
    # If they have same sign values compared to the mean, it will be positive.
    numerator = np.sum((img1 - img1_mean) * (img2 - img2_mean))

    denominator = np.sqrt(np.sum((img1 - img1_mean) ** 2) * np.sum((img2 - img2_mean) ** 2))

    if denominator == 0:
        return 0.0  # Avoid division by zero

    return numerator / denominator


def int_to_binary_array_numpy(number: int, num_bits: int = 8) -> np.ndarray:
    """
    Converts an integer into a binary array of a specified length (MSB first).
    """
    if number < 0 or number >= (1 << num_bits):
        raise ValueError(f"Number {number} out of range for {num_bits} bits.")

    # Create an array of bit positions to check (e.g., for 8 bits: [7, 6, 5, 4, 3, 2, 1, 0])
    bit_positions = np.arange(num_bits - 1, -1, -1)

    # Perform bitwise right shift (>>) and check the last bit (& 1)
    # This is broadcast across all bit positions simultaneously.
    binary_array = ((number >> bit_positions) & 1).astype(np.uint8)

    return binary_array



