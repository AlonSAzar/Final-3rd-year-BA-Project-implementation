import numpy as np
from numba import njit


@njit(fastmath=True)
def _simulate_1d_numba(rule: int, seed: np.ndarray, T: int) -> np.ndarray:
    L = len(seed)
    output = np.zeros((T, L), dtype=np.uint8)
    output[0] = seed

    for t in range(1, T):
        prev = output[t - 1]
        curr = output[t]

        # --- Handle Boundary i = 0 (left wraps to L - 1) ---
        left = prev[L - 1]
        center = prev[0]
        right = prev[1]
        nb = (left << 2) | (center << 1) | right
        curr[0] = (rule >> nb) & 1

        # --- Tight Inner Loop (No Modulo) ---
        for i in range(1, L - 1):
            left = prev[i - 1]
            center = prev[i]
            right = prev[i + 1]
            nb = (left << 2) | (center << 1) | right
            curr[i] = (rule >> nb) & 1

        # --- Handle Boundary i = L - 1 (right wraps to 0) ---
        left = prev[L - 2]
        center = prev[L - 1]
        right = prev[0]
        nb = (left << 2) | (center << 1) | right
        curr[L - 1] = (rule >> nb) & 1

    return output


def _rule_to_lut(rule: int) -> np.ndarray:
    """Convert rule number to lookup table of 8 values (3-bit neighborhoods).

    Index i corresponds directly to neighborhood value i (0-7):
    lut[7] = Bit 7 (neighborhood 111)
    lut[0] = Bit 0 (neighborhood 000)
    """
    return np.array([(rule >> i) & 1 for i in range(8)], dtype=np.uint8)


class CASimulator:
    """Base class for CA Simulations."""

    def __init__(self, L: int, T: int):
        self.L = L
        self.T = T

    def generate_seed(self, seed_type="random") -> np.ndarray:
        if seed_type == "random":
            return np.random.randint(0, 2, size=self.L, dtype=np.uint8)
        elif seed_type == "center":
            s = np.zeros(self.L, dtype=np.uint8)
            s[self.L // 2] = 1
            return s
        return np.zeros(self.L, dtype=np.uint8)


class ElementaryCA(CASimulator):
    """Handles 1D Elementary Cellular Automata (Rules 0-255)."""

    def run(self, rule: int, seed: np.ndarray = None) -> np.ndarray:
        """Runs the simulation, returns (T, L) image."""
        if seed is None:
            seed = self.generate_seed()

        return _simulate_1d_numba(rule, seed, self.T)

    def name(self):
        return "Elementary CA"