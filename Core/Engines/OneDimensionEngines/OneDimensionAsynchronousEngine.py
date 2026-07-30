import numpy as np
from numba import njit

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import CASimulator, _rule_to_lut


@njit
def _simulate_1d_numba_async(trans_rule, seed, T):
    L = len(seed)
    output = np.zeros((T, L), dtype=np.uint8)

    # Current state (updated in-place)
    state = seed.copy()
    output[0] = state

    indices = np.arange(L)

    for t in range(1, T):

        # Random update order
        np.random.shuffle(indices)

        for k in range(L):
            i = indices[k]

            left   = state[(i - 1) % L]
            center = state[i]
            right  = state[(i + 1) % L]

            neighborhood = (left << 2) | (center << 1) | right
            state[i] = trans_rule[neighborhood]

        output[t] = state

    return output


class AsynchronousCA(CASimulator):
    """Handles 1D Elementary Cellular Automata (Rules 0-255)."""

    def run(self, rule: int, seed: np.ndarray = None) -> np.ndarray:
        """Runs the simulation, returns (T, L) image."""
        if seed is None:
            seed = self.generate_seed()

        lut = _rule_to_lut(rule)
        return _simulate_1d_numba_async(lut, seed, self.T)

    def name(self): return "Asynchronous CA"
