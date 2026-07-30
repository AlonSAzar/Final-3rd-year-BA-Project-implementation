import numpy as np
from numba import njit

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import CASimulator, _rule_to_lut

@njit
def _simulate_1d_numba_frozen(trans_rule, seed, T, locked_mask):
    L = len(seed)
    output = np.zeros((T, L), dtype=np.uint8)
    output[0] = seed

    for t in range(1, T):
        for i in range(L):

            # --- Frozen cell: copy previous value ---
            if locked_mask[i]:
                output[t, i] = output[t - 1, i]
                continue

            left   = output[t - 1][(i - 1) % L]
            center = output[t - 1][i]
            right  = output[t - 1][(i + 1) % L]

            neighborhood = (left << 2) | (center << 1) | right
            output[t, i] = trans_rule[neighborhood]

    return output

def generate_locked_mask(L, locked_fraction):
    """
    locked_fraction ∈ [0, 1]
    """
    return (np.random.random(L) < locked_fraction).astype(np.uint8)

class FrozenCellsCA(CASimulator):

    def __init__(self, locked_fraction, L: int, T: int):
        super().__init__(L, T)
        self.locked_fraction = locked_fraction

    def run(self, rule: int, seed: np.ndarray = None) -> np.ndarray:
        if seed is None:
            seed = self.generate_seed()

        locked_mask = generate_locked_mask(self.L, self.locked_fraction)
        lut = _rule_to_lut(rule)

        return _simulate_1d_numba_frozen(
            lut, seed, self.T, locked_mask
        )

    def name(self): return "Frozen Cell CA"


