import numpy as np
from numba import njit

from Core.Engines.OneDimensionEngines.OneDimensionBasicEngine import CASimulator, _rule_to_lut


@njit
def _simulate_1d_numba_noise(trans_rule, seed, T, noise_level):
    L = len(seed)
    output = np.zeros((T, L), dtype=np.uint8)
    output[0] = seed

    for t in range(1, T):
        for i in range(L):
            left   = output[t - 1][(i - 1) % L]
            center = output[t - 1][i]
            right  = output[t - 1][(i + 1) % L]

            neighborhood = (left << 2) | (center << 1) | right
            new_state = trans_rule[neighborhood]

            # Noise injection
            if np.random.random() < noise_level:
                new_state = 1 - new_state

            output[t][i] = new_state

    return output


class RandomNoiseCA(CASimulator):

    def __init__(self, noise_level, L: int, T: int):
        super().__init__(L, T)
        self.noise_level = noise_level

    def run(self, rule: int, seed: np.ndarray = None) -> np.ndarray:
        """Runs the simulation, returns (T, L) image."""
        if seed is None:
            seed = self.generate_seed()

        lut = _rule_to_lut(rule)
        return _simulate_1d_numba_noise(lut, seed, self.T, self.noise_level)

    def name(self): return "Random Noise CA"
