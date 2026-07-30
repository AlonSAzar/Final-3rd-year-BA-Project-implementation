import numpy as np
from numba import njit


@njit
def _simulate_1d_numba_chimera(rule_lut_A, rule_lut_B, mask_B, seed, T):
    L = len(seed)
    output = np.zeros((T, L), dtype=np.uint8)
    output[0] = seed

    for t in range(1, T):
        for i in range(L):

            left = output[t - 1][(i - 1) % L]
            center = output[t - 1][i]
            right = output[t - 1][(i + 1) % L]

            neighborhood = (left << 2) | (center << 1) | right

            if mask_B[i]:
                output[t, i] = rule_lut_B[neighborhood]
            else:
                output[t, i] = rule_lut_A[neighborhood]

    return output

# TODO use this and complete