# strategies.py
from abc import ABC, abstractmethod
import numpy as np
import random


class MutationStrategy(ABC):
    @abstractmethod
    def get_variations_count(self, engine, rule: int, seed: np.ndarray, default: int): pass

    @abstractmethod
    def apply(self, engine, rule: int, seed: np.ndarray, bit_index): pass

    @abstractmethod
    def name(self): pass

class SameSeedAndRuleStrategy(MutationStrategy):
    def get_variations_count(self, engine, rule: int, seed: np.ndarray, default: int):
        return default

    def apply(self, engine, rule: int, seed: np.ndarray, bit_index):
        return rule, seed

    def name(self): return "Same Parameters"

class BitFlipRuleStrategy(MutationStrategy):
    def get_variations_count(self, engine, rule, seed, default):
        return 8

    def apply(self, engine, rule, seed, bit_index):
        """Flips a single bit in the rule definition (bit_index)."""
        new_rule = rule ^ (1 << bit_index)
        return new_rule, seed

    def name(self): return "Rule Bit-Flip"


class BitFlipSeedStrategy(MutationStrategy):
    def get_variations_count(self, engine, rule, seed, default):
        return len(seed)

    def apply(self, engine, rule, seed, bit_index):
        # Make a copy so we don't mess up the original for the next loop
        s = seed.copy()

        s[bit_index] = 1 - s[bit_index]  # Flip the bit
        return rule, s

    def name(self): return "Seed Bit-Flip"


    """ TODO with/out shuffle control
    """

class RandomSeedStrategy(MutationStrategy):
    def get_variations_count(self, engine, rule: int, seed: np.ndarray, default):
        return default

    def apply(self, engine, rule: int, seed: np.ndarray, bit_index):
        rand_seed = engine.generate_seed()
        return rule, rand_seed

    def name(self): return "Random Seed"

class RandomRuleStrategy(MutationStrategy):
    def get_variations_count(self, engine, rule: int, seed: np.ndarray, default):
        return default

    def apply(self, engine, rule: int, seed: np.ndarray, bit_index):
        rand_rule = random.randint(0, 255)
        return rand_rule, seed

    def name(self): return "Random Rule"


class RandomSeedAndRuleStrategy(MutationStrategy):
    def get_variations_count(self, engine, rule: int, seed: np.ndarray, default):
        return default

    def apply(self, engine, rule: int, seed: np.ndarray, bit_index):
        rand_seed = engine.generate_seed()
        rand_rule = random.randint(0, 255)
        return rand_rule, rand_seed

    def name(self): return "Random Seed & Rule"





