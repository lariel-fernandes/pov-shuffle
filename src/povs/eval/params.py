from typing import NamedTuple

from povs import POVSOptions


class TVDPerIterParams(NamedTuple):
    """TVD per iteration experiment parameters.

    - `seed`: RNG seed for reproducibility.
    - `deck_size`: Number of elements in the deck (dataset size proxy).
    - `num_samples`: Number of independent shuffles sampled to estimate the output distribution.
    - `max_iterations`: Number of shuffle iterations tested (from 1 to this value, inclusive).
    - `povs_options`: POV Shuffle algorithm options used in this run.
    - `ngram_degrees`: N-gram degrees for which TVD was measured.
    """

    seed: int
    num_samples: int
    deck_size: int
    max_iterations: int
    ngram_degrees: list[int]
    povs_options: POVSOptions
