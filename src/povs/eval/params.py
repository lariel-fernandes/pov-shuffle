from typing import NamedTuple

from povs import POVSOptions


class TVDPerIterParams(NamedTuple):
    """TVD per iteration experiment parameters."""

    seed: int
    num_samples: int
    deck_size: int
    max_iterations: int
    ngram_degrees: list[int]
    povs_options: POVSOptions
