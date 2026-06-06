from typing import NamedTuple

from povs import FullOptions


class OptionsSetEntry(NamedTuple):
    """A labeled POV Shuffle options configuration for use in timing experiments."""

    label: str
    options: FullOptions


class TimePerDeckSizeParams(NamedTuple):
    """Time per deck size experiment parameters.

    - `seed`: RNG seed for reproducibility.
    - `deck_sizes`: Deck sizes to benchmark, in ascending order.
    - `iterations`: Number of POV Shuffle iterations per timed call.
    - `instance_size`: Feature dimension of each deck element; tensor shape is `(deck_size, instance_size)`.
    - `num_runs`: Number of timed runs per deck size (for averaging).
    - `num_warmup_runs`: Number of warm-up calls before timing begins (not measured).
    - `povs_options_per_deck_size`: POV Shuffle options to use for each deck size, keyed by deck size.
    """

    seed: int
    deck_sizes: list[int]
    iterations: int
    instance_size: int
    num_runs: int
    num_warmup_runs: int
    povs_options_per_deck_size: dict[int, FullOptions]


class TimePerOptionsParams(NamedTuple):
    """Time per options set experiment parameters.

    - `seed`: RNG seed for reproducibility.
    - `deck_size`: Fixed number of elements in the deck for all runs.
    - `instance_size`: Feature dimension of each deck element; tensor shape is `(deck_size, instance_size)`.
    - `iterations`: Number of POV Shuffle iterations per timed call.
    - `num_runs`: Number of timed runs per options set (for averaging).
    - `num_warmup_runs`: Number of warm-up calls before timing begins (not measured).
    - `options_sets`: List of labeled POV Shuffle options configurations to benchmark.
    """

    seed: int
    deck_size: int
    instance_size: int
    iterations: int
    num_runs: int
    num_warmup_runs: int
    options_sets: list[OptionsSetEntry]


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
    povs_options: FullOptions
