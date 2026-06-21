# Glossary

## Common Concepts

### Deck Size / Dataset Size
The number of instances to be shuffled.

### Instance
Same-sized, same dtype objects to be shuffled along the axis 0.

Example: For a rank-3 `float32` tensor of shape `(I, M, N)`, there are `I` instances, each instance being a `(M, N)` matrix, and the instance size is `M * N`, with memory footprint of `M * N * 32bits`.

### (Absolute) Travel Distance and Relative Travel Distance
After shuffling a dataset, the travel distance of each instance is how far it ended up from its initial position.

For a tuple of instances, their relative travel distance is how far apart they became, relative to how far apart they used to be.
- Example 1: If `1` and `2` end up next to each other, the relative travel distance of the bigram `[1, 2]` is `0` (i.e. they are just as far as they used to be).
- Example 2: If we observe `[1, _, _, 10]`, the relative travel distance of the skip-gram `[1, 10]` is `-6` (i.e. they are `6` positions _closer_ than they used to be).
- Example 3: The relative travel distance tuple of the 3-gram `[10, 12, 1]` is `(X, Y)`, where `X = (distance_now - distance_before) = 1 - 2 = -1` and `Y` follows the same formula,
  but depends on the dataset size (`distance_before` requires going from `10` to `1` by moving the cursor to the right until it wraps around the dataset).

## Algorithm Internals

### Physical Block
Non-overlapping, same-sized, contiguous logical partitions of the dataset, measured in number of instances.

Built by mapping dataset sections to physical block IDs, starting at a certain data offset (in number of instances).
The mapping wraps around the end of the dataset to cover instances skipped by the offset,
and guards against overindexing in the last physical block (in case the dataset size is not divisible by the physical block size).

### Virtual Block
Non-overlapping, same-sized, contiguous-by-parts, logical partitions of the dataset, measured in number of assigned physical blocks.

Built by randomly assigning physical block IDs to virtual block IDs (possibly with padding assignments when total physical blocks are not divisible by the virtual block size).

## Bias Metrics

### TVD (Total Variation Distance)
Given the event space of a random variable, and the estimated probability distribution of those events based on observations,
the TVD measures how much that distribution differs from a reference distribution.

In the context of this project, the reference distribution is always a uniform, which is the expected
distribution for any event space produced by an unbiased shuffle.

In undersampled experiments, where the number of event observations is smaller than the event space itself, we use an adaptive TVD formula:
- The reference (i.e. uniform) probability of any event is not `1 / event_space_size`, but `1 / num_observations`
- The contribution of unobserved events to the TVD is capped by the number of observations
  (intuition: if a die of 100 sides is thrown 10 times, at most 10 sides can show up, not 100).

### Positional TVD

With the event space being all possible (absolute) travel distances, we build a sample of independent shuffling episodes
(using either the baseline `numpy.shuffle` or the POV-shuffle algorithm) and use the observations to estimate the
distribution of travel distances. The positional TVD measures how much that distribution differs from a uniform one.

### N-gram TVD

With the event space being all possible tuples of relative travel distances for N-grams of size N: We build a sample
of independent shuffling episodes (using either the baseline `numpy.shuffle` or the POV-shuffle algorithm)
and use the observations to estimate the distribution of relative travel distance tuples.
The N-gram TVD measures how much that distribution differs from a uniform one.
