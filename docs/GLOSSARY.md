# Glossary

## Common Concepts

### Deck Size / Dataset Size
The number of instances to be shuffled.

### Instance
Same-sized, same dtype objects to be shuffled along the axis 0.

Example: For a rank-3 `float32` tensor of shape `(I, M, N)`, there are `I` instances, each instance being a `(M, N)` matrix, and the instance size is `M * N`, with memory footprint of `M * N * 32bits`.

### (Absolute) Travel Distance and Relative Travel Distance
After shuffling a dataset, the travel distance of each instance is how far it ended up from its initial position.

For an n-gram or skip-gram of instances, the **signed** relative travel distance of the k-th element with respect to the anchor (first element) is:

```
rtd_k = (v_k - v_0) - k * step
```

where `v_k - v_0` is the original signed positional gap and `k * step` is their current separation. Positive means they got closer; negative means they drifted further apart. Values range approximately from `-deck_size` to `+deck_size`.

- **Example 1:** Bigram `[1, 2]` ends up adjacent: `rtd = (2 - 1) - 1 = 0` (unchanged relative distance).
- **Example 2:** Skip-gram `[0, 10]` with step `3` observed after shuffling: `rtd = (10 - 0) - 3 = 7` (7 positions closer).
- **Example 3:** 3-gram `[10, 15, 3]` at adjacent positions: `rtd_1 = (15 - 10) - 1 = 4`, `rtd_2 = (3 - 10) - 2 = -9` (15 got 4 closer to 10; 3 drifted 9 further from 10).

> **Note:** The [N-gram TVD metric](#n-gram-tvd) applies `% deck_size` to each signed RTD before counting.
> This collapses the signed range into an event space of `deck_size - 1` equivalence classes that are equally likely
> under a uniform shuffle, which is required for the reference distribution to be uniform.

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
- The contribution of unobserved events to the TVD is capped by the observation budget
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
