# POV Shuffle
_**P**arallel **O**ffset **V**irtual-block Shuffle_

A parallelizable, iterative algorithm for efficiently shuffling large datasets in place at scale,
while sufficiently approximating a uniform shuffle within few iterations.

## Installation
```bash
uv add pov-shuffle
```

## Usage
```python
from povs.numpy import pov_shuffle

pov_shuffle(my_array, iterations=3)
```
- See `help(pov_shuffle)` for more details.
