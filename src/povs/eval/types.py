from typing import Iterable, NamedTuple, Union


class NgramSpec(NamedTuple):
    """N-gram specification."""

    n: int
    skip: int = 0

    @classmethod
    def parse(cls, x: Union[int, "NgramSpec", Iterable[int]]) -> "NgramSpec":
        if isinstance(x, cls):
            return x
        if isinstance(x, int):
            return cls(n=x)
        return cls(*x)

    @property
    def title(self) -> str:
        suffix = f" (skip {self.skip})" if self.skip else ""
        return f"{self.n}-gram{suffix}"
