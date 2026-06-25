from typing import NamedTuple, Union


class NgramSpec(NamedTuple):
    """N-gram specification."""

    n: int
    skip: int = 0

    @classmethod
    def parse(cls, x: Union[int, "NgramSpec"]) -> "NgramSpec":
        return cls(n=x) if isinstance(x, int) else x

    @property
    def title(self) -> str:
        suffix = f" (skip {self.skip})" if self.skip else ""
        return f"{self.n}-gram{suffix}"
