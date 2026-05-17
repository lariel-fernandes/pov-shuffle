from typing import Any, Type

import yaml as _yaml


class _Dumper(_yaml.Dumper):
    """Patch of `yaml.Dumper`.

    Additional features:
       - Serializes NamedTuple as dict
    """

    def represent_data(self, data):
        if isinstance(data, tuple) and callable(f := getattr(data, "_asdict", None)):
            data = f()

        return super().represent_data(data)


class yaml:
    """Namespace for extensions of the `yaml` package."""

    @staticmethod
    def dump(
        data: Any,
        *,
        sort_keys: bool = False,
        width: float = float("inf"),
        Dumper: Type[_yaml.Dumper] = _Dumper,
        **kwargs,
    ) -> str:
        """Drop-in replacement of yaml.dump with more convenient defaults."""
        return str(
            _yaml.dump(
                data,
                width=width,
                sort_keys=sort_keys,
                Dumper=Dumper,
                **kwargs,
            ),
        )
