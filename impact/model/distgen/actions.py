from typing import Any

from distgen import Generator

from impact.model.base import Action, WritableAction


class DistgenAction(Action[Generator]):
    """Abstract base for all distgen actions."""


class WritableDistgenAction(WritableAction[Generator], DistgenAction):
    """Abstract base for writable distgen actions."""


class DistgenInputAction(WritableDistgenAction):
    """Maps a distgen input parameter via a colon-separated key (e.g. ``r_dist:sigma_xy:value``).

    If ``has_units`` is True the parameter is a quantity dict; the ``:value``
    suffix is already included in ``key`` and only the magnitude is read/written.
    """

    key: str
    has_units: bool

    def get(self, gen: Generator) -> Any:
        return gen[self.key]

    def set(self, gen: Generator, value: Any) -> None:
        gen[self.key] = value
