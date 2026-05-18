from __future__ import annotations

from typing import Any

from distgen import Generator

from impact.model.base import Action, WritableAction


class DistgenAction(Action[Generator]):
    """Abstract base for all distgen actions."""


class WritableDistgenAction(WritableAction[Generator], DistgenAction):
    """Abstract base for writable distgen actions."""


class DistgenInputAction(WritableDistgenAction):
    """Maps a distgen input parameter via a colon-separated key (e.g. ``r_dist:sigma_xy:value``)."""

    key: str

    def _get(self, simulator: Generator) -> Any:
        return simulator[self.key]

    def _set(self, simulator: Generator, value: Any) -> None:
        simulator[self.key] = value
