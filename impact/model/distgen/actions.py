from __future__ import annotations

from typing import Any

from distgen import Generator

from lume.actions import WritableActionMixin
from lume.variables import ScalarVariable


class DistgenInputAction(WritableActionMixin[Generator], ScalarVariable):
    """Maps a distgen input parameter via a colon-separated key (e.g. ``r_dist:sigma_xy:value``)."""

    key: str

    def _get(self, simulator: Generator) -> Any:
        return simulator[self.key]

    def _set(self, simulator: Generator, value: Any) -> None:
        simulator[self.key] = value
