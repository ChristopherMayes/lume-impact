from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import model_validator

from pmd_beamphysics import ParticleGroup

from impact.impact import Impact
from lume.actions import ReadOnlyActionMixin, WritableActionMixin
from lume.variables import (
    BoolVariable,
    NDVariable,
    ParticleGroupVariable,
    ScalarVariable,
    StrVariable,
)


def _empty_particle_group() -> ParticleGroup:
    return ParticleGroup(
        data={
            "x": np.array([]),
            "px": np.array([]),
            "y": np.array([]),
            "py": np.array([]),
            "z": np.array([]),
            "pz": np.array([]),
            "t": np.array([]),
            "weight": np.array([]),
            "status": np.array([], dtype=int),
            "species": "electron",
        }
    )


class ScalarEleAction(WritableActionMixin[Impact], ScalarVariable):
    """Maps a numeric element attribute: ``impact.ele[ele_name][attribute]``."""

    ele_name: str
    attribute: str

    def _get(self, simulator: Impact) -> Any:
        return simulator.ele[self.ele_name][self.attribute]

    def _set(self, simulator: Impact, value: Any) -> None:
        simulator.ele[self.ele_name][self.attribute] = value


class StrEleAction(WritableActionMixin[Impact], StrVariable):
    """Maps a string element attribute: ``impact.ele[ele_name][attribute]``."""

    ele_name: str
    attribute: str

    def _get(self, simulator: Impact) -> Any:
        return simulator.ele[self.ele_name][self.attribute]

    def _set(self, simulator: Impact, value: Any) -> None:
        simulator.ele[self.ele_name][self.attribute] = value


class HeaderAction(WritableActionMixin[Impact], ScalarVariable):
    """Maps a header key: ``impact.header[key]``."""

    key: str

    def _get(self, simulator: Impact) -> Any:
        return simulator.header[self.key]

    def _set(self, simulator: Impact, value: Any) -> None:
        simulator.header[self.key] = value


class StatAction(ReadOnlyActionMixin[Impact], NDVariable):
    """Maps an output stat: ``impact.stat(stat_name)``. Read-only."""

    stat_name: str

    def _get(self, simulator: Impact) -> Any:
        arr = simulator.stat(self.stat_name)
        if arr.shape[0] == self.shape[0]:
            return arr
        out = np.full(self.shape[0], np.nan, dtype=float)
        n = min(arr.shape[0], self.shape[0])
        out[:n] = arr[:n]
        return out


class ScalarRunInfoAction(ReadOnlyActionMixin[Impact], ScalarVariable):
    """Maps a numeric run_info entry: ``impact.output['run_info'][key]``. Read-only."""

    key: str

    def _get(self, simulator: Impact) -> Any:
        return simulator.output["run_info"][self.key]


class BoolRunInfoAction(ReadOnlyActionMixin[Impact], BoolVariable):
    """Maps a boolean run_info entry: ``impact.output['run_info'][key]``. Read-only."""

    key: str

    def _get(self, simulator: Impact) -> Any:
        return simulator.output["run_info"][self.key]


class StrRunInfoAction(ReadOnlyActionMixin[Impact], StrVariable):
    """Maps a string run_info entry: ``impact.output['run_info'][key]``. Read-only."""

    key: str

    def _get(self, simulator: Impact) -> Any:
        return simulator.output["run_info"][self.key]


class ParticleGroupAction(WritableActionMixin[Impact], ParticleGroupVariable):
    """Maps a particle group: ``impact.particles[tool_name]``.

    Only ``initial_particles`` is writable; all other tool names must be
    constructed with ``read_only=True``.
    """

    tool_name: str
    default_value: Any = None

    @model_validator(mode="after")
    def _check_initial_particles(self) -> "ParticleGroupAction":
        if self.tool_name != "initial_particles" and not self.read_only:
            raise ValueError(
                f"Particle group '{self.tool_name}' is not writable; "
                "set read_only=True for non-initial_particles groups"
            )
        return self

    def _get(self, simulator: Impact) -> Any:
        return simulator.particles[self.tool_name]

    def _set(self, simulator: Impact, value: Any) -> None:
        simulator.initial_particles = value
