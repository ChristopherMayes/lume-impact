from __future__ import annotations

import pathlib

import pytest
from pytao import SubprocessTao as Tao

from ...z import ImpactZInput
from .conftest import z_tests

lattice_root = z_tests / "bmad"

lattices = pytest.mark.parametrize(
    "lattice", [pytest.param(fn, id=fn.name) for fn in lattice_root.glob("*.bmad")]
)


@lattices
def test_from_tao(lattice: pathlib.Path) -> None:
    with Tao(lattice_file=lattice, noplot=True) as tao:
        print(ImpactZInput.from_tao(tao))
