import pathlib

import pytest

from ...z.input import ImpactZInput


z_tests = pathlib.Path(__file__).resolve().parent
impact_z_examples = z_tests / "examples"


@pytest.mark.parametrize(
    ("filename",),
    [pytest.param(fn, id=fn.name) for fn in impact_z_examples.glob("*.in")],
)
def test_load_input_smoke(filename: pathlib.Path) -> None:
    ImpactZInput.from_file(filename)
