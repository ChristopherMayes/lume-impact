import pathlib
from typing import Sequence

import numpy as np
import pytest
from pmd_beamphysics import single_particle
from pmd_beamphysics.units import pmd_unit
from pydantic import BaseModel, TypeAdapter

from ...z.types import NDArray, PydanticPmdUnit, PydanticParticleGroup

test_path = pathlib.Path(__file__).resolve().parent


@pytest.mark.parametrize(
    "unit",
    [
        pmd_unit("eV", 1.602176634e-19, (2, 1, -2, 0, 0, 0, 0)),
        pmd_unit("T"),
        pmd_unit("T", 1, (0, 1, -2, -1, 0, 0, 0)),
    ],
)
def test_pmd_unit(unit: pmd_unit) -> None:
    print("Unit:", repr(unit))
    adapter = TypeAdapter(PydanticPmdUnit)
    dumped = adapter.dump_json(unit)
    print("Dumped:", repr(dumped))
    deserialized = adapter.validate_json(dumped, strict=True)
    print("Deserialized:", repr(deserialized))
    assert unit == deserialized


@pytest.mark.parametrize(
    "arr",
    [
        np.arange(10),
        np.arange(10.0),
        np.ones((5, 5)),
    ],
)
def test_nd_array(arr: np.ndarray) -> None:
    print("Array:", arr)
    adapter = TypeAdapter(NDArray)
    dumped = adapter.dump_json(arr)
    print("Dumped:", repr(dumped))
    deserialized = adapter.validate_json(dumped, strict=True)
    print("Deserialized:", repr(deserialized))
    np.testing.assert_allclose(arr, deserialized)


@pytest.mark.parametrize(
    "arr",
    [
        [0, 1, 2, 3],
        (0, 1, 2, 3),
    ],
)
def test_sequence_as_ndarray(arr: Sequence[float]) -> None:
    print("Array:", arr)
    adapter = TypeAdapter(NDArray)
    deserialized = adapter.validate_python(arr, strict=True)
    print("Deserialized:", repr(deserialized))
    np.testing.assert_allclose(arr, deserialized)


def test_particle_group_round_trip_python():
    adapter = TypeAdapter(PydanticParticleGroup)

    P0 = single_particle()
    assert adapter.validate_json(adapter.dump_json(P0)) == P0


def test_particle_group_round_trip_json():
    adapter = TypeAdapter(PydanticParticleGroup)
    P0 = single_particle()
    assert adapter.validate_json(adapter.dump_json(P0)) == P0


def test_particle_group_round_trip_1():
    P0 = single_particle()

    class Test(BaseModel):
        a: PydanticParticleGroup

    test = Test(a=P0)
    assert test.model_validate(test.model_dump()).a == P0
    assert test.model_validate_json(test.model_dump_json()).a == P0
