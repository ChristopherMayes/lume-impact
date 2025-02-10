from __future__ import annotations
import json
import pathlib
import time
import numpy as np
import typing

import h5py
import pytest
import pydantic

from ... import z as IZ
from ...z import ImpactZ
from ...z.input import AnyInputElement
from ...z.archive import (
    store_in_hdf5_file,
    restore_from_hdf5_file,
)
from .conftest import z_example1, test_artifacts


@pytest.fixture(scope="module")
def impact():
    I = IZ.ImpactZ(input=z_example1)
    I.run()
    return I


@pytest.fixture
def hdf5_filename(
    tmp_path: pathlib.Path,
    request: pytest.FixtureRequest,
) -> pathlib.Path:
    return tmp_path / f"{request.node.name}.h5"


lattice_element_classes = typing.get_args(AnyInputElement)


@pytest.mark.parametrize(
    "obj",
    [
        # pytest.param(IZ.Drift(), id="Drift"),
        # pytest.param(IZ.Quadrupole(), id="Quadrupole"),
        # pytest.param(IZ.ConstantFocusing(), id="ConstantFocusing"),
        # pytest.param(IZ.Solenoid(), id="Solenoid"),
        # pytest.param(IZ.Dipole(), id="Dipole"),
        # pytest.param(IZ.Multipole(), id="Multipole"),
        # pytest.param(IZ.DTL(), id="DTL"),
        # pytest.param(IZ.CCDTL(), id="CCDTL"),
        # pytest.param(IZ.CCL(), id="CCL"),
        # pytest.param(IZ.SuperconductingCavity(), id="SuperconductingCavity"),
        # pytest.param(IZ.SolenoidWithRFCavity(), id="SolenoidWithRFCavity"),
        # pytest.param(IZ.TravelingWaveRFCavity(), id="TravelingWaveRFCavity"),
        # pytest.param(IZ.UserDefinedRFCavity(), id="UserDefinedRFCavity"),
        # pytest.param(IZ.ShiftCentroid(), id="ShiftCentroid"),
        # pytest.param(IZ.WriteFull(), id="WriteFull"),
        # pytest.param(IZ.DensityProfileInput(), id="DensityProfileInput"),
        # pytest.param(IZ.DensityProfile(), id="DensityProfile"),
        # pytest.param(IZ.Projection2D(), id="Projection2D"),
        # pytest.param(IZ.Density3D(), id="Density3D"),
        # pytest.param(IZ.WritePhaseSpaceInfo(), id="WritePhaseSpaceInfo"),
        # pytest.param(IZ.WriteSliceInfo(), id="WriteSliceInfo"),
        # pytest.param(
        #     IZ.ScaleMismatchParticle6DCoordinates(),
        #     id="ScaleMismatchParticle6DCoordinates",
        # ),
        # pytest.param(
        #     IZ.CollimateBeamWithRectangularAperture(),
        #     id="CollimateBeamWithRectangularAperture",
        # ),
        # pytest.param(
        #     IZ.RotateBeamWithRespectToLongitudinalAxis(),
        #     id="RotateBeamWithRespectToLongitudinalAxis",
        # ),
        # pytest.param(IZ.BeamShift(), id="BeamShift"),
        # pytest.param(IZ.BeamEnergySpread(), id="BeamEnergySpread"),
        # pytest.param(IZ.ShiftBeamCentroid(), id="ShiftBeamCentroid"),
        # pytest.param(IZ.IntegratorTypeSwitch(), id="IntegratorTypeSwitch"),
        # pytest.param(IZ.BeamKickerByRFNonlinearity(), id="BeamKickerByRFNonlinearity"),
        # pytest.param(IZ.RfcavityStructureWakefield(), id="RfcavityStructureWakefield"),
        # pytest.param(IZ.EnergyModulation(), id="EnergyModulation"),
        # pytest.param(IZ.KickBeamUsingMultipole(), id="KickBeamUsingMultipole"),
        # pytest.param(IZ.HaltExecution(), id="HaltExecution"),
        pytest.param(cls(), id=cls.__name__)
        for cls in lattice_element_classes
    ],
)
def test_round_trip_json(obj: pydantic.BaseModel) -> None:
    print("Object", obj)
    as_json = obj.model_dump_json()
    print("Dumped to JSON:")
    print(as_json)
    deserialized = obj.model_validate_json(as_json)
    print("Back to Python:")
    print(deserialized)
    assert obj == deserialized


def test_hdf_archive(
    impact: ImpactZ,
    hdf5_filename: pathlib.Path,
) -> None:
    # output = impact.run(raise_on_error=True)
    output = impact.output
    assert output is not None
    assert output.run.success

    impact.load_output()
    orig_input = impact.input
    orig_output = impact.output
    assert orig_output is not None

    t0 = time.monotonic()
    impact.archive(hdf5_filename)

    t1 = time.monotonic()
    impact.load_archive(hdf5_filename)

    t2 = time.monotonic()
    print("Took", t1 - t0, "s to archive")
    print("Took", t2 - t1, "s to restore")
    assert impact.output is not None

    # assert orig_input.model_dump_json(indent=True) == impact.input.model_dump_json(indent=True)
    # assert orig_output.model_dump_json(indent=True) == impact.output.model_dump_json(indent=True)
    orig_input.filename = impact.input.filename
    orig_input_repr = repr(orig_input)
    restored_input_repr = repr(impact.input)
    assert orig_input_repr == restored_input_repr

    orig_output_repr = repr(orig_output)
    restored_output_repr = repr(impact.output)
    assert orig_output_repr == restored_output_repr

    if orig_input != impact.input:
        compare(orig_input, impact.input)
        assert False, "Verbose comparison should have failed?"
    if orig_output != impact.output:
        compare(orig_output, impact.output)
        assert False, "Verbose comparison should have failed?"

    with open(test_artifacts / "orig_input.json", "wt") as fp:
        print(json_for_comparison(orig_input), file=fp)
    with open(test_artifacts / "restored_input.json", "wt") as fp:
        print(json_for_comparison(impact.input), file=fp)
    with open(test_artifacts / "orig_output.json", "wt") as fp:
        print(json_for_comparison(orig_output), file=fp)
    with open(test_artifacts / "restored_output.json", "wt") as fp:
        print(json_for_comparison(impact.output), file=fp)

    assert json_for_comparison(orig_input) == json_for_comparison(impact.input)
    assert json_for_comparison(orig_output) == json_for_comparison(impact.output)


# def test_hdf_archive_particles(
#     impact: ImpactZ,
#     hdf5_filename: pathlib.Path,
# ) -> None:
#     impact.input.main.namelists.append(Write(beam="end"))
#     orig_output = impact.run(raise_on_error=True)
#     assert orig_output.run.success
#
#     orig_output.load_particles()
#
#     orig_particles = orig_output.particles
#     assert len(orig_output.particles)
#
#     t0 = time.monotonic()
#     impact.archive(hdf5_filename)
#
#     t1 = time.monotonic()
#     impact.load_archive(hdf5_filename)
#     new_output = impact.output
#
#     t2 = time.monotonic()
#     print("Took", t1 - t0, "s to archive")
#     print("Took", t2 - t1, "s to restore")
#     assert new_output is not None
#     assert list(new_output.particles) == list(orig_particles)
#
#     for key in new_output.particles:
#         particles = orig_particles[key]
#         print("Checking particles", particles)
#         assert particles == new_output.particles[key]
#
#
# def test_pick_from_archive(
#     impact: ImpactZ,
#     hdf5_filename: pathlib.Path,
# ) -> None:
#     impact.input.main.namelists.append(Write(beam="end"))
#     orig_output = impact.run(raise_on_error=True)
#     assert orig_output.run.success
#
#     orig_output.load_particles()
#
#     orig_particles = orig_output.particles
#     assert len(orig_output.particles)
#
#     impact.archive(hdf5_filename)
#
#     with h5py.File(hdf5_filename) as h5:
#         for key in orig_output.particles:
#             particles = orig_particles[key]
#             print("Checking particles", particles)
#
#             loaded = pick_from_archive(h5[f"output/particles/{key}"])
#             assert isinstance(loaded, ParticleGroup)
#             assert loaded == particles
#             assert loaded.species == "electron"


def compare(obj, expected, history=()):
    print("Comparing:", history, type(obj).__name__)
    assert isinstance(obj, type(expected))
    # assert repr(obj) == repr(expected)
    if isinstance(obj, pydantic.BaseModel):
        for attr, fld in obj.model_fields.items():
            value = getattr(obj, attr)
            if isinstance(value, np.ndarray):
                assert fld.annotation is np.ndarray

            compare(
                getattr(obj, attr),
                getattr(expected, attr),
                history=history + (attr,),
            )
    elif isinstance(obj, dict):
        assert set(obj) == set(expected)
        for key in obj:
            compare(
                obj[key],
                expected[key],
                history=history + (key,),
            )
    elif isinstance(obj, (list, tuple)):
        assert len(obj) == len(expected)
        for idx, (value, value_expected) in enumerate(zip(obj, expected)):
            compare(
                value,
                value_expected,
                history=history + (idx,),
            )
    elif isinstance(obj, (np.ndarray, float)):
        if isinstance(obj, np.ndarray):
            if not obj.shape and not expected.shape:
                return
        assert np.allclose(obj, expected, equal_nan=True)
    else:
        assert obj == expected


def json_for_comparison(model: pydantic.BaseModel) -> str:
    # Assuming dictionary keys can't be assumed to be sorted
    data = json.loads(model.model_dump_json())
    return json.dumps(data, sort_keys=True, indent=True)


def test_hdf_archive_using_group(
    impact: ImpactZ,
    request: pytest.FixtureRequest,
    # hdf5_filename: pathlib.Path,
) -> None:
    output = impact.run(raise_on_error=True)
    assert output.run.success

    impact.load_output()
    orig_input = impact.input
    orig_output = impact.output
    assert orig_output is not None

    hdf5_filename = test_artifacts / f"archive-{request.node.name}.h5"
    t0 = time.monotonic()
    with h5py.File(hdf5_filename, "w") as h5:
        impact.archive(h5)

    t1 = time.monotonic()

    with h5py.File(hdf5_filename, "r") as h5:
        impact.load_archive(h5)

    t2 = time.monotonic()
    print("Took", t1 - t0, "s to archive")
    print("Took", t2 - t1, "s to restore")

    assert impact.output is not None

    # with open("orig_output.json", "wt") as fp:
    #     print(json_for_comparison(orig_output), file=fp)
    # with open("restored_output.json", "wt") as fp:
    #     print(json_for_comparison(impact.output), file=fp)

    orig_input_repr = repr(orig_input)
    restored_input_repr = repr(impact.input)
    assert orig_input_repr == restored_input_repr

    orig_output_repr = repr(orig_output)
    restored_output_repr = repr(impact.output)
    assert orig_output_repr == restored_output_repr


class StringModel(pydantic.BaseModel):
    str_value: str
    bytes_value: bytes


def test_null_bytes_storage(hdf5_filename: pathlib.Path):
    orig = StringModel(str_value="one\0two\0three", bytes_value=b"one\0two\0three")
    with h5py.File(hdf5_filename, "w") as h5:
        store_in_hdf5_file(h5, orig)

    with h5py.File(hdf5_filename, "r") as h5:
        output = restore_from_hdf5_file(h5)

    print(orig)
    print(output)
    assert orig == output
