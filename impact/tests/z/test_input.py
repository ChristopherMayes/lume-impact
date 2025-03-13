import logging
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ...z.input import ImpactZInput, Quadrupole, WriteFull, Drift
from ...z.parsers import parse_input_line


logger = logging.getLogger(__name__)
z_tests = pathlib.Path(__file__).resolve().parent
impact_z_examples = z_tests / "examples"


def normalize_source(source: str) -> str:
    """
    Normalize input file source for comparison.

    Removes comments, normalizes whitespace to a single space, and fixes
    floating point representation.
    """
    lines = source.strip().splitlines()

    def fix_line(line: str) -> str:
        line = line.replace("d", "e")
        line = line.replace("D", "e")

        line = line.split("/", 1)[0]

        # Any whitespace becomes a single space:
        return re.sub(r"\s+", " ", line).strip()

    return "\n".join(
        fix_line(line) for line in lines if not line.startswith("!") and line.strip()
    )


def compare_inputs(expected: str, generated: str):
    orig_expected = expected  # noqa
    orig_generated = generated  # noqa
    expected = normalize_source(expected)
    generated = normalize_source(generated)

    expected_lines = expected.splitlines()
    generated_lines = generated.splitlines()
    for lineno, (line_expected, line_generated) in enumerate(
        zip(expected_lines, generated_lines),
        start=1,
    ):
        expected_parts = parse_input_line(line_expected).data
        generated_parts = parse_input_line(line_generated).data

        if lineno == 1 and len(expected_parts) != len(generated_parts):
            # NOTE: a bit of first line handling for GPU settings
            # this is a "do not care about GPU setting" section
            if len(expected_parts) == 3 and len(generated_parts) == 2:
                generated_parts.append(expected_parts[-1])
            elif len(expected_parts) == 2 and len(generated_parts) == 3:
                expected_parts.append(generated_parts[-1])

        if len(expected_parts) != len(generated_parts):
            if len(generated_parts) > len(expected_parts):
                num_extra = len(generated_parts) - len(expected_parts)
                if np.allclose(generated_parts[-num_extra:], 0):
                    msg = f"Ignoring extra input generated on line {line_generated} (expected: {line_expected})"
                    # TODO: we may circle back to this and make a different decision
                    print(msg)
                    logger.warning(msg)
                    continue

            raise ValueError(
                f"Generated input file differs on line {lineno}.\n"
                f"Expected line:  {line_expected!r}\n"
                f"Generated line: {line_generated!r}\n"
                f"Differs in number of parts {len(expected_parts)} != {len(generated_parts)}"
            )

        assert np.allclose(generated_parts, expected_parts)

    if len(expected_lines) != len(generated_lines):
        raise ValueError(
            f"Expected line count {len(expected_lines)} differs "
            f"vs generated lines {len(generated_lines)}"
        )


example_filenames = pytest.mark.parametrize(
    ("filename",),
    [
        pytest.param(fn, id=fn.name)
        for fn in impact_z_examples.glob("*.in")
        if fn.name not in ("particle1.in",)
    ],
)


@example_filenames
def test_load_input_smoke(filename: pathlib.Path) -> None:
    ImpactZInput.from_file(filename)


@example_filenames
def test_input_roundtrip(filename: pathlib.Path) -> None:
    loaded = ImpactZInput.from_file(filename)
    with open(filename, "rt") as fp:
        on_disk_contents = fp.read()

    serialized_contents = loaded.to_contents()
    compare_inputs(on_disk_contents, serialized_contents)


@example_filenames
def test_set_ncpu(filename: pathlib.Path) -> None:
    loaded = ImpactZInput.from_file(filename)
    loaded.verbose = True
    for nproc in range(0, 30):
        loaded.nproc = nproc
        print("Set numprocs", nproc, loaded.ncpu_y, loaded.ncpu_z)


def test_write_particles_initial_final() -> None:
    input = ImpactZInput(
        lattice=[
            Quadrupole(name="foo"),
            Quadrupole(name="bar"),
        ]
    )

    input.write_particles_at(start_file_id=1)
    assert input.lattice == [
        WriteFull(name="initial_particles", file_id=1),
        Quadrupole(name="foo"),
        Quadrupole(name="bar"),
        WriteFull(name="final_particles", file_id=2),
    ]


def test_write_particles_at_every() -> None:
    input = ImpactZInput(
        lattice=[
            Quadrupole(name="foo"),
            Quadrupole(name="bar"),
        ]
    )

    input.write_particles_at(every=Quadrupole, start_file_id=1)
    assert input.lattice == [
        WriteFull(name="initial_particles", file_id=1),
        Quadrupole(name="foo"),
        WriteFull(name="foo_WRITE", file_id=2),
        Quadrupole(name="bar"),
        WriteFull(name="bar_WRITE", file_id=3),
        WriteFull(name="final_particles", file_id=4),
    ]


def test_write_particles_at_foo() -> None:
    input = ImpactZInput(
        lattice=[
            Quadrupole(name="foo"),
            Quadrupole(name="bar"),
        ]
    )

    input.write_particles_at("foo", start_file_id=1)
    assert input.lattice == [
        WriteFull(name="initial_particles", file_id=1),
        Quadrupole(name="foo"),
        WriteFull(name="foo_WRITE", file_id=2),
        Quadrupole(name="bar"),
        WriteFull(name="final_particles", file_id=3),
    ]


def test_write_particles_at_every_multi() -> None:
    input = ImpactZInput(
        lattice=[
            Quadrupole(name="foo"),
            Drift(name="drift"),
            Quadrupole(name="bar"),
        ]
    )

    input.write_particles_at("foo", every=Drift, start_file_id=1)
    assert input.lattice == [
        WriteFull(name="initial_particles", file_id=1),
        Quadrupole(name="foo"),
        WriteFull(name="foo_WRITE", file_id=2),
        Drift(name="drift"),
        WriteFull(name="drift_WRITE", file_id=3),
        Quadrupole(name="bar"),
        WriteFull(name="final_particles", file_id=4),
    ]


def test_lattice_list_setattr():
    input = ImpactZInput(
        lattice=[
            Quadrupole(name="a"),
            Drift(name="drift"),
            Quadrupole(name="b"),
        ]
    )

    assert input.quadrupoles.name == ["a", "b"]
    input.quadrupoles.name = ["c", "d"]
    assert input.quadrupoles.name == ["c", "d"]
    assert input.quadrupoles[0].name == "c"
    assert input.quadrupoles[1].name == "d"

    assert input.quadrupoles.length == [0.0, 0.0]

    # broadcasting a single value to all
    input.quadrupoles.length = 1.0
    assert input.quadrupoles.length == [1.0, 1.0]
    assert input.quadrupoles[0].length == 1.0
    assert input.quadrupoles[1].length == 1.0

    # setting a list of values
    input.quadrupoles.length = [3.0, 4.0]
    assert input.quadrupoles.length == [3.0, 4.0]
    assert input.quadrupoles[0].length == 3.0
    assert input.quadrupoles[1].length == 4.0


@example_filenames
def test_smoke_calculated(filename: pathlib.Path) -> None:
    loaded = ImpactZInput.from_file(filename)
    assert isinstance(loaded.sigma_t, float)
    assert isinstance(loaded.sigma_energy, float)
    assert isinstance(loaded.cov_t__energy, float)


@example_filenames
def test_set_twiss_z(filename: pathlib.Path) -> None:
    input = ImpactZInput.from_file(filename)

    emit0, alpha0, beta0 = (
        input.twiss_norm_emit_z,
        input.twiss_alpha_z,
        input.twiss_beta_z,
    )

    try:
        input.set_twiss_z(
            sigma_t=input.sigma_t,
            sigma_energy=input.sigma_energy,
            cov_t__energy=input.cov_t__energy,
        )
    except Exception:
        if filename.name == "example3.in":
            # Expected as emit <= 0
            return
        raise

    emit1, alpha1, beta1 = (
        input.twiss_norm_emit_z,
        input.twiss_alpha_z,
        input.twiss_beta_z,
    )

    assert np.isclose(emit0, emit1)
    assert np.isclose(alpha0, alpha1)
    assert np.isclose(beta0, beta1)


@example_filenames
def test_plot_layout(filename: pathlib.Path) -> None:
    input = ImpactZInput.from_file(filename)
    input.plot(figsize=(20, 3))
    plt.show()


@pytest.mark.parametrize(
    ("include_labels", "include_markers"),
    [
        pytest.param(True, False, id="labels-no-markers"),
        pytest.param(False, False, id="no-labels-no-markers"),
        pytest.param(True, True, id="labels-and-markers"),
        pytest.param(False, True, id="no-labels-and-markers"),
    ],
)
def test_plot_layout_all(include_labels: bool, include_markers: bool) -> None:
    from ...z import input as IZ

    input = ImpactZInput(
        lattice=[
            IZ.Drift(length=1.0),
            IZ.Quadrupole(length=1.0),
            IZ.ConstantFocusing(length=1.0),
            IZ.Solenoid(length=1.0),
            IZ.Dipole(length=1.0),
            IZ.Multipole(length=1.0),
            IZ.DTL(length=1.0),
            IZ.CCDTL(length=1.0),
            IZ.CCL(length=1.0),
            IZ.SuperconductingCavity(length=1.0),
            IZ.SolenoidWithRFCavity(length=1.0),
            IZ.TravelingWaveRFCavity(length=1.0),
            IZ.UserDefinedRFCavity(length=1.0),
            # Control inputs
            IZ.ShiftCentroid(length=1.0),
            IZ.WriteFull(length=0.0),
            IZ.DensityProfileInput(length=1.0),
            IZ.DensityProfile(length=0.0),
            IZ.Projection2D(length=1.0),
            IZ.Density3D(length=0.0),
            IZ.WritePhaseSpaceInfo(length=1.0),
            IZ.WriteSliceInfo(length=1.0),
            IZ.ScaleMismatchParticle6DCoordinates(length=1.0),
            IZ.CollimateBeam(length=1.0),
            IZ.ToggleSpaceCharge(length=1.0),
            IZ.RotateBeam(length=1.0),
            IZ.BeamShift(length=1.0),
            IZ.BeamEnergySpread(length=1.0),
            IZ.ShiftBeamCentroid(length=1.0),
            IZ.IntegratorTypeSwitch(length=1.0),
            IZ.BeamKickerByRFNonlinearity(length=1.0),
            IZ.RfcavityStructureWakefield(length=1.0),
            IZ.EnergyModulation(length=1.0),
            IZ.KickBeamUsingMultipole(length=1.0),
            IZ.HaltExecution(length=1.0),
        ]
    )
    for ele in input.lattice:
        ele.name = f"{type(ele).__name__}_0"
    input.plot(
        figsize=(20, 3),
        include_labels=include_labels,
        include_markers=include_markers,
    )
    plt.show()


def test_check_names():
    contents = """1 1
-1 1 2 0 2
64 64 64 1 0 0 0
3 0 0 0
1
0.0
-1.9569511835591837e-06
0 10 9.9999999999999995475e-07 1 1 0 0
0 10 9.9999999999999995475e-07 1 1 0 0
1.0000000000000000623e-09 1 9.9999999999999995475e-07 1 1 0 0
0 999489001.04999995232 510998.94999999995343 -1 1300000000 0
0 0 0 -14 0 0 /
0 0 100 -2 0 0 / initial_particles
1 100 10 6 1 0.0010000000000000000208 -1 0.02999999999999999889 0 0.10000000000000000555 0 0 0 0 0 / UND
0 0 200 -2 0 0 / WRITE_END
0 0 101 -2 0 0 / final_particles
"""
    i1 = ImpactZInput.from_contents(contents)

    assert i1.wiggler.name == "UND"
    assert i1.write_fulls.name == ["initial_particles", "WRITE_END", "final_particles"]
