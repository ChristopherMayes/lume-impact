from __future__ import annotations
import pathlib
import numpy as np
import pytest

from ... import z as IZ
from pmd_beamphysics import single_particle
from pmd_beamphysics.units import mec2


@pytest.mark.parametrize(
    "diagnostic_type",
    [
        pytest.param(IZ.DiagnosticType.standard, id="standard"),
        pytest.param(IZ.DiagnosticType.extended, id="extended"),
    ],
)
def test_diagnostic_type(diagnostic_type: IZ.DiagnosticType, tmp_path: pathlib.Path):
    energy = 10e6
    pz = np.sqrt(energy**2 - mec2**2)
    P0 = single_particle(x=1e-3, pz=pz)

    input = IZ.ImpactZInput(
        initial_particles=P0,
        ncpu_y=1,
        ncpu_z=1,
        seed=-1,
        n_particle=1,
        nx=64,
        ny=64,
        nz=64,
        diagnostic_type=diagnostic_type,
        distribution=IZ.DistributionType.read,
        twiss_beta_x=10.0,
        twiss_beta_y=10.0,
        average_current=0.0,
        reference_kinetic_energy=9489001.05,
        reference_particle_mass=510998.94999999995,
        reference_particle_charge=-1.0,
        reference_frequency=1300000000.0,
        lattice=[
            IZ.WriteFull(name="initial_particles", file_id=100),
            IZ.IntegratorTypeSwitch(integrator_type=IZ.IntegratorType.runge_kutta),
            IZ.Multipole(
                name="SEXTUPOLE1",
                length=0.6,
                steps=10,
                map_steps=10,
                field_strength=-0.0333128309370517,
                file_id=-1.0,
                radius=0.03,
            ),
            IZ.IntegratorTypeSwitch(),
            IZ.WriteFull(name="final_particles", file_id=101),
        ],
    )
    I = IZ.ImpactZ(input, workdir=tmp_path, use_temp_dir=False)
    output = I.run(verbose=False)
    stats = output.stats
    should_be_populated = {
        IZ.DiagnosticType.standard: [
            # file 27
            # "max_abs_x",
            "max_abs_px_over_p0",
            # "max_abs_y",  (common)
            "max_abs_py_over_p0",
            # "max_phase",  (common)
            "max_energy_dev",
            # file 29
            "moment3_x",
            "moment3_px_over_p0",
            "moment3_y",
            "moment3_py_over_p0",
            "moment3_phase",
            "moment3_energy",
            # file 30
            "moment4_x",
            "moment4_px_over_p0",
            "moment4_y",
            "moment4_py_over_p0",
            "moment4_phase",
            "moment4_energy",
        ],
        IZ.DiagnosticType.extended: [
            # file 24
            "norm_emit_x_90percent",
            "norm_emit_x_95percent",
            "norm_emit_x_99percent",
            # file 25
            "norm_emit_y_90percent",
            "norm_emit_y_95percent",
            "norm_emit_y_99percent",
            # file 26
            "norm_emit_z_90percent",
            "norm_emit_z_95percent",
            "norm_emit_z_99percent",
            # file 27
            # "max_abs_x", (common)
            "max_abs_gammabeta_x",
            # "max_abs_y",  (common)
            "max_abs_gammabeta_y",
            # "max_phase",  (common)
            "max_gamma_rel",
            # file 29
            "mean_r",
            "sigma_r",
            "mean_r_90percent",
            "mean_r_95percent",
            "mean_r_99percent",
            "max_r_rel",
        ],
    }
    for attr in should_be_populated[diagnostic_type]:
        arr = getattr(stats, attr)
        assert arr.shape == stats.z.shape, f"{attr} not populated?"

    other_diagnostic_type = {
        IZ.DiagnosticType.standard: IZ.DiagnosticType.extended,
        IZ.DiagnosticType.extended: IZ.DiagnosticType.standard,
    }[diagnostic_type]
    for attr in should_be_populated[other_diagnostic_type]:
        arr = getattr(stats, attr)
        assert not len(arr), f"{attr} should not be populated?"
