import os

import pytest

from lume.variables import ScalarVariable

from impact import Impact
from impact.model.actions import HeaderAction
from impact.model.config import VariableMappingConfig
from impact.model.model import LUMEImpactModel


TESLA_INPUT = os.path.join(
    os.path.dirname(__file__), "input", "tesla_9cell_cavity", "ImpactT.in"
)


@pytest.fixture(scope="module")
def fast_impact():
    I = Impact()
    I.header["Np"] = 100
    I.header["Bcurr"] = 0
    I.run()
    return I


@pytest.fixture(scope="module")
def default_model(fast_impact):
    return LUMEImpactModel.from_impact(fast_impact, dummy_run=True)


@pytest.fixture(scope="module")
def tesla_impact():
    I = Impact(TESLA_INPUT)
    I.header["Np"] = 100
    I.run()
    return I


@pytest.fixture(scope="module")
def tesla_model(tesla_impact):
    return LUMEImpactModel.from_impact(tesla_impact, dummy_run=True)


# ---------------------------------------------------------------------------
# Header variables
# ---------------------------------------------------------------------------


def test_header_vars_present(default_model):
    for name in ("header/Bcurr", "header/Np", "header/Bkenergy"):
        assert name in default_model.supported_variables


def test_header_value_matches_impact(fast_impact, default_model):
    assert (
        default_model._get(["header/Bcurr"])["header/Bcurr"]
        == fast_impact.header["Bcurr"]
    )


# ---------------------------------------------------------------------------
# Element variables — default lattice has "drift_1"
# ---------------------------------------------------------------------------


def test_drift_vars_present(default_model):
    for name in ("ele/drift_1/zedge", "ele/drift_1/radius"):
        assert name in default_model.supported_variables


def test_drift_zedge_value_matches_impact(fast_impact, default_model):
    assert (
        default_model._get(["ele/drift_1/zedge"])["ele/drift_1/zedge"]
        == fast_impact.ele["drift_1"]["zedge"]
    )


# ---------------------------------------------------------------------------
# Stat output variables (read-only)
# ---------------------------------------------------------------------------


def test_stat_vars_present(default_model):
    for name in ("stat/mean_x", "stat/mean_kinetic_energy", "stat/norm_emit_x"):
        assert name in default_model.supported_variables


def test_stat_vars_are_read_only(default_model):
    stat_actions = [a for a in default_model.actions if a.name.startswith("stat/")]
    assert stat_actions
    assert all(a.read_only for a in stat_actions)


# ---------------------------------------------------------------------------
# Run-info variables (read-only)
# ---------------------------------------------------------------------------


def test_run_info_vars_present(default_model):
    for name in ("run_info/run_time", "run_info/error"):
        assert name in default_model.supported_variables


# ---------------------------------------------------------------------------
# set / get round-trip (dummy_run=True avoids calling Impact.run())
# ---------------------------------------------------------------------------


def test_set_and_get_header(fast_impact):
    model = LUMEImpactModel.from_impact(fast_impact, dummy_run=True)
    original = fast_impact.header["Np"]
    model._set({"header/Np": 200})
    assert model._get(["header/Np"])["header/Np"] == 200
    fast_impact.header["Np"] = original


# ---------------------------------------------------------------------------
# Config exclusions
# ---------------------------------------------------------------------------


def test_no_element_vars_when_elements_none(fast_impact):
    config = VariableMappingConfig(elements=None)
    model = LUMEImpactModel.from_impact(fast_impact, config=config, dummy_run=True)
    assert not any(n.startswith("ele/") for n in model.supported_variables)


def test_no_header_vars_when_header_none(fast_impact):
    config = VariableMappingConfig(header=None)
    model = LUMEImpactModel.from_impact(fast_impact, config=config, dummy_run=True)
    assert not any(n.startswith("header/") for n in model.supported_variables)


def test_no_stat_vars_when_stats_none(fast_impact):
    config = VariableMappingConfig(stats=None)
    model = LUMEImpactModel.from_impact(fast_impact, config=config, dummy_run=True)
    assert not any(n.startswith("stat/") for n in model.supported_variables)


# ---------------------------------------------------------------------------
# register_action
# ---------------------------------------------------------------------------


def test_register_new_action(fast_impact):
    model = LUMEImpactModel.from_impact(fast_impact, dummy_run=True)
    action = HeaderAction(
        key="Ntstep", var=ScalarVariable(name="header/Ntstep", default_value=1000)
    )
    model.register_action(action)
    assert "header/Ntstep" in model.supported_variables


def test_register_action_replaces_existing(fast_impact):
    model = LUMEImpactModel.from_impact(fast_impact, dummy_run=True)
    count_before = len(model.actions)
    action = HeaderAction(
        key="Bcurr", var=ScalarVariable(name="header/Bcurr", default_value=99.0)
    )
    model.register_action(action)
    assert len(model.actions) == count_before


# ---------------------------------------------------------------------------
# Tesla 9-cell cavity lattice
# ---------------------------------------------------------------------------


def test_tesla_has_element_vars(tesla_model):
    ele_vars = [n for n in tesla_model.supported_variables if n.startswith("ele/")]
    assert len(ele_vars) > 0


def test_tesla_has_rf_frequency_vars(tesla_model):
    rf_freq_vars = [
        n for n in tesla_model.supported_variables if n.endswith("/rf_frequency")
    ]
    assert len(rf_freq_vars) > 0
