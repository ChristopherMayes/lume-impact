import os

import pytest
from distgen import Generator
from lume.variables import ScalarVariable

from impact import Impact
from impact.model.distgen.actions import DistgenInputAction
from impact.model.distgen.config import DistgenVariableMappingConfig
from impact.model.distgen.distgen_impact_model import LUMEDistgenImpactModel
from impact.model.distgen.model import LUMEDistgenModel
from impact.model.actions import HeaderAction


DISTGEN_YAML = os.path.join(
    os.path.dirname(__file__),
    "../../../docs/examples/templates/lcls_injector/distgen.yaml",
)


@pytest.fixture(scope="module")
def gen():
    return Generator(DISTGEN_YAML)


@pytest.fixture(scope="module")
def distgen_model(gen):
    return LUMEDistgenModel.from_generator(gen, dummy_run=True)


@pytest.fixture(scope="module")
def fast_impact():
    I = Impact()
    I.header["Np"] = 100
    I.header["Bcurr"] = 0
    I.run()
    return I


@pytest.fixture(scope="module")
def combined_model(gen, fast_impact):
    return LUMEDistgenImpactModel.from_objects(gen, fast_impact, dummy_run=True)


# LUMEDistgenModel — variables present


def test_distgen_model_nonempty(distgen_model):
    assert len(distgen_model.supported_variables) > 0


def test_root_vars_present(distgen_model):
    for name in ("distgen/n_particle", "distgen/total_charge"):
        assert name in distgen_model.supported_variables


def test_start_cathode_mte_present(distgen_model):
    assert "distgen/start/cathode/MTE" in distgen_model.supported_variables


def test_r_dist_sigma_xy_present(distgen_model):
    assert "distgen/r_dist/sigma_xy" in distgen_model.supported_variables


def test_t_dist_vars_present(distgen_model):
    for name in ("distgen/t_dist/length", "distgen/t_dist/ratio"):
        assert name in distgen_model.supported_variables


# LUMEDistgenModel — values match generator


def test_n_particle_value_matches(gen, distgen_model):
    assert (
        distgen_model._get(["distgen/n_particle"])["distgen/n_particle"]
        == gen["n_particle"]
    )


def test_total_charge_value_matches(gen, distgen_model):
    assert (
        distgen_model._get(["distgen/total_charge"])["distgen/total_charge"]
        == gen["total_charge:value"]
    )


def test_start_mte_value_matches(gen, distgen_model):
    assert (
        distgen_model._get(["distgen/start/cathode/MTE"])["distgen/start/cathode/MTE"]
        == gen["start:MTE:value"]
    )


# LUMEDistgenModel — set/get round-trip


def test_set_and_get_n_particle(gen):
    model = LUMEDistgenModel.from_generator(gen, dummy_run=True)
    original = gen["n_particle"]
    model._set({"distgen/n_particle": 500})
    assert model._get(["distgen/n_particle"])["distgen/n_particle"] == 500
    gen["n_particle"] = original


def test_set_and_get_total_charge(gen):
    model = LUMEDistgenModel.from_generator(gen, dummy_run=True)
    original = gen["total_charge:value"]
    model._set({"distgen/total_charge": 100.0})
    assert model._get(["distgen/total_charge"])[
        "distgen/total_charge"
    ] == pytest.approx(100.0)
    gen["total_charge:value"] = original


# LUMEDistgenModel — config exclusions


def test_no_vars_when_inputs_none(gen):
    config = DistgenVariableMappingConfig(inputs=None)
    model = LUMEDistgenModel.from_generator(gen, config=config, dummy_run=True)
    assert len(model.supported_variables) == 0


# LUMEDistgenModel — register_action


def test_register_new_distgen_action(gen):
    model = LUMEDistgenModel.from_generator(gen, dummy_run=True)
    action = DistgenInputAction(
        key="n_particle",
        has_units=False,
        var=ScalarVariable(name="distgen/n_particle_custom", default_value=None),
    )
    model.register_action(action)
    assert "distgen/n_particle_custom" in model.supported_variables


def test_register_distgen_action_replaces_existing(gen):
    model = LUMEDistgenModel.from_generator(gen, dummy_run=True)
    count_before = len(model.actions)
    action = DistgenInputAction(
        key="n_particle",
        has_units=False,
        var=ScalarVariable(name="distgen/n_particle", default_value=None),
    )
    model.register_action(action)
    assert len(model.actions) == count_before


# LUMEDistgenImpactModel — variables present from both sides


def test_combined_model_has_distgen_vars(combined_model):
    distgen_vars = [
        n for n in combined_model.supported_variables if n.startswith("distgen/")
    ]
    assert len(distgen_vars) > 0


def test_combined_model_has_impact_vars(combined_model):
    impact_vars = [
        n
        for n in combined_model.supported_variables
        if n.startswith("header/") or n.startswith("ele/")
    ]
    assert len(impact_vars) > 0


def test_combined_model_has_n_particle(combined_model):
    assert "distgen/n_particle" in combined_model.supported_variables


def test_combined_model_has_header_bcurr(combined_model):
    assert "header/Bcurr" in combined_model.supported_variables


# LUMEDistgenImpactModel — set routes to the correct side


def test_set_distgen_var_updates_gen(gen, fast_impact):
    model = LUMEDistgenImpactModel.from_objects(gen, fast_impact, dummy_run=True)
    original = gen["n_particle"]
    model._set({"distgen/n_particle": 42})
    assert gen["n_particle"] == 42
    gen["n_particle"] = original


def test_set_impact_var_updates_impact(gen, fast_impact):
    model = LUMEDistgenImpactModel.from_objects(gen, fast_impact, dummy_run=True)
    original = fast_impact.header["Np"]
    model._set({"header/Np": 77})
    assert fast_impact.header["Np"] == 77
    fast_impact.header["Np"] = original


# LUMEDistgenImpactModel — register_action routing


def test_register_distgen_action_on_combined(gen, fast_impact):
    model = LUMEDistgenImpactModel.from_objects(gen, fast_impact, dummy_run=True)
    action = DistgenInputAction(
        key="n_particle",
        has_units=False,
        var=ScalarVariable(name="distgen/n_particle_v2", default_value=None),
    )
    model.register_action(action)
    assert "distgen/n_particle_v2" in model.supported_variables
    assert "distgen/n_particle_v2" in model._distgen_by_name


def test_register_impact_action_on_combined(gen, fast_impact):
    model = LUMEDistgenImpactModel.from_objects(gen, fast_impact, dummy_run=True)
    action = HeaderAction(
        key="Ntstep",
        var=ScalarVariable(name="header/Ntstep", default_value=1000),
    )
    model.register_action(action)
    assert "header/Ntstep" in model.supported_variables
    assert "header/Ntstep" in model._impact_by_name
