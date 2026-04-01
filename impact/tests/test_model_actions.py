"""Unit tests for impact.model.actions."""

import pytest
from unittest.mock import MagicMock
from lume.variables import NDVariable, ParticleGroupVariable, ScalarVariable

from impact.model.actions import (
    EleAction,
    HeaderAction,
    ImpactAction,
    ParticleGroupAction,
    RunInfoAction,
    StatAction,
    WritableImpactAction,
)
from impact.model.base import Action, WritableAction


@pytest.fixture
def impact():
    m = MagicMock()
    m.ele = {"Q1": {"b1_gradient": 1.5, "radius": 0.02}}
    m.header = {"Bcurr": 0.1, "Np": 1000}
    m.stat.return_value = MagicMock(shape=(100,))
    m.output = {"run_info": {"run_time": 3.2, "error": False}}
    m.particles = {
        "initial_particles": MagicMock(),
        "final_particles": MagicMock(),
    }
    return m


def scalar_var(name="test", read_only=False):
    return ScalarVariable(name=name, default_value=0.0, read_only=read_only)


def nd_var(name="test_nd"):
    return NDVariable(name=name, shape=(100,), default_value=None, read_only=True)


def pg_var(name="test_pg", read_only=False):
    return ParticleGroupVariable(name=name, default_value=None, read_only=read_only)


# ------------------------------------------------------------------
# Base class validation
# ------------------------------------------------------------------


def test_read_only_action_requires_read_only_var():
    with pytest.raises(ValueError, match="read-only"):
        StatAction(stat_name="mean_x", var=scalar_var(read_only=False))


def test_read_only_action_accepts_read_only_var():
    action = StatAction(stat_name="mean_x", var=nd_var())
    assert action.read_only is True


def test_writable_action_accepts_writable_var():
    action = EleAction(
        ele_name="Q1", attribute="b1_gradient", var=scalar_var(read_only=False)
    )
    assert action.read_only is False


def test_writable_action_also_accepts_read_only_var():
    action = HeaderAction(key="Np", var=scalar_var(read_only=True))
    assert action.read_only is True


def test_impact_action_is_generic_action():
    assert isinstance(StatAction(stat_name="mean_x", var=nd_var()), Action)


def test_writable_impact_action_is_generic_writable_action():
    assert isinstance(
        EleAction(ele_name="Q1", attribute="b1_gradient", var=scalar_var()),
        WritableAction,
    )


def test_all_concrete_actions_are_impact_action():
    assert isinstance(
        EleAction(ele_name="Q1", attribute="b1_gradient", var=scalar_var()),
        ImpactAction,
    )
    assert isinstance(HeaderAction(key="Bcurr", var=scalar_var()), ImpactAction)
    assert isinstance(StatAction(stat_name="mean_x", var=nd_var()), ImpactAction)
    assert isinstance(
        RunInfoAction(key="run_time", var=scalar_var(read_only=True)), ImpactAction
    )
    assert isinstance(
        ParticleGroupAction(tool_name="initial_particles", var=pg_var()), ImpactAction
    )


def test_writable_actions_are_writable_impact_action():
    assert isinstance(
        EleAction(ele_name="Q1", attribute="b1_gradient", var=scalar_var()),
        WritableImpactAction,
    )
    assert isinstance(HeaderAction(key="Bcurr", var=scalar_var()), WritableImpactAction)
    assert isinstance(
        ParticleGroupAction(tool_name="initial_particles", var=pg_var()),
        WritableImpactAction,
    )


def test_read_only_actions_are_not_writable_impact_action():
    assert not isinstance(
        StatAction(stat_name="mean_x", var=nd_var()), WritableImpactAction
    )
    assert not isinstance(
        RunInfoAction(key="run_time", var=scalar_var(read_only=True)),
        WritableImpactAction,
    )


# ------------------------------------------------------------------
# safe_set guards
# ------------------------------------------------------------------


def test_safe_set_raises_for_read_only_var(impact):
    action = HeaderAction(key="Np", var=scalar_var(read_only=True))
    with pytest.raises(TypeError, match="read-only"):
        action.safe_set(impact, 500)


def test_safe_set_succeeds_for_writable_var(impact):
    action = HeaderAction(key="Bcurr", var=scalar_var(read_only=False))
    action.safe_set(impact, 0.5)
    assert impact.header["Bcurr"] == 0.5


def test_set_bypasses_read_only_guard(impact):
    # set() is the raw implementation with no guard; safe_set is what models call
    action = HeaderAction(key="Np", var=scalar_var(read_only=True))
    action.set(impact, 999)
    assert impact.header["Np"] == 999


# ------------------------------------------------------------------
# Construction validation
# ------------------------------------------------------------------


def test_stat_action_requires_read_only_var():
    with pytest.raises(ValueError, match="read-only"):
        StatAction(stat_name="mean_x", var=scalar_var(read_only=False))


def test_run_info_action_requires_read_only_var():
    with pytest.raises(ValueError, match="read-only"):
        RunInfoAction(key="run_time", var=scalar_var(read_only=False))


def test_particle_group_non_initial_requires_read_only_var():
    with pytest.raises(ValueError, match="not writable"):
        ParticleGroupAction(tool_name="final_particles", var=pg_var(read_only=False))


def test_particle_group_initial_accepts_writable_var():
    action = ParticleGroupAction(
        tool_name="initial_particles", var=pg_var(read_only=False)
    )
    assert action.read_only is False


def test_particle_group_non_initial_accepts_read_only_var():
    action = ParticleGroupAction(
        tool_name="final_particles", var=pg_var(read_only=True)
    )
    assert action.read_only is True


# ------------------------------------------------------------------
# EleAction
# ------------------------------------------------------------------


def test_ele_get(impact):
    action = EleAction(ele_name="Q1", attribute="b1_gradient", var=scalar_var())
    assert action.get(impact) == 1.5


def test_ele_set(impact):
    action = EleAction(ele_name="Q1", attribute="b1_gradient", var=scalar_var())
    action.set(impact, 2.0)
    assert impact.ele["Q1"]["b1_gradient"] == 2.0


def test_ele_name():
    action = EleAction(ele_name="Q1", attribute="b1_gradient", var=scalar_var("my_var"))
    assert action.name == "my_var"


# ------------------------------------------------------------------
# HeaderAction
# ------------------------------------------------------------------


def test_header_get(impact):
    action = HeaderAction(key="Bcurr", var=scalar_var())
    assert action.get(impact) == 0.1


def test_header_set(impact):
    action = HeaderAction(key="Bcurr", var=scalar_var())
    action.set(impact, 0.5)
    assert impact.header["Bcurr"] == 0.5


def test_header_read_only_raises(impact):
    action = HeaderAction(key="Np", var=scalar_var(read_only=True))
    with pytest.raises(TypeError, match="read-only"):
        action.safe_set(impact, 500)


# ------------------------------------------------------------------
# StatAction
# ------------------------------------------------------------------


def test_stat_get(impact):
    action = StatAction(stat_name="mean_x", var=nd_var())
    result = action.get(impact)
    impact.stat.assert_called_once_with("mean_x")
    assert result is impact.stat.return_value


def test_stat_read_only():
    action = StatAction(stat_name="mean_x", var=nd_var())
    assert action.read_only is True


# ------------------------------------------------------------------
# RunInfoAction
# ------------------------------------------------------------------


def test_run_info_get(impact):
    action = RunInfoAction(key="run_time", var=scalar_var(read_only=True))
    assert action.get(impact) == 3.2


# ------------------------------------------------------------------
# ParticleGroupAction
# ------------------------------------------------------------------


def test_particle_group_get_initial(impact):
    action = ParticleGroupAction(tool_name="initial_particles", var=pg_var())
    assert action.get(impact) is impact.particles["initial_particles"]


def test_particle_group_get_final(impact):
    action = ParticleGroupAction(
        tool_name="final_particles", var=pg_var(read_only=True)
    )
    assert action.get(impact) is impact.particles["final_particles"]


def test_particle_group_set_initial(impact):
    action = ParticleGroupAction(tool_name="initial_particles", var=pg_var())
    new_pg = MagicMock()
    action.set(impact, new_pg)
    assert impact.initial_particles == new_pg


def test_particle_group_set_read_only_raises(impact):
    action = ParticleGroupAction(
        tool_name="final_particles", var=pg_var(read_only=True)
    )
    with pytest.raises(TypeError, match="read-only"):
        action.safe_set(impact, MagicMock())
