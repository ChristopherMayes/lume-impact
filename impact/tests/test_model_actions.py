"""Unit tests for impact.model.actions."""

import pytest
from unittest.mock import MagicMock
from lume.variables import NDVariable, ParticleGroupVariable, ScalarVariable

from impact.model.actions import (
    EleAction,
    HeaderAction,
    ParticleGroupAction,
    RunInfoAction,
    StatAction,
    WritableAction,
)


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


def test_ele_is_writable_action():
    action = EleAction(ele_name="Q1", attribute="b1_gradient", var=scalar_var())
    assert isinstance(action, WritableAction)


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
        action.set(impact, 500)


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


def test_stat_has_no_set():
    action = StatAction(stat_name="mean_x", var=nd_var())
    assert not isinstance(action, WritableAction)


# ------------------------------------------------------------------
# RunInfoAction
# ------------------------------------------------------------------


def test_run_info_get(impact):
    action = RunInfoAction(key="run_time", var=scalar_var(read_only=True))
    assert action.get(impact) == 3.2


def test_run_info_has_no_set():
    action = RunInfoAction(key="run_time", var=scalar_var(read_only=True))
    assert not isinstance(action, WritableAction)


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
        action.set(impact, MagicMock())
