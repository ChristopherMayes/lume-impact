"""Unit tests for impact.model.actions."""

import pytest
from unittest.mock import MagicMock
from lume.variables import NDVariable, ParticleGroupVariable, ScalarVariable

from impact.model.actions import (
    EleVarAction,
    HeaderVarAction,
    ParticleGroupVarAction,
    RunInfoVarAction,
    StatVarAction,
)


@pytest.fixture
def imp():
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
# EleVarAction
# ------------------------------------------------------------------


def test_ele_get(imp):
    action = EleVarAction(ele_name="Q1", attribute="b1_gradient", var=scalar_var())
    assert action.get(imp) == 1.5


def test_ele_set(imp):
    action = EleVarAction(ele_name="Q1", attribute="b1_gradient", var=scalar_var())
    action.set(imp, 2.0)
    assert imp.ele["Q1"]["b1_gradient"] == 2.0


def test_ele_name():
    action = EleVarAction(
        ele_name="Q1", attribute="b1_gradient", var=scalar_var("my_var")
    )
    assert action.name == "my_var"


def test_ele_not_read_only():
    action = EleVarAction(ele_name="Q1", attribute="b1_gradient", var=scalar_var())
    assert action.read_only is False


def test_header_get(imp):
    action = HeaderVarAction(key="Bcurr", var=scalar_var())
    assert action.get(imp) == 0.1


def test_header_set(imp):
    action = HeaderVarAction(key="Bcurr", var=scalar_var())
    action.set(imp, 0.5)
    assert imp.header["Bcurr"] == 0.5


def test_header_read_only_raises(imp):
    action = HeaderVarAction(key="Np", var=scalar_var(read_only=True))
    with pytest.raises(TypeError, match="read-only"):
        action.set(imp, 500)


def test_stat_get(imp):
    action = StatVarAction(stat_name="mean_x", var=nd_var())
    result = action.get(imp)
    imp.stat.assert_called_once_with("mean_x")
    assert result is imp.stat.return_value


def test_stat_set_raises(imp):
    action = StatVarAction(stat_name="mean_x", var=nd_var())
    with pytest.raises(TypeError, match="read-only"):
        action.set(imp, None)


def test_stat_read_only():
    action = StatVarAction(stat_name="mean_x", var=nd_var())
    assert action.read_only is True


def test_run_info_get(imp):
    action = RunInfoVarAction(key="run_time", var=scalar_var())
    assert action.get(imp) == 3.2


def test_run_info_set_raises(imp):
    action = RunInfoVarAction(key="run_time", var=scalar_var())
    with pytest.raises(TypeError, match="read-only"):
        action.set(imp, 0.0)


def test_particle_group_get_initial(imp):
    action = ParticleGroupVarAction(tool_name="initial_particles", var=pg_var())
    assert action.get(imp) is imp.particles["initial_particles"]


def test_particle_group_get_final(imp):
    action = ParticleGroupVarAction(
        tool_name="final_particles", var=pg_var(read_only=True)
    )
    assert action.get(imp) is imp.particles["final_particles"]


def test_particle_group_set_initial(imp):
    action = ParticleGroupVarAction(tool_name="initial_particles", var=pg_var())
    new_pg = MagicMock()
    action.set(imp, new_pg)
    assert imp.initial_particles == new_pg


def test_particle_group_set_non_initial_raises(imp):
    action = ParticleGroupVarAction(
        tool_name="final_particles", var=pg_var(read_only=True)
    )
    with pytest.raises(TypeError, match="read-only"):
        action.set(imp, MagicMock())
