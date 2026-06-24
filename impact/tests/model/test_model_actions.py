"""Unit tests for impact.model.actions."""

import pytest
from unittest.mock import MagicMock

from impact.model.actions import (
    HeaderAction,
    ParticleGroupAction,
    ScalarEleAction as EleAction,
    ScalarRunInfoAction as RunInfoAction,
    StatAction,
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


# ------------------------------------------------------------------
# EleAction
# ------------------------------------------------------------------


def test_ele_get(impact):
    action = EleAction(
        ele_name="Q1", attribute="b1_gradient", name="test", default_value=0.0
    )
    assert action._get(impact) == 1.5


def test_ele_set(impact):
    action = EleAction(
        ele_name="Q1", attribute="b1_gradient", name="test", default_value=0.0
    )
    action._set(impact, 2.0)
    assert impact.ele["Q1"]["b1_gradient"] == 2.0


def test_ele_name():
    action = EleAction(
        ele_name="Q1", attribute="b1_gradient", name="my_var", default_value=0.0
    )
    assert action.name == "my_var"


# ------------------------------------------------------------------
# HeaderAction
# ------------------------------------------------------------------


def test_header_get(impact):
    action = HeaderAction(key="Bcurr", name="test", default_value=0.0)
    assert action._get(impact) == 0.1


def test_header_set(impact):
    action = HeaderAction(key="Bcurr", name="test", default_value=0.0)
    action._set(impact, 0.5)
    assert impact.header["Bcurr"] == 0.5


def test_header_read_only_blocks_set_via_model(impact):
    from lume.actions import ActionModel

    action = HeaderAction(key="Np", name="test", default_value=0.0, read_only=True)
    model = ActionModel(simulator=impact, action_variables=[action])
    with pytest.raises(Exception):
        model.set({"test": 500})


# ------------------------------------------------------------------
# StatAction
# ------------------------------------------------------------------


def test_stat_get(impact):
    action = StatAction(
        stat_name="mean_x",
        name="test_nd",
        shape=(100,),
        default_value=None,
        read_only=True,
    )
    result = action._get(impact)
    impact.stat.assert_called_once_with("mean_x")
    assert result is impact.stat.return_value


def test_stat_read_only():
    action = StatAction(
        stat_name="mean_x",
        name="test_nd",
        shape=(100,),
        default_value=None,
        read_only=True,
    )
    assert action.read_only is True


# ------------------------------------------------------------------
# RunInfoAction
# ------------------------------------------------------------------


def test_run_info_get(impact):
    action = RunInfoAction(
        key="run_time", name="test", default_value=None, read_only=True
    )
    assert action._get(impact) == 3.2


# ------------------------------------------------------------------
# ParticleGroupAction
# ------------------------------------------------------------------


def test_particle_group_get_initial(impact):
    action = ParticleGroupAction(
        tool_name="initial_particles", name="test_pg", default_value=None
    )
    assert action._get(impact) is impact.particles["initial_particles"]


def test_particle_group_get_final(impact):
    action = ParticleGroupAction(
        tool_name="final_particles", name="test_pg", default_value=None, read_only=True
    )
    assert action._get(impact) is impact.particles["final_particles"]


def test_particle_group_set_initial(impact):
    action = ParticleGroupAction(
        tool_name="initial_particles", name="test_pg", default_value=None
    )
    new_pg = MagicMock()
    action._set(impact, new_pg)
    assert impact.initial_particles == new_pg
