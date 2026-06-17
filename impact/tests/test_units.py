"""Tests for the centralized unit dictionaries.

Static units for header keys, element attributes, and run-info keys live
in module-level dicts (``HEADER_UNITS``, ``ELE_UNITS``, ``RUN_INFO_UNITS``)
and are loaded into ``Impact._units`` at construction time. This file
verifies the dicts are self-consistent and reachable via ``Impact.units``.
"""

import pytest

from impact import Impact
from impact.impact import RUN_INFO_UNITS, STATIC_UNITS
from impact.parsers import (
    ELE_DEFAULTS,
    ELE_UNITS,
    HEADER_DEFAULT,
    HEADER_UNITS,
)


# ----------
# HEADER_UNITS
# ----------


def test_header_units_keys_are_known_header_keys():
    """Every key in HEADER_UNITS must correspond to a real header field."""
    unknown = set(HEADER_UNITS) - set(HEADER_DEFAULT)
    assert not unknown, f"HEADER_UNITS has unknown keys: {sorted(unknown)}"


def test_header_units_values_are_nonempty_strings():
    for k, v in HEADER_UNITS.items():
        assert isinstance(v, str) and v, f"HEADER_UNITS[{k!r}] is not a unit string"


# ----------
# ELE_UNITS
# ----------


def test_ele_units_keys_are_known_element_attributes():
    """Every key in ELE_UNITS must appear in at least one element type's
    default-attribute dict."""
    all_ele_attrs = set()
    for type_defaults in ELE_DEFAULTS.values():
        all_ele_attrs.update(type_defaults)
    unknown = set(ELE_UNITS) - all_ele_attrs
    assert not unknown, f"ELE_UNITS has unknown keys: {sorted(unknown)}"


def test_ele_units_values_are_nonempty_strings():
    for k, v in ELE_UNITS.items():
        assert isinstance(v, str) and v, f"ELE_UNITS[{k!r}] is not a unit string"


# ----------
# RUN_INFO_UNITS
# ----------


def test_run_info_units_values_are_nonempty_strings():
    for k, v in RUN_INFO_UNITS.items():
        assert isinstance(v, str) and v, f"RUN_INFO_UNITS[{k!r}] is not a unit string"


# ----------
# STATIC_UNITS aggregation
# ----------


def test_static_units_covers_all_three_sources():
    """STATIC_UNITS must include every key from HEADER_UNITS, ELE_UNITS,
    and RUN_INFO_UNITS."""
    for src in (HEADER_UNITS, ELE_UNITS, RUN_INFO_UNITS):
        missing = set(src) - set(STATIC_UNITS)
        assert not missing, f"STATIC_UNITS missing keys: {sorted(missing)}"


def test_static_units_keeps_extra_units():
    """Bz/Ez from EXTRA_UNITS must still be present after aggregation."""
    assert "Bz" in STATIC_UNITS
    assert "Ez" in STATIC_UNITS


# ----------
# Impact.units lookup
# ----------


@pytest.mark.parametrize(
    "key,expected_str",
    [
        ("Dt", "s"),
        ("Bcurr", "A"),
        ("sigx", "m"),
        ("zedge", "m"),
        ("b1_gradient", "T/m"),
        ("rf_frequency", "Hz"),
        ("x_rotation", "rad"),
        ("run_time", "s"),
    ],
)
def test_impact_units_lookup(key, expected_str):
    """``Impact.units(key)`` returns a unit whose string form matches the
    declared unit for keys spanning header, element, and run_info dicts."""
    impact = Impact(verbose=False)
    u = impact.units(key)
    assert str(u) == expected_str
