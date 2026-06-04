"""Tests for backwards-compatible header key aliases in ``impact.parsers``."""

import warnings

import pytest

from impact.parsers import HEADER_ALIASES, HEADER_DEFAULT, header_bookkeeper


def test_alias_only_header_is_translated():
    """Deprecated-only keys are rewritten to canonical form."""
    out = header_bookkeeper({"sigx(m)": 1.5, "xmu1(m)": 2.5}, verbose=False)
    assert out["sigx"] == 1.5
    assert out["xmu1"] == 2.5
    assert "sigx(m)" not in out
    assert "xmu1(m)" not in out


def test_alias_after_canonical_overrides():
    """When alias is assigned after canonical, alias value wins."""
    out = header_bookkeeper({"sigx": 0.001, "sigx(m)": 9.9}, verbose=False)
    assert out["sigx"] == 9.9


def test_canonical_after_alias_overrides():
    """When canonical is assigned after alias, canonical value wins."""
    out = header_bookkeeper({"sigx(m)": 9.9, "sigx": 0.001}, verbose=False)
    assert out["sigx"] == 0.001


def test_alias_equal_values_no_conflict(capsys):
    """Equal canonical and alias values produce no conflict warning."""
    out = header_bookkeeper({"sigx": 1.0, "sigx(m)": 1.0}, verbose=True)
    assert out["sigx"] == 1.0
    captured = capsys.readouterr().out
    assert "both canonical and deprecated forms" not in captured


def test_alias_conflict_warning_emitted(capsys):
    """Differing canonical/alias values emit a warning naming both values."""
    header_bookkeeper({"sigx": 0.001, "sigx(m)": 9.9}, verbose=True)
    captured = capsys.readouterr().out
    assert "both canonical and deprecated forms" in captured
    assert "0.001" in captured
    assert "9.9" in captured


def test_all_aliases_round_trip():
    """Every deprecated alias is mapped to its canonical key."""
    header = {old: float(i) for i, old in enumerate(HEADER_ALIASES, start=1)}
    out = header_bookkeeper(header, verbose=False)
    for i, (old, canonical) in enumerate(HEADER_ALIASES.items(), start=1):
        assert out[canonical] == float(i)
        assert old not in out


def test_canonical_keys_are_in_defaults():
    """All alias targets exist as canonical keys in the default header."""
    for canonical in HEADER_ALIASES.values():
        assert canonical in HEADER_DEFAULT


def test_deprecated_alias_emits_deprecation_warning():
    """Encountering a deprecated key issues a DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match=r"sigx\(m\).*sigx"):
        header_bookkeeper({"sigx(m)": 1.0}, verbose=False)


def test_canonical_only_emits_no_deprecation_warning():
    """Canonical-only headers do not raise a DeprecationWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        header_bookkeeper({"sigx": 1.0}, verbose=False)
