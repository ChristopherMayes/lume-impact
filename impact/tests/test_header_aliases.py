"""Tests for deprecated header key translation.

Translation from deprecated header keys (e.g. ``"sigx(m)"``) to their
canonical names (``"sigx"``) happens at the user-facing boundaries of the
``Impact`` class — ``__getitem__``/``__setitem__`` and archive loading.
The internal header dict is always canonical.
"""

import warnings

import pytest

from impact import Impact
from impact.impact import _normalize_header_keys
from impact.parsers import HEADER_ALIASES, HEADER_DEFAULT, header_bookkeeper


# ----------
# Aliases / defaults
# ----------


def test_canonical_keys_are_in_defaults():
    """All alias targets exist as canonical keys in the default header."""
    for canonical in HEADER_ALIASES.values():
        assert canonical in HEADER_DEFAULT


def test_header_bookkeeper_does_not_translate_aliases():
    """header_bookkeeper operates on canonical keys only and does not
    rewrite deprecated aliases (translation happens at the boundaries)."""
    out = header_bookkeeper({"sigx(m)": 1.5}, verbose=False)
    assert "sigx(m)" in out
    # And it also fills the canonical key from defaults
    assert out["sigx"] == HEADER_DEFAULT["sigx"]


# ----------
# _normalize_header_keys (used by load_archive)
# ----------


def test_normalize_rewrites_deprecated_keys_in_place():
    """Legacy alias keys are rewritten to canonical in place."""
    header = {"sigx(m)": 1.5, "Np": 100}
    with pytest.warns(DeprecationWarning, match=r"sigx\(m\).*sigx"):
        _normalize_header_keys(header)
    assert "sigx(m)" not in header
    assert header["sigx"] == 1.5
    assert header["Np"] == 100


def test_normalize_preserves_canonical_when_both_present():
    """If both forms exist, canonical value is preserved and alias is dropped."""
    header = {"sigx": 0.001, "sigx(m)": 9.9}
    with pytest.warns(DeprecationWarning):
        _normalize_header_keys(header)
    assert "sigx(m)" not in header
    assert header["sigx"] == 0.001


def test_normalize_canonical_only_no_warning():
    """Canonical-only headers emit no DeprecationWarning."""
    header = {"sigx": 1.0}
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _normalize_header_keys(header)
    assert header == {"sigx": 1.0}


def test_normalize_all_aliases():
    """Every deprecated alias is rewritten to its canonical key."""
    header = {old: float(i) for i, old in enumerate(HEADER_ALIASES, start=1)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        _normalize_header_keys(header)
    for i, (old, canonical) in enumerate(HEADER_ALIASES.items(), start=1):
        assert old not in header
        assert header[canonical] == float(i)


# ----------
# Impact boundary translation
# ----------


def test_impact_setitem_translates_deprecated_key():
    """``I['header:sigx(m)'] = x`` stores under ``'sigx'`` and warns."""
    impact = Impact()
    with pytest.warns(DeprecationWarning, match=r"sigx\(m\).*sigx"):
        impact["header:sigx(m)"] = 0.42
    assert impact.header["sigx"] == 0.42
    assert "sigx(m)" not in impact.header


def test_impact_getitem_translates_deprecated_key():
    """``I['header:sigx(m)']`` reads ``'sigx'`` and warns."""
    impact = Impact()
    impact.header["sigx"] = 0.42
    with pytest.warns(DeprecationWarning, match=r"sigx\(m\).*sigx"):
        assert impact["header:sigx(m)"] == 0.42


def test_impact_canonical_access_no_warning():
    """Canonical setitem/getitem produce no DeprecationWarning."""
    impact = Impact()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        impact["header:sigx"] = 0.123
        assert impact["header:sigx"] == 0.123
