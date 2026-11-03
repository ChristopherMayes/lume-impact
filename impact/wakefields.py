"""Tabulated short-range wakefields for the IMPACT codes."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping

import numpy as np
from beamphysics.wakefields import TabularWakefield


def rebin_tabular_wakefield(
    wakefield: TabularWakefield,
    dz: float | None = None,
    zmax: float | None = None,
    max_rows: int | None = None,
) -> TabularWakefield:
    """
    Rebin a bin-averaged wake table to the mesh width an IMPACT code bins at.

    A table for a wake that is singular at s = 0, such as the diffraction model,
    holds bin averages of the wake rather than samples, and is only correct at
    the bin width it was built for. IMPACT-T and IMPACT-Z bin the bunch at their
    own mesh width `dz` and read the table by linear interpolation at multiples
    of it, so the table is rebinned to `dz` preserving its integral:

    1. The antiderivative F of the wake is recovered at the table's bin edges
       (k + 1/2) * Delta, after halving the first row, which holds twice the
       self-kick bin average as in Bmad and the IMPACT codes.
    2. F is interpolated linearly at the edges (k + 1/2) * dz and differenced.
    3. The first row is doubled again.

    This is exact when `dz` is an odd multiple of the table spacing Delta, where
    the edges coincide, and otherwise accurate to well under 0.1% in the
    energy loss and spread of a bunch. For a smooth wake, a bin average equals
    the point value to second order in the bin width, so a table that samples a
    smooth wake is also handled correctly.

    Parameters
    ----------
    wakefield : TabularWakefield
        Table on a uniform grid ending at z = 0, as read from a Bmad `z_long`
        wake with `TabularWakefield.from_bmad`.
    dz : float, optional
        Target spacing (m): the longitudinal mesh width of the IMPACT code at
        the element. If omitted, the table keeps its own spacing and its rows
        pass through unchanged.
    zmax : float, optional
        Smallest trailing distance (m) the table should cover. The table covers
        at least its own range, beyond which the wake is zero.
    max_rows : int, optional
        Largest number of rows the code reads. A longer table is shortened in
        range, never coarsened, which would lose the self-kick resolution.

    Returns
    -------
    TabularWakefield
        Table with spacing `dz` and linear interpolation.

    Raises
    ------
    ValueError
        If the table is not uniform and ending at z = 0, or `dz` is not positive.

    Warns
    -----
    UserWarning
        If `max_rows` shortens the table below `zmax`.
    """
    z = wakefield.z_data
    delta = (z[-1] - z[0]) / (len(z) - 1)
    if z[-1] != 0 or not np.allclose(np.diff(z), delta, rtol=1e-6, atol=0):
        raise ValueError("The wake table must be uniform and end at z = 0")
    if dz is not None and dz <= 0:
        raise ValueError(f"dz must be positive, got {dz}")
    h = delta if dz is None else float(dz)

    table_range = -float(z[0])
    target_range = table_range if zmax is None else max(table_range, zmax)
    # TabularWakefield needs at least four rows.
    n = max(math.ceil(target_range / h * (1 - 1e-9)) + 1, 4)
    if max_rows is not None and n > max_rows:
        n = max_rows
        if zmax is not None and (n - 1) * h < zmax:
            warnings.warn(
                f"The wake table at spacing {h:.6e} m is limited to {max_rows} "
                f"rows, covering {(n - 1) * h:.6e} m, which is shorter than "
                f"wakefield_zmax = {zmax:.6e} m. The spacing is kept to resolve "
                "the self-kick.",
                UserWarning,
                stacklevel=2,
            )

    # Rows ascending in the trailing distance s = -z.
    K = wakefield.W_data[::-1].astype(float)
    if dz is None or np.isclose(h, delta, rtol=1e-9, atol=0):
        W = np.zeros(n)
        m = min(n, len(K))
        W[:m] = K[:m]
    else:
        K[0] /= 2
        edges = np.concatenate([[0.0], (np.arange(len(K)) + 0.5) * delta])
        F = np.concatenate([[0.0], np.cumsum(K) * delta])
        F_h = np.interp((np.arange(n) + 0.5) * h, edges, F)
        W = np.diff(F_h, prepend=0.0) / h
        W[0] *= 2

    return TabularWakefield(-h * np.arange(n)[::-1], W[::-1], kind="linear")


def wakefield_dz_for(
    wakefield_dz: float | Mapping[str, float] | None, name: str
) -> float | None:
    """Look up the target wake table spacing for an element by name."""
    if isinstance(wakefield_dz, Mapping):
        return wakefield_dz.get(name)
    return wakefield_dz


def check_wakefield_dz_names(
    wakefield_dz: float | Mapping[str, float] | None, names: set[str]
) -> None:
    """Raise if a per-element `wakefield_dz` names an unknown element."""
    if isinstance(wakefield_dz, Mapping):
        unknown = sorted(set(wakefield_dz) - names)
        if unknown:
            raise ValueError(f"wakefield_dz names unknown elements: {unknown}")
