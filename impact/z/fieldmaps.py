import numpy as np
from scipy.constants import c


def make_solenoid_rfcavity_rfdata_simple(
    *,
    rf_frequency: float | None = None,
    rf_wavelength: float | None = None,
    n_cells: int = 1,
) -> np.ndarray:
    """
    Create minimal IMPACT-Z rfdata for the `SolenoidWithRFCavity` element for a
    purely sinusoidal longitudinal electric field.

    If rf_frequency is given, the RF wavelength is computed. Exactly one of
    rf_frequency or rf_wavelength must be specified.

    Parameters
    ----------
    rf_frequency : float or None, optional
        The RF frequency in Hertz. If specified, rf_wavelength must not be specified.
    rf_wavelength : float or None, optional
        The RF wavelength in meters. If specified, rf_frequency must not be specified.
    n_cells : int, optional
        Number of cells. Default is 1.

    Returns
    -------
    np.ndarray
        Numpy array representing the minimal Impact rfdata for the solrf element.

    Raises
    ------
    ValueError
        If both rf_frequency and rf_wavelength are specified, or if neither is specified.
    """
    if rf_frequency is not None:
        if rf_wavelength is not None:
            raise ValueError(
                "rf_wavelength may not be specified if rf_frequency is None"
            )
        rf_wavelength = c / rf_frequency

    if rf_wavelength is None:
        raise ValueError("Either rf_frequency or rf_wavelength must be specified")

    L = rf_wavelength
    z0 = 0
    z1 = n_cells * L
    data = [
        3,  # of Fourier coef. of Ez on axis
        z0,  # distance before the zedge.
        z1,  # distance after the zedge.
        L,  # length of the Fourier expanded field.
        0,
        0,  # cos
        1,  # sin
        1.0,  # of Fourier coef. of Bz on axis.
        0.0,
        0.0,
        0.0,
        0.0,
    ]  # Fourier coefficients of the Bz on axis.

    return np.array(data, dtype=float)
