import enum

PARTICLE_TYPE = {
    "Electron": "510998.9461 -1.0",
    "Proton": "938272081.3 1.0",
    "Positron": "510998.9461 1.0",
    "Antiproton": "938272081.3 -1.0",
    "Other": "Other_NONE",
}


class GPUFlag(enum.IntEnum):
    disabled = 0
    enabled = 5


class DistributionTType(enum.IntEnum):
    """
    Impact-T distribution types.

    1 Uniform - 6d uniform distribution
    2 Gauss3 - 6d Gaussian distribution
    3 Waterbag - 6d Waterbag distribution
    4 Semigauss - 3d Waterbag distribution in spatial and 3d Gaussian
    distribution in momentum space
    5 KV3d - transverse KV distribution and longitudinal uniform distribution
    10 ParobGauss - transverse parabolic and longitudinal Gaussian distribution
    15 SemicirGauss - transverse semi-circle and longitudinal Gaussian distribution
    16 Read - read in an initial particle distribution from file Partcl.data
    24 readParmela - read in Parmela particle format
    25 readElegant - read in Elegant particle format
    27 CylcoldZSob
    """

    uniform = 1
    gauss = 2
    waterbag = 3
    semigauss = 4
    KV = 5
    parabolic_gaussian = 10
    semicircular_gaussian = 15
    read = 16
    read_parmela = 24
    read_elegant = 25
    cylcold_zsob = 27


class DistributionZType(enum.IntEnum):
    """Impact-Z distribution types."""

    uniform = 1
    gauss = 2
    waterBag = 3
    semiGauss = 4
    kV = 5
    read = 19
    multi_charge_state_waterbag = 16
    multi_charge_state_gaussian = 17


class DiagnosticType(enum.IntEnum):
    at_given_time = 1
    at_bunch_centroid = 2
    no_output = 3


class OutputZType(enum.IntEnum):
    none = 0
    standard = 1
    extended = 2


class BoundaryType(enum.IntEnum):
    trans_open_longi_open = 1
    trans_open_longi_period = 2
    trans_round_longi_open = 3
    trans_round_longi_period = 4
    trans_rect_longi_open = 5
    trans_rect_longi_period = 6


class IntegratorType(enum.IntEnum):
    linear = 1
    non_linear = 2


class MultipoleType(enum.IntEnum):
    sextupole = 2
    octupole = 3
    decapole = 4


class RFCavityDataMode(enum.IntEnum):
    discrete = 1
    both = 2  # analytical + discrete
    analytical = 3  # other


class RFCavityCoordinateType(enum.IntEnum):
    cartesian = 2
    cylindrical = 1


class ElementID(enum.IntEnum):
    drift = 0
    quad = 1
    bend = 4
    scrf = 104
    write_full = -2
    restart = -7
    halt = -99


# PLOTTYPE = {
#     "Centriod location": 2,
#     "Rms size": 3,
#     "Centriod momentum": 4,
#     "Rms momentum": 5,
#     "Twiss": 6,
#     "Emittance": 7,
# }
