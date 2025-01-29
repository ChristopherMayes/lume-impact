from __future__ import annotations
import enum
from typing import Any, TypeVar
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

PARTICLE_TYPE = {
    "Electron": "510998.9461 -1.0",
    "Proton": "938272081.3 1.0",
    "Positron": "510998.9461 1.0",
    "Antiproton": "938272081.3 -1.0",
    "Other": "Other_NONE",
}


E = TypeVar("E", bound=enum.Enum)


def _pydantic_enum(enum_cls: type[E]) -> type[E]:
    """
    Wrapper to make an Enum class validatable by either its name or value.

    ref: https://github.com/pydantic/pydantic/discussions/2980#discussioncomment-9912210

    Parameters
    ----------
    enum_cls : subclass of enum.Enum

    Returns
    -------
    enum_cls : subclass of enum.Enum
    """

    def __get_pydantic_core_schema__(
        cls: type[E],
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ):
        assert source_type is cls

        def get_enum(
            value: Any,
            validate_next: core_schema.ValidatorFunctionWrapHandler,
        ):
            if isinstance(value, cls):
                return value

            name_or_value: str = validate_next(value)
            if isinstance(name_or_value, int):
                return cls(name_or_value)
            return enum_cls[name_or_value]

        def serialize(enum: E):
            return enum.name

        expected = [member.name for member in cls]
        expected.extend(
            list(set(member.value for member in cls if member.value not in expected))
        )
        name_schema = core_schema.literal_schema(expected)

        return core_schema.no_info_wrap_validator_function(
            get_enum,
            name_schema,
            ref=cls.__name__,
            serialization=core_schema.plain_serializer_function_ser_schema(serialize),
        )

    setattr(
        enum_cls,
        "__get_pydantic_core_schema__",
        classmethod(__get_pydantic_core_schema__),
    )
    return enum_cls


@_pydantic_enum
class GPUFlag(enum.IntEnum):
    disabled = 0
    enabled = 5


@_pydantic_enum
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
    16 Read - read in an initial particle distribution from file `particle.in`
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


@_pydantic_enum
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


@_pydantic_enum
class DiagnosticType(enum.IntEnum):
    at_given_time = 1
    at_bunch_centroid = 2
    no_output = 3


@_pydantic_enum
class OutputZType(enum.IntEnum):
    none = 0
    standard = 1
    extended = 2


@_pydantic_enum
class BoundaryType(enum.IntEnum):
    trans_open_longi_open = 1
    trans_open_longi_period = 2
    trans_round_longi_open = 3
    trans_round_longi_period = 4
    trans_rect_longi_open = 5
    trans_rect_longi_period = 6


@_pydantic_enum
class IntegratorType(enum.IntEnum):
    linear_map = 1
    runge_kutta = 2


@_pydantic_enum
class MultipoleType(enum.IntEnum):
    sextupole = 2
    octupole = 3
    decapole = 4


@_pydantic_enum
class RFCavityDataMode(enum.IntEnum):
    discrete = 1
    both = 2  # analytical + discrete
    analytical = 3  # other


@_pydantic_enum
class RFCavityCoordinateType(enum.IntEnum):
    cartesian = 2
    cylindrical = 1


@_pydantic_enum
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
