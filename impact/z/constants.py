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


class IntEnum(enum.IntEnum):
    def __repr__(self):
        return f"IZ.{type(self).__name__}.{self.name}"


@_pydantic_enum
class GPUFlag(IntEnum):
    disabled = 0
    enabled = 5


@_pydantic_enum
class DistributionType(IntEnum):
    """Impact-Z distribution types."""

    uniform = 1
    gauss = 2
    waterBag = 3
    semiGauss = 4
    kV = 5
    unknown = 6
    read = 19
    multi_charge_state_waterbag = 16
    multi_charge_state_gaussian = 17


@_pydantic_enum
class DiagnosticType(IntEnum):
    none = 0
    standard = 1
    extended = 2


@_pydantic_enum
class BoundaryType(IntEnum):
    trans_open_longi_open = 1
    trans_open_longi_period = 2
    trans_round_longi_open = 3
    trans_round_longi_period = 4
    trans_rect_longi_open = 5
    trans_rect_longi_period = 6


@_pydantic_enum
class IntegratorType(IntEnum):
    linear_map = 1
    runge_kutta = 2


@_pydantic_enum
class MultipoleType(IntEnum):
    sextupole = 2
    octupole = 3
    decapole = 4


@_pydantic_enum
class RFCavityDataMode(IntEnum):
    discrete = 1
    both = 2  # analytical + discrete
    analytical = 3  # other


@_pydantic_enum
class RFCavityCoordinateType(IntEnum):
    cartesian = 2
    cylindrical = 1


@_pydantic_enum
class WigglerType(IntEnum):
    planar = 1
    helical = 2
