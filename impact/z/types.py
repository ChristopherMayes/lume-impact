from __future__ import annotations

from inspect import isclass
import pathlib
from typing import Any, Union
from collections.abc import Iterable, Sequence

import annotated_types
import numpy as np
import pydantic
import pydantic_core
from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.units import pmd_unit
from rich.pretty import pretty_repr
from typing_extensions import Literal, NotRequired, TypeAlias, TypedDict, override
from typing import Annotated

from ..repr import detailed_html_repr
from . import tools


NegativeFloat: TypeAlias = Annotated[float, annotated_types.Lt(0.0)]
PositiveFloat: TypeAlias = Annotated[float, annotated_types.Gt(0.0)]
NonzeroFloat: TypeAlias = Union[NegativeFloat, PositiveFloat]


class ReprTableData(TypedDict):
    """Data to use for table output."""

    obj: BaseModel | dict[str, Any]
    descriptions: dict[str, str] | None
    annotations: dict[str, str] | None


def _check_equality(obj1: Any, obj2: Any) -> bool:
    """
    Check equality of `obj1` and `obj2`.`

    Parameters
    ----------
    obj1 : Any
    obj2 : Any

    Returns
    -------
    bool
    """
    if not isinstance(obj1, type(obj2)):
        return False

    if isinstance(obj1, pydantic.BaseModel):
        return all(
            _check_equality(
                getattr(obj1, attr),
                getattr(obj2, attr),
            )
            for attr, fld in obj1.model_fields.items()
            if not fld.exclude
        )

    if isinstance(obj1, dict):
        if set(obj1) != set(obj2):
            return False

        return all(
            _check_equality(
                obj1[key],
                obj2[key],
            )
            for key in obj1
        )

    if isinstance(obj1, (list, tuple)):
        if len(obj1) != len(obj2):
            return False
        return all(
            _check_equality(obj1_value, obj2_value)
            for obj1_value, obj2_value in zip(obj1, obj2)
        )

    if isinstance(obj1, np.ndarray):
        if not obj1.shape and not obj2.shape:
            return True
        return np.allclose(obj1, obj2, equal_nan=True)

    if isinstance(obj1, float):
        return np.allclose(obj1, obj2, equal_nan=True)

    return bool(obj1 == obj2)


class BaseModel(pydantic.BaseModel, extra="forbid", validate_assignment=True):
    """
    LUME-Impact customized pydantic BaseModel.

    * `dir()` handling for user convenience.
    * JupyterLab and console repr improvements.
    * Customized equality checks for fields with numpy arrays.
    * `to_string` and `to_table` helpers.
    """

    def _repr_table_data_(self) -> ReprTableData:
        units = getattr(self, "units", None)
        return {
            "obj": self,
            "descriptions": None,
            "annotations": units if isinstance(units, dict) else None,
        }

    @classmethod
    def _to_docstring(cls):
        print('    """')
        print("    Attributes")
        print("    ----------")
        for attr, fld in cls.model_fields.items():
            annotation = fld.annotation
            if isclass(annotation):
                annotation = annotation.__name__

            annotation = str(annotation)
            annotation = annotation.replace("typing.", "")

            print(f"    {attr} : {annotation}")
            if fld.description:
                print(f"        {fld.description}")
        print('    """')

    def to_table(self):
        return tools.table_output(**self._repr_table_data_())

    def to_string(
        self, mode: Literal["html", "markdown", "native", "genesis", "repr"]
    ) -> str:
        if mode == "html":
            return tools.html_table_repr(**self._repr_table_data_(), seen=[])
        if mode == "markdown":
            return str(tools.ascii_table_repr(**self._repr_table_data_(), seen=[]))
        if mode == "native" or mode == "genesis":  # TODO compat
            to_contents = getattr(self, "to_contents", None)
            if callable(to_contents):
                return to_contents()
            return repr(self)
        if mode == "repr":
            return repr(self)

        raise NotImplementedError(f"Render mode {mode} unsupported")

    def __repr__(self):
        return pretty_repr(self)

    def _repr_html_(self):
        return detailed_html_repr(self)

    @override
    def __eq__(self, other: Any) -> bool:
        return _check_equality(self, other)

    @override
    def __ne__(self, other: Any) -> bool:
        return not _check_equality(self, other)

    @override
    def __str__(self) -> str:
        return self.to_string(tools.global_display_options.console_render_mode)

    @override
    def __dir__(self) -> Iterable[str]:
        full = super().__dir__()
        if not tools.global_display_options.filter_tab_completion:
            return full

        base_model = set(dir(pydantic.BaseModel))
        return [
            attr for attr in full if not attr.startswith("_") and attr not in base_model
        ]


class SequenceBaseModel(pydantic.BaseModel, extra="forbid", validate_assignment=True):
    """
    LUME-Impact customized pydantic BaseModel that represents a fixed sequence of data.

    This means that positional instantiation (via `.from_sequence`) can be a
    useful and natural way of using the model.
    """

    @classmethod
    def from_sequence(cls, args: Sequence[Any]):
        kwargs: dict[str, Any] = dict(zip(cls.model_fields, args))
        return cls(**kwargs)


class _PydanticNDArray:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: type[Any],
        handler: pydantic.GetCoreSchemaHandler,
    ) -> pydantic_core.core_schema.CoreSchema:
        def serialize(obj: np.ndarray, info: pydantic.SerializationInfo):
            if not isinstance(obj, np.ndarray):
                raise ValueError(
                    f"Only supports numpy ndarray. Got {type(obj).__name__}: {obj}"
                )

            return obj.tolist()

        return pydantic_core.core_schema.with_info_plain_validator_function(
            cls._pydantic_validate,
            serialization=pydantic_core.core_schema.plain_serializer_function_ser_schema(
                serialize, when_used="json-unless-none", info_arg=True
            ),
        )

    @classmethod
    def _pydantic_validate(
        cls,
        value: Any | np.ndarray | Sequence | dict,
        info: pydantic.ValidationInfo,
    ) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, Sequence):
            return np.asarray(value)
        if np.isscalar(value):
            return np.asarray([value])
        raise ValueError(f"No conversion from {value!r} to numpy ndarray")


class ParticleDictData(TypedDict):
    """
    ParticleGroup raw data as a dictionary.

    The following keys are required:
    * `x`, `y`, `z` are np.ndarray in units of [m]
    * `px`, `py`, `pz` are np.ndarray momenta in units of [eV/c]
    * `t` is a np.ndarray of time in [s]
    * `status` is a status coordinate np.ndarray
    * `weight` is the macro-charge weight in [C], used for all statistical calculations.
    * `species` is a proper species name: `'electron'`, etc.
    The following keys are optional:
    * `id` is an optional np.ndarray of unique IDs
    """

    # `x`, `y`, `z` are positions in units of [m]
    x: NDArray
    y: NDArray
    z: NDArray

    # `px`, `py`, `pz` are momenta in units of [eV/c]
    px: NDArray
    py: NDArray
    pz: NDArray

    # `t` is time in [s]
    t: NDArray
    status: NDArray

    # `weight` is the macro-charge weight in [C], used for all statistical
    # calculations.
    weight: NDArray

    # `species` is a proper species name: `'electron'`, etc.
    species: str
    id: NotRequired[NDArray]


class ParticleData(BaseModel):
    """
    ParticleGroup raw data as a Pydantic model.

    Used internally for serialization of ParticleGroup to JSON.
    """

    # `x`, `y`, `z` are positions in units of [m]
    x: NDArray = np.zeros(0)
    y: NDArray = np.zeros(0)
    z: NDArray = np.zeros(0)

    # `px`, `py`, `pz` are momenta in units of [eV/c]
    px: NDArray = np.zeros(0)
    py: NDArray = np.zeros(0)
    pz: NDArray = np.zeros(0)

    # `t` is time in [s]
    t: NDArray = np.zeros(0)
    status: NDArray = np.zeros(0)

    # `weight` is the macro-charge weight in [C], used for all statistical
    # calculations.
    weight: NDArray = np.zeros(0)

    # `species` is a proper species name: `'electron'`, etc.
    species: str = "electron"
    id: NDArray = np.zeros(0)


class _PydanticParticleGroup:
    data: ParticleDictData

    @staticmethod
    def _from_dict(data: ParticleDictData) -> ParticleGroup:
        dat = dict(data)
        id_ = dat.get("id", None)
        if id_ is not None and len(id_) == 0:
            dat.pop("id")
        return ParticleGroup(data=dat)

    def _to_model(self):
        return ParticleData(**self.data)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: type[Any],
        handler: pydantic.GetCoreSchemaHandler,
    ) -> pydantic_core.core_schema.CoreSchema:
        return pydantic_core.core_schema.no_info_plain_validator_function(
            cls._pydantic_validate,
            serialization=pydantic_core.core_schema.plain_serializer_function_ser_schema(
                cls._to_model, when_used="json-unless-none"
            ),
        )

    @classmethod
    def _pydantic_validate(
        cls, value: ParticleData | ParticleDictData | ParticleGroup
    ) -> ParticleGroup:
        if isinstance(value, ParticleGroup):
            return value
        if isinstance(value, ParticleData):
            return value
        if isinstance(value, dict):
            return cls._from_dict(value)
        raise ValueError(f"No conversion from {value!r} to ParticleGroup")  # type: ignore[unreachable]


class _PydanticPmdUnit:
    unitSI: float
    unitSymbol: str
    unitDimension: tuple[int, ...]

    @staticmethod
    def _from_dict(dct: dict) -> pmd_unit:
        dct = dict(dct)
        dim = dct.pop("unitDimension", None)
        if dim is not None:
            dim = tuple(dim)
        return pmd_unit(**dct, unitDimension=dim)

    def _as_dict(self) -> dict[str, Any]:
        return {
            "unitSI": self.unitSI,
            "unitSymbol": self.unitSymbol,
            "unitDimension": tuple(self.unitDimension),
        }

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: type[Any],
        handler: pydantic.GetCoreSchemaHandler,
    ) -> pydantic_core.core_schema.CoreSchema:
        return pydantic_core.core_schema.no_info_plain_validator_function(
            cls._pydantic_validate,
            serialization=pydantic_core.core_schema.plain_serializer_function_ser_schema(
                cls._as_dict, when_used="json-unless-none"
            ),
        )

    @classmethod
    def _pydantic_validate(cls, value: dict[str, Any] | pmd_unit | Any) -> pmd_unit:
        if isinstance(value, pmd_unit):
            return value
        if isinstance(value, dict):
            return cls._from_dict(value)
        raise ValueError(f"No conversion from {value!r} to pmd_unit")


def _get_output_discriminator_value(value):
    # Note: this is a bit of a hack to instruct pydantic which type should
    # be used in the union. As someone new to custom types in Pydantic v2,
    # I'm sure there's a better way to do this - and am open to suggestions!
    if isinstance(value, np.ndarray):
        return "array"
    if isinstance(value, np.generic):
        value = value.item()
    return type(value).__name__


PydanticParticleGroup = Annotated[ParticleGroup, _PydanticParticleGroup]
PydanticPmdUnit = Annotated[pmd_unit, _PydanticPmdUnit]
NDArray = Annotated[np.ndarray, _PydanticNDArray]
AnyPath = Union[pathlib.Path, str]
OutputDataType = Annotated[
    Union[
        Annotated[float, pydantic.Tag("float")],
        Annotated[int, pydantic.Tag("int")],
        Annotated[str, pydantic.Tag("str")],
        Annotated[bool, pydantic.Tag("bool")],
        # Annotated[NDArray, pydantic.Tag("array")],
    ],
    pydantic.Discriminator(_get_output_discriminator_value),
]
