from __future__ import annotations

from abc import abstractmethod
import enum
import logging
import pathlib
import shlex
import types
import typing
from typing import (
    Any,
    ClassVar,
    Iterable,
    Literal,
    NamedTuple,
    TypeVar,
    Union,
    cast,
)
from collections.abc import Sequence

import h5py
import matplotlib.axes
import numpy as np
import pydantic
import pydantic.alias_generators
from lume import tools as lume_tools
from scipy.constants import e
from typing_extensions import Protocol, runtime_checkable

from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.particles import c_light

from ..impact import suggested_processor_domain
from .. import tools
from . import archive as _archive, parsers
from .constants import (
    BoundaryType,
    DistributionType,
    GPUFlag,
    IntegratorType,
    MultipoleType,
    DiagnosticType,
    RFCavityCoordinateType,
    RFCavityDataMode,
    WigglerType,
)
from .errors import MultipleElementError, NoSuchElementError
from .particles import ImpactZParticles, detect_species
from .types import AnyPath, BaseModel, NonzeroFloat, NDArray, PydanticParticleGroup

if typing.TYPE_CHECKING:
    from pytao import Tao
    from .interfaces.bmad import Which as TaoWhich


input_element_by_id: dict[int, type[InputElement]] = {}
logger = logging.getLogger(__name__)


class _InputElementClassMetadata(BaseModel):
    element_id: int
    has_input_file: bool = False
    has_output_file: bool = False


@runtime_checkable
class HasInputFile(Protocol):
    file_id: float

    @property
    @abstractmethod
    def input_filename(self) -> str | None: ...


@runtime_checkable
class HasOutputFile(Protocol):
    file_id: float


InputElementMetadata = dict[str, Union[int, float, str, bool, NDArray]]


class InputElement(BaseModel):
    _impactz_metadata_: ClassVar[_InputElementClassMetadata]
    _impactz_fields_: ClassVar[tuple[str, ...]]
    # Minimum file ID to be recognized as an input file
    _impactz_min_file_id_: ClassVar[int] = 0
    name: str = ""
    metadata: dict[str, int | float | str | bool | NDArray] = {}

    def __init_subclass__(
        cls,
        element_id: int,
        has_input_file: bool = False,
        has_output_file: bool = False,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)

        if not isinstance(element_id, int):
            raise ValueError(f"element_id expected to be int, got: {type(element_id)}")
        if element_id in input_element_by_id:
            raise RuntimeError(f"Duplicate element {element_id}")

        input_element_by_id[element_id] = cls
        cls._impactz_metadata_ = _InputElementClassMetadata(
            element_id=element_id,
            has_input_file=has_input_file,
            has_output_file=has_output_file,
        )
        cls._impactz_fields_ = tuple(cls.__annotations__)

    @classmethod
    def class_information(cls):
        return cls._impactz_metadata_

    @staticmethod
    def from_line(
        line: str | parsers.InputLine,
        *,
        name: str | None = None,
    ):
        if isinstance(line, str):
            line = parsers.parse_input_line(line)

        type_idx = int(line.data[3])
        ele_cls = input_element_by_id[type_idx]

        # if ele_cls is Drift and len(parts) == 7:
        #     # TODO: a known bit of 'extra' data in the examples
        #     # patching in a hotfix here, but we may adjust later...
        #     parts = parts[:5]
        if len(line.data) > len(ele_cls._impactz_fields_):
            raise ValueError(
                f"Too many input elements for {ele_cls.__name__}: "
                f"expected {len(ele_cls._impactz_fields_)} at most, got {len(line.data)}"
            )

        kwargs = dict(zip(ele_cls._impactz_fields_, line.data))
        return ele_cls(
            **kwargs,
            name=name or line.inline_comment or "",
        )

    def to_line(
        self, *, with_description: bool = True, z_start: float | None = None
    ) -> str:
        def as_string(v: float | int):
            if isinstance(v, (bool, float)):
                return f"{v:.20g}"
            if isinstance(v, enum.IntEnum):
                return str(v.value)
            return str(v)

        attr_to_value = {
            attr: as_string(getattr(self, attr)) for attr in self._impactz_fields_
        }

        line = " ".join(attr_to_value.values())
        if not with_description:
            return f"{line} /"

        name = f" {self.name}" if self.name else ""

        z_desc = f"z={z_start:.3f} " if z_start is not None else ""
        desc = f"! [{z_desc}{type(self).__name__}] " + " ".join(attr_to_value)
        return f"{desc}\n{line} /{name}"

    @property
    def input_filename(self) -> str | None:
        file_id = getattr(self, "file_id", None)
        if file_id is not None and file_id >= 0:
            return f"rfdata{int(file_id)}.in"

        return None


class Drift(InputElement, element_id=0):
    """
    Drift element.

    Attributes
    ----------
    length : float
        Length of the element in meters.
    steps : int
        Number of space-charge kicks through the beamline element. Each
        "step" consists of a half-step, a space-charge kick, and another half-step.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    radius : float
        Radius of the pipe, in meters.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[0] = 0
    radius: float = 1.0
    unused_0: float = pydantic.Field(
        default=0.0, repr=False
    )  # unused/undocumented; should we just ignore?
    unused_1: float = pydantic.Field(
        default=0.0, repr=False
    )  # unused/undocumented; should we just ignore?


class Quadrupole(InputElement, element_id=1, has_input_file=True):
    """
    A quadrupole element.

    Attributes
    ----------
    length : float
        Length of the element in meters.
    steps : int
        Number of space-charge kicks through the beamline element. Each
        "step" consists of a half-step, a space-charge kick, and another half-step.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    k1 : float
        The quadrupole strength, 1/m^2. (NOTE: the manual is actually wrong here, this
        is not B1 in units of T/m)
    file_id : float
        An ID for the input gradient file. Determines profile behavior:
        if greater than 0, a fringe field profile is read; if less than -10,
        a linear transfer map of an undulator is used; if between -10 and 0,
        it's the k-value linear transfer map; if equal to 0, it uses the linear
        transfer map with the gradient.
    radius : float
        The radius of the quadrupole, measured in meters.
    misalignment_error_x : float, optional
        The x-direction misalignment error, given in meters.
    misalignment_error_y : float, optional
        The y-direction misalignment error, given in meters.
    rotation_error_x : float, optional
        Rotation error in radians.
    rotation_error_y : float, optional
        Rotation error in radians.
    rotation_error_z : float, optional
        Rotation error in radians.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[1] = 1
    k1: float = 0.0
    file_id: float = 0.0
    radius: float = 0.0
    misalignment_error_x: float = 0.0
    misalignment_error_y: float = 0.0

    rotation_error_x: float = 0.0
    rotation_error_y: float = 0.0
    rotation_error_z: float = 0.0

    @property
    def input_filename(self) -> str | None:
        # An ID for the input gradient file. Determines profile behavior:
        # if greater than 0, a fringe field profile is read; if less than -10,
        # a linear transfer map of an undulator is used; if between -10 and 0,
        # it's the k-value linear transfer map; if equal to 0, it uses the linear
        # transfer map with the gradient.
        # TODO what does this... mean?
        if self.file_id <= 0:
            return None
        return f"rfdata{int(self.file_id)}.in"


class ConstantFocusing(InputElement, element_id=2):
    """
    3D constant focusing.

    Attributes
    ----------
    length : float
        Length of the element in meters.
    steps : int
        Number of space-charge kicks through the beamline element. Each
        "step" consists of a half-step, a space-charge kick, and another half-step.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    kx0_squared : float
        The square of the kx0 parameter.
    ky0_squared : float
        The square of the ky0 parameter.
    kz0_squared : float
        The square of the kz0 parameter.
    radius : float
        The radius of the focusing element in meters.

    Notes
    -----
    This class does not work for the Lorentz integrator option.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[2] = 2
    kx0_squared: float = 0.0
    ky0_squared: float = 0.0
    kz0_squared: float = 0.0
    radius: float = 0.0


class Solenoid(InputElement, element_id=3, has_input_file=True):
    """
    Solenoid used in beam dynamics simulations.

    Attributes
    ----------
    length : float
        The effective length of the solenoid in meters, including two linear
        fringe regions and a flat top region (solenoid integrator).
    steps : int
        Number of space-charge kicks through the beamline element. Each
        "step" consists of a half-step, a space-charge kick, and another half-step.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    Bz0 : float
        The axial magnetic field at the center of the solenoid in Tesla.
    file_id : float
        The identifier for the input field file.
    radius : float
        The radius of the solenoid in meters.
    misalignment_error_x : float
        Misalignment error in the x-direction in meters.
    misalignment_error_y : float
        Misalignment error in the y-direction in meters.
    x_rotation_error : float
        Rotation error in the x-direction in radians.
    rotation_error_y : float
        Rotation error in the y-direction in radians.
    rotation_error_z : float
        Rotation error in the z-direction in radians.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[3] = 3

    Bz0: float = 0.0
    file_id: float = 0.0
    radius: float = 0.0
    misalignment_error_x: float = 0.0
    misalignment_error_y: float = 0.0
    rotation_error_x: float = 0.0
    rotation_error_y: float = 0.0
    rotation_error_z: float = 0.0


class Dipole(InputElement, element_id=4, has_input_file=True):
    """
    Represents a dipole element used in beam simulations.

    Attributes
    ----------
    length : float
        Length of the element in meters.
    steps : int
        Number of space-charge kicks through the beamline element. Each
        "step" consists of a half-step, a space-charge kick, and another half-step.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    angle : float, optional
        Bending angle [rad]. Must be non-zero.
    k1 : float, optional
        Field strength.
    input_switch : float, optional
        CSR settings.
        - `input_switch <= 200`: no CSR
        - `200 < input_switch <=500`: CSR in the bend
        - `input_switch > 500`: CSR in the bend and the next following drift
        Ref: https://github.com/impact-lbl/IMPACT-Z/blob/96ae896517bcb83fa741cd203892cb42a88f0e4f/src/Contrl/AccSimulator.f90#L999-L1004
    hgap : float, optional
        Half gap [m].
    e1 : float, optional
        Entrance pole face angle [rad].
    e2 : float, optional
        Exit pole face angle [rad].
    entrance_curvature : float, optional
        Curvature of entrance face [rad].
    exit_curvature : float, optional
        Curvature of exit face [rad].
    fint : float, optional
        Integrated fringe field K of entrance (Kf). Fringe field K of exit
        assumed to be equal (Kb = Kf).
    misalignment_error_x : float, optional
        Misalignment error in the x direction.
    misalignment_error_y : float, optional
        Misalignment error in the y direction.
    rotation_error_x : float, optional
        Rotation error around the x axis.
    rotation_error_y : float, optional
        Rotation error around the y axis.
    rotation_error_z : float, optional
        Rotation error around the z axis.

    Notes
    -----

    ```fortran
    hd0 = angle/blength !k0
    tanphiF = tan(e1)
    psi1 = hd0*2*hgap*fint*(1.0+sin(e1)*sin(e1))/cos(e1)
    tanphiFb = tan(e1-psi1)
    tanphiB = tan(e2)
    psi2 = hd0*2*hgap*fint*(1.0+sin(e2)*sin(e2))/cos(e2)
    tanphiBb = tan(e2-psi2)
    qm0 = Bpts%Charge/Bpts%Mass
    r0  = abs(1.0d0/hd0)
    ```
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[4] = 4

    angle: NonzeroFloat = pydantic.Field(default=1e-6)  # dparam(2)
    k1: float = 0.0  # dparam(3)
    input_switch: float = 0.0  # dparam(4)
    hgap: float = 0.0  # dparam(5)
    e1: float = 0.0  # dparam(6)
    e2: float = 0.0  # dparam(7)
    entrance_curvature: float = 0.0  # dparam(8)
    exit_curvature: float = 0.0  # dparam(9)
    fint: float = 0.0  # dparam(10)
    misalignment_error_x: float = 0.0
    misalignment_error_y: float = 0.0
    rotation_error_x: float = 0.0
    rotation_error_y: float = 0.0
    rotation_error_z: float = 0.0

    def set_csr(self, enabled: bool, following_drift: bool) -> None:
        """
        Set the input switch CSR setting.

        Parameters
        ----------
        enabled : bool
            Enable CSR for the dipole bend.
        following_drift : bool
            In addition to CSR for the dipole, also use CSR on the following
            drift element.  Only applies when `enabled`.
        """
        if not enabled:
            value = 0.0
        elif following_drift:
            value = 501.0
        else:
            value = 201.0
        self.input_switch = value

    @property
    def csr_enabled(self) -> bool:
        return self.input_switch > 200.0


class Multipole(InputElement, element_id=5, has_input_file=True):
    """
    Represents a multipole element used in beam simulations.

    Attributes
    ----------
    length : float
        Length of the element in meters.
    steps : int
        Number of space-charge kicks through the beamline element. Each
        "step" consists of a half-step, a space-charge kick, and another half-step.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    multipole_type : MultipoleType
        The type of multipole element, sextupole, octupole, or decapole.
    field_strength : float, optional
        The strength of the magnetic field.  Units of T/m^n.
    file_id : float, optional
        Identifier for related input data file.
    radius : float, optional
        The radius of the multipole.
    misalignment_error_x : float, optional
        Misalignment error in the x-direction.
    misalignment_error_y : float, optional
        Misalignment error in the y-direction.
    rotation_error_x : float, optional
        Rotation error around the x-axis.
    rotation_error_y : float, optional
        Rotation error around the y-axis.
    rotation_error_z : float, optional
        Rotation error around the z-axis.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[5] = 5

    multipole_type: MultipoleType = MultipoleType.sextupole
    field_strength: float = 0.0
    file_id: float = 0.0
    radius: float = 0.0
    misalignment_error_x: float = 0.0
    misalignment_error_y: float = 0.0
    rotation_error_x: float = 0.0
    rotation_error_y: float = 0.0
    rotation_error_z: float = 0.0


class Wiggler(InputElement, element_id=6):
    """
    Represents a planar or helical wiggler element used in beam simulations.

    Only supports the integrator type `IntegratorType.runge_kutta`.

    Attributes
    ----------
    length : float, optional
        Length of the element in meters.
    steps : int, optional
        Number of space-charge kicks through the beamline element. Each
        "step" consists of a half-step, a space-charge kick, and another half-step.
    map_steps : int, optional
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    wiggler_type : WigglerType, optional
        Wiggler type. Defaults to `WigglerType.planar`.
    max_field_strength : float, optional
        The maximum strength of the magnetic field.  Units of T/m^n.
    file_id : float, optional
        File ID (unused?)
    radius : float, optional
        Radius in meters.
    kx : float, optional
        Wiggler strength.
    period : float
        Period of the wiggler.
    misalignment_error_x : float, optional
        Misalignment error in the x-direction.
    misalignment_error_y : float, optional
        Misalignment error in the y-direction.
    rotation_error_x : float, optional
        Rotation error around the x-axis.
    rotation_error_y : float, optional
        Rotation error around the y-axis.
    rotation_error_z : float, optional
        Rotation error around the z-axis.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[6] = 6

    wiggler_type: WigglerType = WigglerType.planar
    max_field_strength: float = 0.0
    file_id: float = 0.0
    radius: float = 0.0
    kx: float = 0.0
    period: float = 0.0
    misalignment_error_x: float = 0.0
    misalignment_error_y: float = 0.0
    rotation_error_x: float = 0.0
    rotation_error_y: float = 0.0
    rotation_error_z: float = 0.0


class DTL(InputElement, element_id=101, has_input_file=True):
    """
    Discrete-Transmission-Line element with specified parameters.

    Attributes
    ----------
    length : float
        Length of the element in meters.
    steps : int
        Number of space-charge kicks through the beamline element. Each
        "step" consists of a half-step, a space-charge kick, and another half-step.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    field_scaling : float
        Scaling factor for the electrical/magnetic field.
    rf_frequency : float
        RF frequency in Hertz.
    phase_deg : float
        Driven phase in degrees.
    file_id : float
        Input field ID (using a simple sinusoidal model if ID<0).
    radius : float
        Radius in meters.
    quad1_length : float
        Length of the first quadrupole in meters.
    quad1_gradient : float
        Gradient of the first quadrupole in Tesla/meter.
    quad2_length : float
        Length of the second quadrupole in meters.
    quad2_gradient : float
        Gradient of the second quadrupole in Tesla/meter.
    misalignment_error_x : float
        Misalignment error in the x-direction in meters.
    misalignment_error_y : float
        Misalignment error in the y-direction in meters.
    rotation_error_x : float
        Rotation error around the x-axis in radians.
    rotation_error_y : float
        Rotation error around the y-axis in radians.
    rotation_error_z : float
        Rotation error around the z-axis in radians.
    displacement_x : float
        Displacement in the x-direction in meters.
    displacement_y : float
        Displacement in the y-direction in meters.
    rotation_error_rf_x : float
        Rotation error around the x-axis for the RF field in radians.
    rotation_error_rf_y : float
        Rotation error around the y-axis for the RF field in radians.
    rotation_error_rf_z : float
        Rotation error around the z-axis for the RF field in radians.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[101] = 101

    field_scaling: float = 0.0
    rf_frequency: float = 0.0
    phase_deg: float = 0.0
    file_id: float = 0.0
    radius: float = 0.0
    quad1_length: float = 0.0
    quad1_gradient: float = 0.0
    quad2_length: float = 0.0
    quad2_gradient: float = 0.0

    q1_misalignment_error_x: float = 0.0
    q1_misalignment_error_y: float = 0.0
    q1_rotation_error_x: float = 0.0
    q1_rotation_error_y: float = 0.0
    q1_rotation_error_z: float = 0.0

    # TODO: docs are wrong per the code past this point
    q2_misalignment_error_x: float = 0.0
    q2_misalignment_error_y: float = 0.0
    q2_rotation_error_x: float = 0.0
    q2_rotation_error_y: float = 0.0
    q2_rotation_error_z: float = 0.0

    rf_misalignment_error_x: float = 0.0
    rf_misalignment_error_y: float = 0.0
    rf_rotation_error_x: float = 0.0
    rf_rotation_error_y: float = 0.0
    rf_rotation_error_z: float = 0.0


class CCDTL(InputElement, element_id=102, has_input_file=True):
    """
    A CCDTL (Cell-Coupled Drift Tube Linac) input element represented by its parameters.

    Attributes
    ----------
    length : float
        Length of the element in meters.
    steps : int
        Number of space-charge kicks through the beamline element. Each
        "step" consists of a half-step, a space-charge kick, and another half-step.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    field_scaling : float
        Field scaling factor.
    rf_frequency : float
        RF frequency in Hertz.
    phase_deg : float
        Driven phase in degrees.
    file_id : float
        Input field ID (if ID<0, use simple sinusoidal model, only works for the map integrator).
        The phase is the design phase with 0 for maximum energy gain.
    radius : float
        Radius in meters.
    misalignment_x : float
        X misalignment error in meters.
    misalignment_error_y : float
        Y misalignment error in meters.
    rotation_error_x : float
        Rotation error around the x-axis in radians.
    rotation_error_y : float
        Rotation error around the y-axis in radians.
    rotation_error_z : float
        Rotation error around the z-axis in radians.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[102] = 102

    field_scaling: float = 0.0
    rf_frequency: float = 0.0
    phase_deg: float = 0.0  # theta0
    file_id: float = 0.0
    radius: float = 0.0
    misalignment_error_x: float = 0.0
    misalignment_error_y: float = 0.0
    rotation_error_x: float = 0.0
    rotation_error_y: float = 0.0
    rotation_error_z: float = 0.0


class CCL(InputElement, element_id=103, has_input_file=True):
    """
    CCL input element with specific parameters.

    Attributes
    ----------
    length : float
        Length of the element in meters.
    steps : int
        Number of space-charge kicks through the beamline element. Each
        "step" consists of a half-step, a space-charge kick, and another half-step.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    field_scaling : float
        Field scaling factor.
    rf_frequency : float
        RF frequency in Hertz.
    phase_deg : float
        Driven phase in degrees.
    file_id : float
        Input field ID. If ID < 0, use the simple sinusoidal model
        (only works for the map integrator, phase is the design phase
        with 0 for maximum energy gain).
    radius : float
        Radius of the element in meters.
    misalignment_error_x : float
        X-axis misalignment error in meters.
    misalignment_error_y : float
        Y-axis misalignment error in meters.
    rotation_error_x : float
        Rotation error about the x-axis in radians.
    rotation_error_y : float
        Rotation error about the y-axis in radians.
    rotation_error_z : float
        Rotation error about the z-axis in radians.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[103] = 103

    field_scaling: float = 0.0
    rf_frequency: float = 0.0
    phase_deg: float = 0.0  # driven phase
    file_id: float = 0.0
    radius: float = 0.0
    misalignment_error_x: float = 0.0
    misalignment_error_y: float = 0.0
    rotation_error_x: float = 0.0
    rotation_error_y: float = 0.0
    rotation_error_z: float = 0.0


class SuperconductingCavity(InputElement, element_id=104, has_input_file=True):
    """
    Attributes
    ----------
    length : float
        Length of the element in meters.
    steps : int
        Number of space-charge kicks through the beamline element. Each
        "step" consists of a half-step, a space-charge kick, and another half-step.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    scale : float
        Field scaling factor.
    rf_frequency : float
        RF frequency in Hz.
    phase_deg : float
        Driven phase in degrees (design phase with 0 for maximum energy gain).
    file_id : float
        Input field ID (if ID < 0, only works for the map integrator).
    radius : float
        Radius in meters.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[104] = 104

    field_scaling: float = 0.0
    rf_frequency: float = 0.0
    phase_deg: float = 0.0  # theta0
    file_id: float = 0.0
    radius: float = 0.0

    # TODO not in the docs:
    misalignment_error_x: float = 0.0
    misalignment_error_y: float = 0.0
    rotation_error_x: float = 0.0
    rotation_error_y: float = 0.0
    rotation_error_z: float = 0.0


class SolenoidWithRFCavity(InputElement, element_id=105, has_input_file=True):
    """
    A solenoid with an RF cavity.

    Attributes
    ----------
    length : float
        Length of the element in meters.
    steps : int
        Number of space-charge kicks through the beamline element. Each
        "step" consists of a half-step, a space-charge kick, and another half-step.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    field_scaling : float
        The field scaling factor.
    rf_frequency : float
        The RF frequency in Hertz.
    phase_deg : float
        The driven phase in degrees.
    file_id : float
        The input field ID.
    radius : float
        The radius of the solenoid in meters.
    misalignment_error_x : float
        The x-axis misalignment error in meters.
    misalignment_error_y : float
        The y-axis misalignment error in meters.
    rotation_error_x : float
        The error in rotation about the x-axis in radians.
    rotation_error_y : float
        The error in rotation about the y-axis in radians.
    rotation_error_z : float
        The error in rotation about the z-axis in radians.
    bz0 : float
        The Bz0 field value in Tesla.
    aperture_size_for_wakefield : float
        The aperture size for wakefield computations.
    gap_size_for_wakefield : float
        The gap size for the wake field.
    length_for_wakefield : float
        The length for wake, indicating RF structure wakefield should be turned on if this value is greater than zero.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[105] = 105

    field_scaling: float = 0.0  # field scaling factor
    rf_frequency: float = 0.0  # RF frequency in Hz
    phase_deg: float = 0.0  # driven phase in degrees
    file_id: float = pydantic.Field(ge=1.0, le=999.0, default=1.0)
    radius: float = 0.0  # radius in meters
    misalignment_error_x: float = 0.0  # x misalignment error in meters
    misalignment_error_y: float = 0.0  # y misalignment error in meters
    rotation_error_x: float = 0.0  # x rotation error in radians
    rotation_error_y: float = 0.0  # y rotation error in radians
    rotation_error_z: float = 0.0  # z rotation error in radians
    bz0: float = 0.0  # Bz0 in Tesla
    aperture_size_for_wakefield: float = 0.0  # aperture size for wakefield
    gap_size_for_wakefield: float = 0.0  # gap size for wake
    # length for wake, RF structure wakefield turned on if > 0
    length_for_wakefield: float = 0.0

    @property
    def rf_wavelength(self) -> float:
        """RF wavelength."""
        return c_light / self.rf_frequency

    @rf_wavelength.setter
    def rf_wavelength(self, value: float) -> None:
        self.rf_frequency = c_light / value


class TravelingWaveRFCavity(InputElement, element_id=106, has_input_file=True):
    """
    Traveling Wave RF Cavity element.

    This element represents a traveling wave RF cavity specified by several
    parameters that define its physical and operational characteristics.

    Attributes
    ----------
    length : float
        Length of the element in meters.
    steps : int
        Number of space-charge kicks through the beamline element. Each
        "step" consists of a half-step, a space-charge kick, and another half-step.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    field_scaling : float
        Scaling factor for the field.
    rf_frequency : float
        RF frequency, in Hertz.
    phase_deg : float
        Driven phase in degrees.
    file_id : float
        Input field ID.
    radius : float
        Radius of the cavity in meters.
    misalignment_error_x : float
        X misalignment error in meters.
    misalignment_error_y : float
        Y misalignment error in meters.
    rotation_error_x : float
        Rotation errors in x [rad].
    rotation_error_y : float
        Rotation errors in x [rad].
    rotation_error_z : float
        Rotation errors in x [rad].
    phase_diff : float
        Phase difference B and A (pi - beta * d).
    aperture_size_for_wakefield : float
        Aperture size for wakefield. An aperture size >0 enables the
        wakefield calculation.
    gap_size : float
        Gap size for wakefield.

    Note
    ----
    RF structure wakefield is only turned on with a non-zero
    `length_for_wakefield`.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[106] = 106

    field_scaling: float = 0.0  # scale
    rf_frequency: float = 0.0  # rf freq
    phase_deg: float = 0.0  # driven_phase
    file_id: float = 0.0  # file_id
    radius: float = 0.0  # radius
    misalignment_error_x: float = 0.0
    misalignment_error_y: float = 0.0
    rotation_error_x: float = 0.0
    rotation_error_y: float = 0.0
    rotation_error_z: float = 0.0
    phase_diff: float = 0.0  # phase diff
    aperture_size_for_wakefield: float = 0.0
    gap_size: float = 0.0
    length_for_wakefield: float = 0.0


class UserDefinedRFCavity(InputElement, element_id=110, has_input_file=True):
    """
    A user-defined RF cavity element in the simulation.

    EMfld in IMPACT-Z.

    Attributes
    ----------
    length : float
        Length of the element in meters.
    steps : int
        Number of space-charge kicks through the beamline element. Each
        "step" consists of a half-step, a space-charge kick, and another half-step.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    field_scaling : float
        Scaling factor for the field.
    rf_frequency : float
        RF frequency in Hertz.
    phase_deg : float
        Driven phase in degrees.
    file_id : float
        ID of the input field.
    x_radius : float
        X radius in meters.
    radius_y : float
        Y radius in meters.
    misalignment_error_x : float
        Misalignment error in the X direction, in meters.
    misalignment_error_y : float
        Misalignment error in the Y direction, in meters.
    rotation_error : tuple
        Rotation error in X, Y, Z directions, in radians.
    data_mode : float
        Mode of using field data. 1.0 uses discrete data only, 2.0 uses both
        discrete data and analytical function, other values use analytical
        function only.
    coordinate_type : float
        Coordinate type for the field. 2.0 for Cartesian coordinates,
        1.0 for Cylindrical coordinates.

    Notes
    -----
    The class allocates electric and magnetic field grids with grid sizes
    defined in the given data format. The grid is adjusted by +1 in
    each dimension compared to the input intervals.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[110] = 110

    field_scaling: float = 0.0
    rf_frequency: float = 0.0
    phase_deg: float = 0.0  # driven phase
    file_id: float = 0.0
    radius_x: float = 0.0
    radius_y: float = 0.0
    misalignment_error_x: float = 0.0
    misalignment_error_y: float = 0.0
    rotation_error_x: float = 0.0
    rotation_error_y: float = 0.0
    rotation_error_z: float = 0.0
    data_mode: RFCavityDataMode = RFCavityDataMode.discrete
    coordinate_type: RFCavityCoordinateType = RFCavityCoordinateType.cartesian


class ShiftCentroid(InputElement, element_id=-1):
    """
    Shift the centroid.

    Attributes
    ----------
    length : float
        Length of the element in meters.
    steps : int
        Number of space-charge kicks through the beamline element. Each
        "step" consists of a half-step, a space-charge kick, and another half-step.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-1] = -1


class WriteFull(InputElement, element_id=-2, has_output_file=True):
    """
    Write the particle distribution into a fort.N file.

    Attributes
    ----------
    length : float
        Unused.
    steps : int
        Unused.
    file_id : int
        The File ID.
    unused_2 : float
        Unused
    sample_frequency : int
        Write every Nth particle.

    Notes
    -----
    - The file written will not support N values of 5, 6, 24, 25, 26, 27, 29,
      30, or 32 when using Fortran code.
    - The printed dataset uses sample frequency `10`, meaning every 10th
      particle is output.
    - Particles recorded are dimensionless, in an IMPACT internal unit.
    - A positive sample frequency specifies magnitude in standard units; a
      negative implies adoption of the ImpactT format (z as delta z and pz as
      gamma).
    """

    length: float = 0.0
    steps: int = 0
    file_id: int = pydantic.Field(
        default=0,
        validation_alias=pydantic.AliasChoices("file_id", "map_steps"),
    )
    type_id: Literal[-2] = -2
    unused_2: float = 0.0
    sample_frequency: int = 0


class DensityProfileInput(InputElement, element_id=-3):
    """
    Input element: density profile input parameters.

    Attributes
    ----------
    length : float
        Unused.
    steps : int
        Unused.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    radius : float
        Radius in meters.
    xmax : float
        Maximum X in meters.
    pxmax : float
        Maximum Px in mc.
    ymax : float
        Maximum Y in meters.
    pymax : float
        Maximum Py in mc.
    zmax : float
        Maximum Z in meters.
    pzmax : float
        Maximum Pz in mc^2.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-3] = -3

    radius: float = 0.0
    xmax: float = 0.0
    pxmax: float = 0.0
    ymax: float = 0.0
    pymax: float = 0.0
    zmax: float = 0.0
    pzmax: float = 0.0


class DensityProfile(InputElement, element_id=-4):
    """
    Input element: write the density along R, X, Y into files.

    Attributes
    ----------
    length : float
        Unused.
    steps : int
        Unused.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    radius : float
        Radius in meters.
    xmax : float
        Maximum value in X direction in meters.
    pxmax : float
        Maximum value in X direction momentum in mc.
    ymax : float
        Maximum value in Y direction in meters.
    pymax : float
        Maximum value in Y direction momentum in mc.
    zmax : float
        Maximum value in Z direction in radians.
    pzmax : float
        Maximum value in Z direction momentum in mc^2.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-4] = -4

    radius: float = 0.0
    xmax: float = 0.0
    pxmax: float = 0.0
    ymax: float = 0.0
    pymax: float = 0.0
    zmax: float = 0.0
    pzmax: float = 0.0


class Projection2D(InputElement, element_id=-5):
    """
    Represents the 2D projections of a 6D distribution.

    Attributes
    ----------
    length : float
        Unused.
    steps : int
        Unused.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    radius : float
        The radius of the projection.
    xmax : float
        The maximum x value.
    pxmax : float
        The maximum px value.
    ymax : float
        The maximum y value.
    pymax : float
        The maximum py value.
    zmax : float
        The maximum z value.
    pzmax : float
        The maximum pz value.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-5] = -5

    radius: float = 0.0
    xmax: float = 0.0
    pxmax: float = 0.0
    ymax: float = 0.0
    pymax: float = 0.0
    zmax: float = 0.0
    pzmax: float = 0.0


class Density3D(InputElement, element_id=-6):
    """
    Input element: 3D density.

    Attributes
    ----------
    length : float
        Unused.
    steps : int
        Unused.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    radius : float
        Radius in meters.
    xmax : float
        Maximum x value in meters.
    pxmax : float
        Maximum px value in mc.
    ymax : float
        Maximum y value in meters.
    pymax : float
        Maximum py value in mc.
    zmax : float
        Maximum z value in degrees.
    pzmax : float
        Maximum pz value in mc^2.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-6] = -6

    radius: float = 0.0
    xmax: float = 0.0
    pxmax: float = 0.0
    ymax: float = 0.0
    pymax: float = 0.0
    zmax: float = 0.0
    pzmax: float = 0.0


class WritePhaseSpaceInfo(InputElement, element_id=-7):
    """
    Input element: write the 6D phase space information and local computation
    domain information.

    Writes to files fort.1000, fort.1001, fort.1002, ...,
    fort.(1000+Nprocessor-1). This function is used for restart purposes.
    """

    # TODO unsupported
    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-7] = -7


class WriteSliceInfo(InputElement, element_id=-8, has_output_file=True):
    """
    Write slice information into file fort.{file_id} using specific slices.

    If the twiss mismatch parameters (alpha_x, etc.) are not provided, the
    mismatch factor will be ignored.

    Attributes
    ----------
    length : float
        Unused.
    steps : int
        Unused.
    file_id : float
        The file ID to write slice information to.
    slices : int
        Number of slices.
    alpha_x : float
        Twiss parameter alpha_x at the location.
    beta_x : float
        Twiss parameter beta_x at the location (m).
    alpha_y : float
        Twiss parameter alpha_y at the location.
    beta_y : float
        Twiss parameter beta_y at the location (m).
    """

    length: float = 0.0
    steps: int = 0
    file_id: int = pydantic.Field(
        default=0,
        validation_alias=pydantic.AliasChoices("file_id", "map_steps"),
    )
    type_id: Literal[-8] = -8

    slices: int = 0
    alpha_x: float = 0.0
    beta_x: float = 0.0
    alpha_y: float = 0.0
    beta_y: float = 0.0


class ScaleMismatchParticle6DCoordinates(InputElement, element_id=-10):
    """
    Scale/mismatch the particle 6D coordinates.

    Attributes
    ----------
    length : float
        Unused.
    steps : int
        Unused.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    radius : float
        The radius, not used in computations.
    xmis : float
        The x-coordinate mismatch.
    pxmis : float
        The px-coordinate mismatch.
    ymis : float
        The y-coordinate mismatch.
    pymis : float
        The py-coordinate mismatch.
    tmis : float
        The time-coordinate mismatch.
    ptmis : float
        The pt-coordinate mismatch.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-10] = -10

    radius: float = 0.0
    xmis: float = 0.0
    pxmis: float = 0.0
    ymis: float = 0.0
    pymis: float = 0.0
    tmis: float = 0.0
    ptmis: float = 0.0


class CollimateBeam(InputElement, element_id=-13):
    """
    Collimate the beam with transverse rectangular aperture sizes.

    Attributes
    ----------
    length : float
        Unused.
    steps : int
        Unused.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    radius : float
        Radius in meters (not used).
    xmin : float
        Minimum x value in meters.
    xmax : float
        Maximum x value in meters.
    ymin : float
        Minimum y value in meters.
    ymax : float
        Maximum y value in meters.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-13] = -13

    radius: float = 0.0
    xmin: float = 0.0
    xmax: float = 0.0
    ymin: float = 0.0
    ymax: float = 0.0


class ToggleSpaceCharge(InputElement, element_id=-14):
    """
    Toggle space charge. Available in IMPACT-Z v2.5+.

    Attributes
    ----------
    length : float
        Unused.
    steps : int
        Unused.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    unused : float
        Not used.
    enable : float
        Toggle space charge on or off.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-14] = -14

    unused: float = 0.0
    enable: float | bool = False


class RotateBeam(InputElement, element_id=-18):
    """
    Rotate the beam with respect to the longitudinal axis.

    Both (x,y), and (px,py) are rotated.

    Attributes
    ----------
    length : float
        Unused.
    steps : int
        Unused.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    radius : float
        The radius in meters.
    tilt : float
        The rotation angle in radians.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-18] = -18

    radius: float = 0.0
    tilt: float = 0.0


class BeamShift(InputElement, element_id=-19):
    """
    BeamShift shifts the beam longitudinally to the bunch centroid.so that <dt>=<dE>=0.

    Attributes
    ----------
    length : float
        Unused.
    steps : int
        Unused.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    unused : float
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-19] = -19

    unused: float = 0.0


class BeamEnergySpread(InputElement, element_id=-20):
    """
    Input element: a beam energy spread input element.

    Attributes
    ----------
    length : float
        Unused.
    steps : int
        Unused.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    radius : float
        The radius (not used).
    energy_spread : float
        The increased energy spread in eV.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-20] = -20

    radius: float = 0.0
    energy_spread: float = 0.0


class ShiftBeamCentroid(InputElement, element_id=-21):
    """
    Shift the beam centroid in 6D phase space.

    Attributes
    ----------
    length : float
        Unused.
    steps : int
        Unused.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    radius : float
        Radius in meters (not used).
    xshift : float
        Shift in x direction in meters.
    pxshift : float
        Shift in x momentum in radians.
    yshift : float
        Shift in y direction in meters.
    pyshift : float
        Shift in y momentum in radians.
    zshift : float
        Shift in z direction in degrees.
    pzshift : float
        Shift in z momentum in MeV.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-21] = -21

    radius: float = 0.0
    xshift: float = 0.0
    pxshift: float = 0.0
    yshift: float = 0.0
    pyshift: float = 0.0
    zshift: float = 0.0
    pzshift: float = 0.0


class IntegratorTypeSwitch(InputElement, element_id=-25):
    """
    Input element: switch the integrator type.

    Attributes
    ----------
    length : float
        Unused.
    steps : int
        Unused.
    integrator_type : IntegratorType
        Integrator type.
    unused : float
    """

    length: float = 0.0
    steps: int = 0
    integrator_type: IntegratorType = pydantic.Field(
        default=IntegratorType.linear_map,
        validation_alias=pydantic.AliasChoices("integrator_type", "map_steps"),
    )
    type_id: Literal[-25] = -25

    unused: float = 0.0


class BeamKickerByRFNonlinearity(InputElement, element_id=-40):
    """
    Beam kicker element that applies a longitudinal kick to the beam by the RF
    nonlinearity.

    Note that the linear part has been included in the map integrator and
    subtracted.

    Attributes
    ----------
    length : float
        Unused.
    steps : int
        Unused.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    radius : float
        Radius in meters (not used).
    vmax : float
        Maximum voltage in volts (V).
    phi0 : float
        Initial phase offset in degrees.
    harm : int
        Harmonic number with respect to the reference frequency.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-40] = -40

    radius: float = 0.0
    vmax: float = 0.0
    phi0: float = 0.0
    harm: int = 0


class RfcavityStructureWakefield(InputElement, element_id=-41, has_input_file=True):
    """
    Input element: read in RF cavity structure wakefield.

    Attributes
    ----------
    length : float
        Unused.
    steps : int
        Unused.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    file_id : float
        The file ID to load from.
    enable_wakefield : float
        -1.0 RF off, 1.0 RF on, < 10 no transverse wakefield effects included
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-41] = -41

    unused: float = 1.0
    file_id: float = 0.0
    enable_wakefield: float = 0.0

    # Minimum file ID to be recognized as an input file. Override the parent class.
    _impactz_min_file_id_: ClassVar[int] = 0

    def set_rf(self, rf_on: bool, transverse_wake_effects: bool) -> None:
        if rf_on:
            if transverse_wake_effects:
                self.enable_wakefield = 10.0
            else:
                self.enable_wakefield = 1.0
        else:
            self.enable_wakefield = -1.0

    @property
    def rf_on(self) -> bool:
        return self.enable_wakefield > 0.0

    @property
    def transverse_wake_effects(self) -> bool:
        return self.enable_wakefield >= 10.0


class EnergyModulation(InputElement, element_id=-52):
    """
    Input element: energy modulation (emulate laser heater).

    Attributes
    ----------
    length : float
        Unused.
    steps : int
        Unused.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    beam_size : float
        The matched beam size in meters.
    laser_wavelength : float
        The laser wavelength in meters.
    energy_spread : float
        The uncorrelated energy spread in eV.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-52] = -52

    beam_size: float = 0.0
    laser_wavelength: float = 0.0
    energy_spread: float = 0.0


class KickBeamUsingMultipole(InputElement, element_id=-55):
    """
    Input element: kick the beam using thin lens multipole.

    Attributes
    ----------
    length : float
        Unused.
    steps : int
        Unused.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    unused : float
        First parameter, "1.0" not used.
    k0 : float
        Dipole strength.
    k1 : float
        Quadrupole strength.
    k2 : float
        Sextupole strength.
    k3 : float
        Octupole strength.
    k4 : float
        Decapole strength.
    k5 : float
        Dodecapole strength.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-55] = -55

    unused: float = 0.0
    k0: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0


class HaltExecution(InputElement, element_id=-99):
    """
    Halt execution at this point in the input file.

    This is useful if you have a big file and want to run part-way through it
    without deleting a lot of lines.

    Attributes
    ----------
    length : float
        Unused.
    steps : int
        Unused.
    map_steps : int
        Unused.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-99] = -99


AnyInputElement = Union[
    Drift,
    Quadrupole,
    ConstantFocusing,
    Solenoid,
    Dipole,
    Multipole,
    Wiggler,
    DTL,
    CCDTL,
    CCL,
    SuperconductingCavity,
    SolenoidWithRFCavity,
    TravelingWaveRFCavity,
    UserDefinedRFCavity,
    ShiftCentroid,
    WriteFull,
    DensityProfileInput,
    DensityProfile,
    Projection2D,
    Density3D,
    WritePhaseSpaceInfo,
    WriteSliceInfo,
    ScaleMismatchParticle6DCoordinates,
    CollimateBeam,
    ToggleSpaceCharge,
    RotateBeam,
    BeamShift,
    BeamEnergySpread,
    ShiftBeamCentroid,
    IntegratorTypeSwitch,
    BeamKickerByRFNonlinearity,
    RfcavityStructureWakefield,
    EnergyModulation,
    KickBeamUsingMultipole,
    HaltExecution,
]


T_InputElement = TypeVar("T_InputElement", bound=InputElement)


class ElementListProxy(list[T_InputElement]):
    """
    A list proxy class for input elements.

    Getting or setting an attribute on an instance of this class will get or
    set that attribute on each element of the list.

    May be used as a normal list with indexing and standard methods such as `.append()`.
    """

    # ** auto-generated section begins ** (see _generate_attr_list_ below)
    Bz0: list[float] | float
    alpha_x: list[float] | float
    alpha_y: list[float] | float
    angle: list[float] | float
    aperture_size_for_wakefield: list[float] | float
    beam_size: list[float] | float
    beta_x: list[float] | float
    beta_y: list[float] | float
    bz0: list[float] | float
    coordinate_type: list[RFCavityCoordinateType] | RFCavityCoordinateType
    data_mode: list[RFCavityDataMode] | RFCavityDataMode
    e1: list[float] | float
    e2: list[float] | float
    enable: list[float | bool] | float | bool
    enable_wakefield: list[float] | float
    energy_spread: list[float] | float
    entrance_curvature: list[float] | float
    exit_curvature: list[float] | float
    field_scaling: list[float] | float
    field_strength: list[float] | float
    fint: list[float] | float
    gap_size: list[float] | float
    gap_size_for_wakefield: list[float] | float
    harm: list[int] | int
    hgap: list[float] | float
    input_switch: list[float] | float
    integrator_type: list[IntegratorType] | IntegratorType
    k0: list[float] | float
    k1: list[float] | float
    k2: list[float] | float
    k3: list[float] | float
    k4: list[float] | float
    k5: list[float] | float
    kx: list[float] | float
    kx0_squared: list[float] | float
    ky0_squared: list[float] | float
    kz0_squared: list[float] | float
    laser_wavelength: list[float] | float
    length: list[float] | float
    length_for_wakefield: list[float] | float
    map_steps: list[int] | int
    max_field_strength: list[float] | float
    metadata: list[dict] | dict
    misalignment_error_x: list[float] | float
    misalignment_error_y: list[float] | float
    multipole_type: list[MultipoleType] | MultipoleType
    name: list[str] | str
    period: list[float] | float
    phase_deg: list[float] | float
    phase_diff: list[float] | float
    phi0: list[float] | float
    ptmis: list[float] | float
    pxmax: list[float] | float
    pxmis: list[float] | float
    pxshift: list[float] | float
    pymax: list[float] | float
    pymis: list[float] | float
    pyshift: list[float] | float
    pzmax: list[float] | float
    pzshift: list[float] | float
    q1_misalignment_error_x: list[float] | float
    q1_misalignment_error_y: list[float] | float
    q1_rotation_error_x: list[float] | float
    q1_rotation_error_y: list[float] | float
    q1_rotation_error_z: list[float] | float
    q2_misalignment_error_x: list[float] | float
    q2_misalignment_error_y: list[float] | float
    q2_rotation_error_x: list[float] | float
    q2_rotation_error_y: list[float] | float
    q2_rotation_error_z: list[float] | float
    quad1_gradient: list[float] | float
    quad1_length: list[float] | float
    quad2_gradient: list[float] | float
    quad2_length: list[float] | float
    radius: list[float] | float
    radius_x: list[float] | float
    radius_y: list[float] | float
    rf_frequency: list[float] | float
    rf_misalignment_error_x: list[float] | float
    rf_misalignment_error_y: list[float] | float
    rf_rotation_error_x: list[float] | float
    rf_rotation_error_y: list[float] | float
    rf_rotation_error_z: list[float] | float
    rotation_error_x: list[float] | float
    rotation_error_y: list[float] | float
    rotation_error_z: list[float] | float
    sample_frequency: list[int] | int
    slices: list[int] | int
    steps: list[int] | int
    tilt: list[float] | float
    tmis: list[float] | float
    unused: list[float] | float
    unused_0: list[float] | float
    unused_1: list[float] | float
    unused_2: list[float] | float
    vmax: list[float] | float
    wiggler_type: list[WigglerType] | WigglerType
    xmax: list[float] | float
    xmin: list[float] | float
    xmis: list[float] | float
    xshift: list[float] | float
    ymax: list[float] | float
    ymin: list[float] | float
    ymis: list[float] | float
    yshift: list[float] | float
    zmax: list[float] | float
    zshift: list[float] | float
    # ** auto-generated section ends ** (see _generate_attr_list_ below)

    @staticmethod
    def _generate_attr_list_():
        eles = typing.get_args(AnyInputElement)
        all_fields = {}
        for ele in eles:
            for name, fld in ele.model_fields.items():
                all_fields.setdefault(name, []).append(fld)

        for name, flds in sorted(all_fields.items()):
            annotation = list(set(fld.annotation for fld in flds))
            if len(annotation) == 1:
                (cls,) = annotation
                if isinstance(cls, types.UnionType):
                    type_name = str(cls)
                elif cls is NonzeroFloat:
                    type_name = "float"
                else:
                    type_name = cls.__name__
                print(f"{name}: list[{type_name}] | {type_name}")

    def __getattr__(self, attr: str):
        return [getattr(element, attr) for element in self]

    def __setattr__(self, attr: str, value) -> None:
        if attr.startswith("_"):
            # Allow setting of "internal" attributes on this proxy object.
            return object.__setattr__(self, attr, value)

        if not len(self):
            return

        if not isinstance(value, Sequence):
            value = [value] * len(self)

        for element, val in zip(self, value):
            setattr(element, attr, val)


def load_rfdata_from_file(path: pathlib.Path) -> np.ndarray:
    return parsers.lines_to_ndarray(parsers.read_input_file(path))


def load_file_data_from_lattice(
    lattice: list[AnyInputElement],
    work_dir: pathlib.Path | None = None,
) -> dict[str, NDArray]:
    file_data = {}
    for idx, ele in enumerate(lattice):
        if ele.input_filename and work_dir is not None:
            if not ele.class_information().has_input_file or not isinstance(
                ele, HasInputFile
            ):
                continue

            ele_file_id = int(ele.file_id)
            ext_data_fn = work_dir / ele.input_filename
            try:
                file_data[str(ele_file_id)] = load_rfdata_from_file(ext_data_fn)
            except FileNotFoundError:
                pass
    return file_data


def lattice_from_input_lines(
    lines: list[parsers.InputLine],
) -> list[AnyInputElement]:
    lattice = []
    seen = {}
    for line in lines:
        # the element name comes from:
        #  1. the inline comment
        #  2. the class name with _{idx} appended
        ele = InputElement.from_line(line)
        lattice.append(ele)

        cls = type(ele)
        seen.setdefault(cls, 0)
        seen[cls] += 1

        if not ele.name:
            ele.name = f"{cls.__name__}_{seen[cls]}"

    return lattice


class ElementWithData(NamedTuple):
    """A tuple of an element with file_data information."""

    ele: AnyInputElement
    file_id: int
    filename: str
    data: NDArray | None


class ZElement(NamedTuple):
    """A tuple of a Z position and its corresponding beamline element."""

    z_start: float
    z_end: float
    ele: AnyInputElement


class ImpactZInput(BaseModel):
    """
    Input settings for an IMPACT-Z run.

    In the docstring, parameters are grouped by their line number in the
    file, where `ncpu_y` starts on line 1.

    String values may be used in place of enumeration instances. For example,

    >>> ImpactZInput(integrator_type="linear_map")

    is equivalent to:

    >>> ImpactZInput(integrator_type=IntegratorType.linear_map)

    Parameters
    ----------
    initial_particles : ParticleGroup or None, optional
        Initial particle distribution. Default is None.
    file_data : dict[str, NDArray], optional
        User-provided external file data, indexed by either file ID or name of
        element - both as strings. Default is an empty dict.

    ncpu_y : int, optional
        Number of processors in y direction. Default is 1.
    ncpu_z : int, optional
        Number of processors in z direction. Default is 1.

    seed : int, optional
        Random number seed. Default is 0.
    n_particle : int, optional
        Number of particles. Default is 0.
    integrator_type : IntegratorType or str, optional
        Type of integrator. Default is `IntegratorType.linear_map`
        (equivalently "linear_map").
    err : int, optional
        Error flag. Default is 1.
    diagnostic_type : DiagnosticType, optional
        Type of diagnostics. Default is DiagnosticType.extended.

    nx : int, optional
        Number of mesh points in x. Default is 0.
    ny : int, optional
        Number of mesh points in y. Default is 0.
    nz : int, optional
        Number of mesh points in z. Default is 0.
    boundary_type : BoundaryType, optional
        Boundary condition type. Default is `BoundaryType.trans_open_longi_open`.
    radius_x : float, optional
        Pipe radius in x direction. Default is 0.0.
    radius_y : float, optional
        Pipe radius in y direction. Default is 0.0.
    z_period_size : float, optional
        Period size in z direction. Default is 0.0.

    distribution : DistributionType, optional
        Particle distribution type. Default is `DistributionType.uniform`.
    restart : int, optional
        Restart flag. Default is 0.
    subcycle : int, optional
        Subcycling flag. Default is 0.
    nbunch : int, optional
        Number of bunches. Default is 0.

    particle_list : list[int], optional
        List of particles. Default is [0].
        This must be the same length as `current_list` and
        `charge_over_mass_list`.

    current_list : list[float], optional
        List of currents for each particle type. Default is [0.0].
        This is used with:
        * `DistributionType.waterbag` and
          `DistributionType.multi_charge_state_gaussian`.

    charge_over_mass_list : list[float], optional
        List of charge-to-mass ratios for each particle type. Default is [0.0].
        This is used with:
        * `DistributionType.waterbag` and
          `DistributionType.multi_charge_state_gaussian`.
        * `CollimateBeam` elements
        * Calculating space charge forces

    twiss_alpha_x : float, optional
        Alpha Twiss parameter in x plane. Default is 0.0.
    twiss_beta_x : float, optional
        Beta Twiss parameter in x plane. Default is 1.0.
    twiss_norm_emit_x : float, optional
        Normalized emittance in x plane. Default is 1e-6.
    twiss_mismatch_x : float, optional
        Mismatch factor for x coordinate. Default is 1.0.
    twiss_mismatch_px : float, optional
        Mismatch factor for px coordinate. Default is 1.0.
    twiss_offset_x : float, optional
        Offset in x coordinate. Default is 0.0.
    twiss_offset_px : float, optional
        Offset in px coordinate. Default is 0.0.

    twiss_alpha_y : float, optional
        Alpha Twiss parameter in y plane. Default is 0.0.
    twiss_beta_y : float, optional
        Beta Twiss parameter in y plane. Default is 1.0.
    twiss_norm_emit_y : float, optional
        Normalized emittance in y plane. Default is 1e-6.
    twiss_mismatch_y : float, optional
        Mismatch factor for y coordinate. Default is 1.0.
    twiss_mismatch_py : float, optional
        Mismatch factor for py coordinate. Default is 1.0.
    twiss_offset_y : float, optional
        Offset in y coordinate. Default is 0.0.
    twiss_offset_py : float, optional
        Offset in py coordinate. Default is 0.0.

    twiss_alpha_z : float, optional
        Alpha Twiss parameter in z plane. Default is 1e-9.
        This must be non-zero, or compute domain calculations in IMPACT-Z
        may lead to crashes.
    twiss_beta_z : float, optional
        Beta Twiss parameter in z plane. Default is 1.0.
    twiss_norm_emit_z : float, optional
        Normalized emittance in z plane. Default is 1e-6.
    twiss_mismatch_z : float, optional
        Mismatch factor for z coordinate. Default is 1.0.
    twiss_mismatch_e_z : float, optional
        Mismatch factor for energy coordinate. Default is 1.0.
    twiss_offset_phase_z : float, optional
        Offset in z phase. Default is 0.0.
    twiss_offset_energy_z : float, optional
        Offset in energy. Default is 0.0.

    average_current : float, optional
        Average beam current in Amperes. Default is 1.0.
    reference_kinetic_energy : float, optional
        Reference particle kinetic energy. Default is 0.0.
    reference_particle_mass : float, optional
        Reference particle mass. Default is 0.0.
    reference_particle_charge : float, optional
        Reference particle charge. Default is 0.0.
    reference_frequency : float, optional
        Reference RF frequency. Default is 0.0.
    initial_phase_ref : float, optional
        Initial phase of reference particle. Default is 0.0.

    lattice : list[AnyInputElement], optional
        List of lattice elements. Default is empty list.

    filename : pathlib.Path or None, optional
        Input file path. Default is None. (Excluded from JSON serialization.)
    verbose : bool, optional
        Verbose output flag when running IMPACT-Z. Default is False.
    """

    initial_particles: PydanticParticleGroup | None = None
    # User-provided external file data, indexed by file number or element name
    file_data: dict[str, NDArray] = pydantic.Field(default={}, repr=False)

    # Line 1
    ncpu_y: int = 1
    ncpu_z: int = 1
    gpu: GPUFlag = GPUFlag.disabled

    # Line 2
    seed: int = 0
    n_particle: int = 0
    integrator_type: IntegratorType = IntegratorType.linear_map
    err: int = 1
    diagnostic_type: DiagnosticType = DiagnosticType.extended

    # Line 3
    nx: int = 0
    ny: int = 0
    nz: int = 0
    boundary_type: BoundaryType = BoundaryType.trans_open_longi_open
    radius_x: float = 0.0
    radius_y: float = 0.0
    z_period_size: float = 0.0

    # Line 4
    distribution: DistributionType = DistributionType.uniform
    restart: int = 0
    subcycle: int = 0
    nbunch: int = 1

    # Line 5
    particle_list: list[int] = [0]

    # Line 6
    current_list: list[float] = [0.0]

    # Line 7
    charge_over_mass_list: list[float] = [0.0]

    # Line 8
    twiss_alpha_x: float = 0.0
    twiss_beta_x: float = 1.0
    twiss_norm_emit_x: float = 1e-6
    twiss_mismatch_x: float = 1.0
    twiss_mismatch_px: float = 1.0
    twiss_offset_x: float = 0.0
    twiss_offset_px: float = 0.0

    # Line 9
    twiss_alpha_y: float = 0.0
    twiss_beta_y: float = 1.0
    twiss_norm_emit_y: float = 1e-6
    twiss_mismatch_y: float = 1.0
    twiss_mismatch_py: float = 1.0
    twiss_offset_y: float = 0.0
    twiss_offset_py: float = 0.0

    # Line 10
    twiss_alpha_z: NonzeroFloat = pydantic.Field(default=1e-9)
    twiss_beta_z: float = 1.0
    twiss_norm_emit_z: float = 1e-6
    twiss_mismatch_z: float = 1.0
    twiss_mismatch_e_z: float = 1.0
    twiss_offset_phase_z: float = 0.0
    twiss_offset_energy_z: float = 0.0

    # Line 11
    average_current: float = 1.0
    reference_kinetic_energy: float = 0.0
    reference_particle_mass: float = 0.0
    reference_particle_charge: float = 0.0
    reference_frequency: float = 0.0
    initial_phase_ref: float = 0.0

    # Line 12+
    lattice: list[AnyInputElement] = []

    # Internal
    filename: pathlib.Path | None = pydantic.Field(default=None, exclude=True)
    verbose: bool = False

    def write_particles_at(
        self,
        elements: Sequence[int | str] | int | str = (),
        *,
        every: Sequence[type[AnyInputElement]] | type[AnyInputElement] | None = None,
        initial_particles: bool = True,
        final_particles: bool = True,
        start_file_id: int = 100,
        suffix: str = "_WRITE",
        in_place: bool = True,
    ) -> list[AnyInputElement] | None:
        """
        Insert WriteFull elements in the lattice to record particle data at
        specified points.

        This function inserts WriteFull elements after each given lattice
        element index or name, and can optionally add them at the beginning and
        end of the lattice to record initial and final particle data,
        respectively.

        Specify `every` to write particles after each element of that type(s).

        Parameters
        ----------
        elements : Sequence[int | str] or int or str
            One or more element indices or names where new WriteFull elements
            will be inserted.
        every : Sequence[class], class or None, optional
            If provided, a WriteFull element will also be inserted after each
            lattice element whose class is in this sequence. By default None.
        initial_particles : bool, optional
            If True, adds a WriteFull element at the beginning of the lattice
            to record the initial particle distribution, by default True.
        final_particles : bool, optional
            If True, adds a WriteFull element at the end of the lattice to
            record the final particle distribution, by default True.
        start_file_id : int, optional
            The file ID to assign to the first newly inserted WriteFull
            element, by default 1.
        suffix : str, optional
            Suffix appended to the names of newly inserted WriteFull elements,
            by default "_WRITE".
        in_place : bool, optional
            If True, the lattice is modified in place; otherwise, a new lattice
            with the modifications is returned, by default True.

        Returns
        -------
        list[AnyInputElement] or None
            If in_place is False, returns the new lattice list with WriteFull
            elements inserted. If in_place is True, the lattice is modified in
            place and None is returned.

        Raises
        ------
        ValueError
            If an element name in the input cannot be found in the lattice.
        """
        if isinstance(elements, (str, int)):
            elements = [elements]

        if every is None:
            every = []
        elif not isinstance(every, Iterable):
            every = [every]

        every = list(every)

        def get_save_indices():
            by_name = self.by_name
            for ele in elements:
                if isinstance(ele, int):
                    idx = ele
                else:
                    try:
                        idx = new_lattice.index(by_name[ele])
                    except Exception:
                        raise ValueError(
                            f"Element {ele} not found in the lattice. Note that WriteFull elements are not supported here."
                        )
                yield idx + 1

            for cls in every:
                for ele in self.by_element.get(cls, []):
                    try:
                        yield new_lattice.index(ele) + 1
                    except IndexError:
                        pass

        new_lattice: list[AnyInputElement] = [
            ele for ele in self.lattice if not isinstance(ele, WriteFull)
        ]
        save_indices = sorted(set(get_save_indices()), reverse=True)

        for idx in save_indices:
            ele = new_lattice[idx - 1]
            if not isinstance(ele, WriteFull):
                new_lattice.insert(idx, WriteFull(name=f"{ele.name}{suffix}"))

        if initial_particles:
            new_lattice.insert(0, WriteFull(name="initial_particles"))
        if final_particles:
            new_lattice.append(WriteFull(name="final_particles"))

        file_id = start_file_id
        for elem in new_lattice:
            if isinstance(elem, WriteFull):
                elem.file_id = file_id
                file_id += 1

        if in_place:
            self.lattice = new_lattice
            return
        return new_lattice

    @property
    def elements_with_data(self) -> list[ElementWithData]:
        eles = []
        for ele in self.lattice:
            if not ele.class_information().has_input_file or not isinstance(
                ele, HasInputFile
            ):
                continue

            fn = ele.input_filename
            if fn is None:
                continue
            file_id = int(ele.file_id)
            logger.info(
                f"Writing file for element {type(ele).__name__}: {fn} (file id={file_id})"
            )

            keys = [file_id, ele.name] if ele.name else [file_id]

            data = None
            for key in keys:
                try:
                    data = self.file_data[str(key)]
                except KeyError:
                    pass

            eles.append(
                ElementWithData(
                    ele=ele,
                    file_id=file_id,
                    filename=fn,
                    data=data,
                )
            )
        return eles

    @classmethod
    def from_file(cls, filename: pathlib.Path | str) -> ImpactZInput:
        lines = parsers.read_input_file(filename)
        return cls._from_parsed_lines(lines, filename=filename)

    @classmethod
    def from_contents(
        cls, contents: str, filename: AnyPath | None = None
    ) -> ImpactZInput:
        lines = parsers.parse_input_lines(contents)
        return cls._from_parsed_lines(lines, filename=filename)

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        track_start: str | None = None,
        track_end: str | None = None,
        *,
        radius_x: float = 0.0,
        radius_y: float = 0.0,
        ncpu_y: int = 1,
        ncpu_z: int = 1,
        nx: int | None = None,
        ny: int | None = None,
        nz: int | None = None,
        which: TaoWhich = "model",
        ix_uni: int = 1,
        ix_branch: int = 0,
        reference_frequency: float = 1300000000.0,
        verbose: bool = False,
        initial_particles_file_id: int = 100,
        final_particles_file_id: int = 101,
        initial_rfdata_file_id: int = 500,
        initial_write_full_id: int = 200,
        write_beam_eles: str | Sequence[str] = ("monitor::*", "marker::*"),
        include_collimation: bool = True,
        integrator_type: IntegratorType = IntegratorType.linear_map,
    ) -> ImpactZInput:
        """
        Create an ImpactZInput object from a Tao instance's lattice.

        This function converts a Tao model into an ImpactZInput by extracting the
        relevant lattice and particle information, and packages it into a structure
        suitable for running IMPACT-Z simulations.

        Parameters
        ----------
        tao : Tao
            The Tao instance.
        track_start : str or None, optional
            Name of the element in the Tao model where tracking begins.
            If None, defaults to the first element.
        track_end : str or None, optional
            Name of the element in the Tao model where tracking ends.
            If None, defaults to the last element.
        radius_x : float, optional
            The transverse aperture radius in the x-dimension.
        radius_y : float, optional
            The transverse aperture radius in the y-dimension.
        ncpu_y : int, optional
            Number of processor divisions along the y-axis.
        ncpu_z : int, optional
            Number of processor divisions along the z-axis.
        nx : int, optional
            Space charge grid mesh points along the x-axis.
            Defaults to space_charge_mesh_size.
        ny : int, optional
            Space charge grid mesh points along the y-axis.
            Defaults to space_charge_mesh_size.
        nz : int, optional
            Space charge grid mesh points along the z-axis.
            Defaults to space_charge_mesh_size.
        which : "model", "base", or "design", optional
            Specifies the source of lattice data used from Tao.
        ix_uni : int, optional
            The universe index.
        ix_branch : int, optional
            The branch index.
        reference_frequency : float, optional
            The reference frequency for IMPACT-Z.
        verbose : bool, optional
            If True, prints additional diagnostic information.
        initial_particles_file_id : int, optional
            File ID for the initial particle distribution.
        final_particles_file_id : int, optional
            File ID for the final particle distribution.
        initial_rfdata_file_id : int, optional
            File ID for the first RF data file.
        initial_write_full_id : int, optional
            File ID for the first WriteFull instance.
        write_beam_eles : str or Sequence[str], optional
            Element(s) by name or Tao-supported match to use at which to write
            particle data via `WriteFull`.
        include_collimation : bool, optional
            If True, includes collimation elements in the lattice conversion.
            Defaults to True.
        integrator_type : IntegratorType, optional
            The integrator scheme to be used in the lattice conversion.
            Defaults to 'linear_map', but this may be switched automatically to
            Runge-Kutta depending on IMPACT-Z run requirements.

        Returns
        -------
        ImpactZInput
        """
        from .interfaces.bmad import ConversionState

        state = ConversionState.from_tao(
            tao=tao,
            track_start=track_start,
            track_end=track_end,
            reference_frequency=reference_frequency,
            ix_uni=ix_uni,
            ix_branch=ix_branch,
            which=which,
            integrator_type=integrator_type,
        )

        lattice, file_data = state.convert_lattice(
            tao=tao,
            verbose=verbose,
            initial_particles_file_id=initial_particles_file_id,
            final_particles_file_id=final_particles_file_id,
            initial_rfdata_file_id=initial_rfdata_file_id,
            initial_write_full_id=initial_write_full_id,
            write_beam_eles=write_beam_eles,
            include_collimation=include_collimation,
        )

        input = state.to_input(
            # tao=tao,
            lattice=lattice,
            file_data=file_data,
            radius_x=radius_x,
            radius_y=radius_y,
            ncpu_y=ncpu_y,
            ncpu_z=ncpu_z,
            nx=nx,
            ny=ny,
            nz=nz,
        )
        return input

    @classmethod
    def _from_parsed_lines(
        cls,
        indexed_lines: list[parsers.InputLine],
        filename: AnyPath | None,
    ) -> ImpactZInput:
        res = cls(filename=pathlib.Path(filename) if filename else None)

        # Casts here are to satisfy the linter. Actual validation will be handled by pydantic.
        if len(indexed_lines[0].data) >= 3:
            # GPU flag is written by the Python GUI but not actually read out
            # by IMPACT-Z.
            res.ncpu_y, res.ncpu_z, res.gpu = cast(
                tuple[int, int, GPUFlag], indexed_lines[0].data[:3]
            )
        else:
            res.ncpu_y, res.ncpu_z = cast(tuple[int, int], indexed_lines[0].data[:2])
        (
            res.seed,
            res.n_particle,
            res.integrator_type,
            res.err,
            res.diagnostic_type,
        ) = cast(
            tuple[int, int, IntegratorType, int, DiagnosticType],
            indexed_lines[1].data[:5],
        )
        (
            res.nx,
            res.ny,
            res.nz,
            res.boundary_type,
            res.radius_x,
            res.radius_y,
            res.z_period_size,
        ) = cast(
            tuple[int, int, int, BoundaryType, float, float, float],
            indexed_lines[2].data[:8],
        )
        (
            res.distribution,
            res.restart,
            res.subcycle,
            res.nbunch,
        ) = cast(tuple[DistributionType, int, int, int], indexed_lines[3].data[:4])

        res.particle_list = [int(v) for v in indexed_lines[4].data]
        res.current_list = [float(v) for v in indexed_lines[5].data]
        res.charge_over_mass_list = [float(v) for v in indexed_lines[6].data]

        (
            res.twiss_alpha_x,
            res.twiss_beta_x,
            res.twiss_norm_emit_x,
            res.twiss_mismatch_x,
            res.twiss_mismatch_px,
            res.twiss_offset_x,
            res.twiss_offset_px,
        ) = indexed_lines[7].data
        (
            res.twiss_alpha_y,
            res.twiss_beta_y,
            res.twiss_norm_emit_y,
            res.twiss_mismatch_y,
            res.twiss_mismatch_py,
            res.twiss_offset_y,
            res.twiss_offset_py,
        ) = indexed_lines[8].data
        (
            res.twiss_alpha_z,
            res.twiss_beta_z,
            res.twiss_norm_emit_z,
            res.twiss_mismatch_z,
            res.twiss_mismatch_e_z,
            res.twiss_offset_phase_z,
            res.twiss_offset_energy_z,
        ) = indexed_lines[9].data

        (
            res.average_current,
            res.reference_kinetic_energy,
            res.reference_particle_mass,
            res.reference_particle_charge,
            res.reference_frequency,
            res.initial_phase_ref,
        ) = indexed_lines[10].data

        if filename is not None:
            work_dir = pathlib.Path(filename).parent
        else:
            work_dir = None

        res.lattice = lattice_from_input_lines(indexed_lines[11:])
        res.file_data = load_file_data_from_lattice(res.lattice, work_dir=work_dir)
        return res

    def to_contents(
        self,
        header="Written by LUME-ImpactZ",
        include_gpu: bool = False,
        include_repr: bool = True,
    ) -> str:
        def stringify_list(lst: Sequence[float | int]):
            return " ".join(str(v) for v in lst)

        if include_gpu:
            gpu = f" {int(self.gpu)}"
        else:
            gpu = ""

        header_lines = [header]
        if include_repr:
            header_lines.extend(repr(self).splitlines())
        full_header = "\n".join(f"! {line}" for line in header_lines)

        lattice = "\n".join(
            ele.ele.to_line(z_start=ele.z_start, with_description=True)
            for ele in self.by_z
        )

        return f"""
{full_header}
! ncpu_y ncpu_z
{self.ncpu_y} {self.ncpu_z}{gpu}
! seed n_particle integrator_type={self.integrator_type.name} err diagnostic_type={self.diagnostic_type.name}
{self.seed} {self.n_particle} {int(self.integrator_type)} {self.err} {int(self.diagnostic_type)}
! nx ny nz boundary_type={self.boundary_type.name} radius_x radius_y z_period_size
{self.nx} {self.ny} {self.nz} {self.boundary_type} {self.radius_x:.20g} {self.radius_y:.20g} {self.z_period_size:.20g}
! distribution={self.distribution.name} restart subcycle nbunch
{self.distribution} {self.restart} {self.subcycle} {self.nbunch}
! particle_list
{stringify_list(self.particle_list)}
! current_list
{stringify_list(self.current_list)}
! charge_over_mass_list
{stringify_list(self.charge_over_mass_list)}
! twiss_alpha_x twiss_beta_x twiss_norm_emit_x twiss_mismatch_x twiss_mismatch_px twiss_offset_x twiss_offset_px
{self.twiss_alpha_x:.20g} {self.twiss_beta_x:.20g} {self.twiss_norm_emit_x:.20g} {self.twiss_mismatch_x:.20g} {self.twiss_mismatch_px:.20g} {self.twiss_offset_x:.20g} {self.twiss_offset_px:.20g}
! twiss_alpha_y twiss_beta_y twiss_norm_emit_y twiss_mismatch_y twiss_mismatch_py twiss_offset_y twiss_offset_py
{self.twiss_alpha_y:.20g} {self.twiss_beta_y:.20g} {self.twiss_norm_emit_y:.20g} {self.twiss_mismatch_y:.20g} {self.twiss_mismatch_py:.20g} {self.twiss_offset_y:.20g} {self.twiss_offset_py:.20g}
! twiss_alpha_z twiss_beta_z twiss_norm_emit_z twiss_mismatch_z twiss_mismatch_e_z twiss_offset_phase_z twiss_offset_energy_z
{self.twiss_alpha_z:.20g} {self.twiss_beta_z:.20g} {self.twiss_norm_emit_z:.20g} {self.twiss_mismatch_z:.20g} {self.twiss_mismatch_e_z:.20g} {self.twiss_offset_phase_z:.20g} {self.twiss_offset_energy_z:.20g}
! average_current reference_kinetic_energy reference_particle_mass reference_particle_charge reference_frequency initial_phase_ref
{self.average_current:.20g} {self.reference_kinetic_energy:.20g} {self.reference_particle_mass:.20g} {self.reference_particle_charge:.20g} {self.reference_frequency:.20g} {self.initial_phase_ref:.20g}
! ** lattice **
{lattice}
        """.strip()

    def get_aligned_initial_particles(
        self, phase_ref: float | None = None
    ) -> ParticleGroup | None:
        """A copy of the initial particles, with time shifted to align with the initial phase."""
        if self.initial_particles is None:
            return None

        if phase_ref is None:
            phase_ref = self.initial_phase_ref

        t_offset = phase_ref / (2 * np.pi * self.reference_frequency)
        particles = self.initial_particles.copy()
        particles.t = particles.t - t_offset

        if len(particles) == 1:
            particles.weight = [0]
        return particles

    def write(
        self,
        workdir: AnyPath,
        error_if_missing: bool = False,
        check: bool = True,
    ) -> list[pathlib.Path]:
        if check:
            self.check(pathlib.Path(workdir))

        contents = self.to_contents()
        workdir = pathlib.Path(workdir)
        if workdir.name == "ImpactZ.in":
            input_file_path = workdir
            workdir = workdir.parent
        else:
            input_file_path = workdir / "ImpactZ.in"

        workdir.mkdir(exist_ok=True)
        with open(input_file_path, "w") as fp:
            print(contents, file=fp)

        extra_paths = []

        initial_particles = self.get_aligned_initial_particles()
        if initial_particles:
            particles_path = workdir / "particle.in"
            iz_particles = ImpactZParticles.from_particle_group(
                initial_particles,
                reference_frequency=self.reference_frequency,
                reference_kinetic_energy=self.reference_kinetic_energy,
            )
            iz_particles.write_impact(particles_path)
            # TODO: support this in openpmd-beamphysics
            # self.initial_particles.write_impact(
            #     str(particles_path),
            #     cathode_kinetic_energy_ref=self.reference_kinetic_energy,
            # )
            extra_paths.append(particles_path)

        for ele in self.lattice:
            if not ele.class_information().has_input_file or not isinstance(
                ele, HasInputFile
            ):
                continue

            fn = ele.input_filename
            if fn is not None:
                file_id = int(ele.file_id)
                logger.info(
                    f"Writing file for element {type(ele).__name__}: {fn} (file id={file_id})"
                )

                keys = [file_id, ele.name] if ele.name else [file_id]

                for key in keys:
                    try:
                        data = self.file_data[str(key)]
                    except KeyError:
                        pass
                    else:
                        np.savetxt(workdir / fn, data)
                        break
                else:
                    if error_if_missing:
                        raise FileNotFoundError(f"Missing input file: {fn}")
                    logger.warning(
                        f"Expected input file not found: {fn} (file id={file_id})"
                    )

        return [input_file_path, *extra_paths]

    @pydantic.model_validator(mode="before")
    @classmethod
    def _update_n_particle(cls, data: dict[str, Any]) -> dict[str, Any]:
        initial_particles = data.get("initial_particles", None)
        if isinstance(initial_particles, ParticleGroup):
            if len(initial_particles) == 1:
                current = 0.0
            else:
                current = float(initial_particles.charge) * data.get(
                    "reference_frequency", 0.0
                )

            data["n_particle"] = len(initial_particles)
            data["distribution"] = DistributionType.read
            data["particle_list"] = [len(initial_particles)]
            data["current_list"] = [current]
            data["charge_over_mass_list"] = [
                float(initial_particles.species_charge / e / initial_particles.mass)
            ]

        return data

    def update_particle_parameters(self) -> None:
        """
        Update all relevant parameters from `self.initial_particles`.

        This method configures particle-related parameters based on the initial particle
        distribution. If no initial particles are defined, it sets default values. Otherwise,
        it configures the distribution type, particle count, current, and charge-to-mass ratio
        based on the initial particles.

        Notes
        -----
        It is not typically required to call this separately.  When setting
        `initial_particles`, a Pydantic validator will set these appropriately.

        The following attributes are updated:
        - `self.distribution`: Set to DistributionType.read if initial particles exist
        - `self.n_particle`: The number of particles
        - `self.particle_list`: List containing the number of particles
        - `self.current_list`: List containing the particle charge scaled by reference frequency
        - `self.charge_over_mass_list`: List containing the charge-to-mass ratio
        """
        if self.initial_particles is None:
            self.particle_list = [self.n_particle]
            self.current_list = [0.0]
            self.charge_over_mass_list = [0.0]
            return

        self.distribution = DistributionType.read
        self.n_particle = len(self.initial_particles)
        self.particle_list = [self.n_particle]
        self.current_list = [self.initial_particles.charge * self.reference_frequency]
        self.charge_over_mass_list = [
            self.initial_particles.species_charge / e / self.initial_particles.mass
        ]

    def check(self, workdir: pathlib.Path = pathlib.Path(".")):
        # if self.seed < 0:
        #     self.seed = 6
        if self.initial_particles is not None:
            num_particles = len(self.initial_particles)
            if num_particles == 0:
                raise ValueError(
                    f"Initial particles is set to an empty ParticleGroup: {self.initial_particles}"
                )
            if num_particles == 1:
                self.current_list = [0.0]
                self.space_charge_off()
        elif self.distribution == DistributionType.read:
            # No particles and 'read' mode may not work:
            raise ValueError(
                "Initial particles unset, yet distribution='read'. "
                "To have IMPACT-Z generate particles, use distribution='uniform' or "
                "one of the supported values (in `DistributionType`)"
            )

        if (len(self.particle_list) != len(self.charge_over_mass_list)) or (
            len(self.particle_list) != len(self.current_list)
        ):
            raise ValueError(
                f"`particle_list`, `charge_over_mass_list`, and `current_list` must all have the same length.\n"
                f"Got lengths: particle_list={len(self.particle_list)}, charge_over_mass_list={len(self.charge_over_mass_list)}, "
                f"current_list={len(self.current_list)}."
            )

        self.nbunch = len(self.particle_list)

        by_ele = self.by_element
        for ele in self.elements_with_data:
            if ele.data is None:
                cls = type(ele.ele)
                default_name = f"{cls.__name__} #{by_ele[cls].index(ele.ele) + 1}"
                name = ele.ele.name or default_name

                path = workdir / ele.filename
                if not path.exists():
                    logger.warning(
                        f"Element {name} may be missing a file. "
                        f"Set file_data for ID {ele.file_id} or element name {ele.ele.name!r} to fix this. "
                    )

    def write_run_script(
        self,
        path: pathlib.Path,
        command_prefix: str = "ImpactZexe",
    ) -> None:
        path.parent.mkdir(exist_ok=True)
        with open(path, mode="w") as fp:
            print(shlex.join(shlex.split(command_prefix)), file=fp)
        lume_tools.make_executable(str(path))

    @property
    def by_name(self) -> dict[str, AnyInputElement]:
        return {ele.name: ele for ele in self.lattice if ele.name}

    @property
    def by_z(self) -> list[ZElement]:
        """
        Get all (flattened) beamline elements by Z location.

        Returns
        -------
        list of (zend, element)
            Each list item is a ZElement, a namedtuple which has `.zend` and
            `.element` that is also usable as a normal tuple.
        """
        z = 0.0
        by_z = []
        for ele in self.lattice:
            by_z.append(
                ZElement(
                    z_start=z,
                    z_end=z + ele.length,
                    ele=ele,
                )
            )
            z += ele.length

        return by_z

    @property
    def by_element(
        self,
    ) -> dict[type[T_InputElement], ElementListProxy[T_InputElement]]:
        """Get beamline elements organized by their class."""
        by_element = {}
        for element in self.lattice:
            by_element.setdefault(type(element), ElementListProxy[type(element)]())
            by_element[type(element)].append(element)
        return by_element

    def _get_only_one(self, cls: type[T_InputElement]) -> T_InputElement:
        items = self.by_element.get(cls, [])
        if len(items) == 0:
            raise NoSuchElementError(
                f"No '{cls.__name__}' instances are defined in the input lattice"
            )
        if len(items) > 1:
            plural_fix = {}
            plural = pydantic.alias_generators.to_snake(cls.__name__)
            plural = plural_fix.get(plural, f"{plural}s")
            raise MultipleElementError(
                f"Multiple {cls.__name__} namelists were defined in the input. "
                f"Please use .{plural}"
            )
        return items[0]

    def space_charge_off(self) -> None:
        self.average_current = 0.0

    def space_charge_on(self, bunch_charge: float | None = None) -> None:
        if bunch_charge is None:
            if self.initial_particles is None:
                raise ValueError(
                    "Must specify `bunch_charge` if `initial_particles` is unset"
                )
            bunch_charge = float(self.initial_particles.charge)

        self.bunch_charge = bunch_charge

    @property
    def bunch_charge(self) -> float:
        """Bunch charge, if space charge is enabled."""
        return self.average_current / self.reference_frequency

    @bunch_charge.setter
    def bunch_charge(self, bunch_charge: float) -> None:
        self.average_current = bunch_charge * self.reference_frequency

    @property
    def reference_species(self) -> str:
        return detect_species(
            self.reference_particle_charge / self.reference_particle_mass
        )

    @property
    def total_charge(self) -> float:
        """Returns the total bunch charge in C. Can be set."""
        return self.average_current / self.reference_frequency

    @total_charge.setter
    def total_charge(self, charge: float) -> None:
        self.average_current = charge * self.reference_frequency
        # Keep particles up-to-date.
        if self.initial_particles is not None and charge > 0.0:
            self.initial_particles.charge = charge

    @property
    def nproc(self):
        """Number of MPI processors."""
        return self.ncpu_y * self.ncpu_z

    @nproc.setter
    def nproc(self, n: int | None) -> None:
        n = int(n or 0)
        if n <= 0:
            n = tools.get_suggested_nproc() + n

        Npcol, Nprow = suggested_processor_domain(self.nz, self.ny, n)

        self.ncpu_y = Npcol
        self.ncpu_z = Nprow

        if self.verbose:
            mpi = " (MPI enabled)" if self.nproc > 1 else ""
            print(f"Setting Npcol, Nprow = {Npcol}, {Nprow}{mpi}")

    def archive(self, h5: h5py.Group) -> None:
        """
        Dump input data into the given HDF5 group.

        Parameters
        ----------
        h5 : h5py.Group
            The HDF5 file in which to write the information.
        """
        _archive.store_in_hdf5_file(h5, self)

    @classmethod
    def from_archive(cls, h5: h5py.Group) -> ImpactZInput:
        """
        Loads input from archived h5 file.

        Parameters
        ----------
        h5 : str or h5py.File
            The filename or handle on h5py.File from which to load data.
        """
        loaded = _archive.restore_from_hdf5_file(h5)
        if not isinstance(loaded, ImpactZInput):
            raise ValueError(
                f"Loaded {loaded.__class__.__name__} instead of a "
                f"ImpactZInput instance.  Was the HDF group correct?"
            )
        return loaded

    @property
    def sigma_t(self) -> float:
        """Calculated RMS bunch duration (s)."""
        fref = self.reference_frequency

        scale_z = self.twiss_mismatch_z
        emit = self.twiss_norm_emit_z
        beta = self.twiss_beta_z

        return np.sqrt(beta * emit) / (fref * 360) * scale_z

    @property
    def sigma_energy(self) -> float:
        """Calculated RMS energy spread (eV)."""

        alpha = self.twiss_alpha_z
        emit = self.twiss_norm_emit_z
        beta = self.twiss_beta_z
        gamma = (1 + alpha**2) / beta
        scale_e_z = self.twiss_mismatch_e_z

        return np.sqrt(gamma * emit) * 1e6 * scale_e_z

    @property
    def cov_t__energy(self) -> float:
        """Calculated <t, energy> (eV*s)."""
        fref = self.reference_frequency
        alpha = self.twiss_alpha_z
        emit = self.twiss_norm_emit_z

        scale_z = self.twiss_mismatch_z
        scale_e_z = self.twiss_mismatch_e_z

        return -alpha * emit / (fref * 360) * 1e6 * scale_z * scale_e_z

    def set_twiss_z(
        self,
        sigma_t: float,
        sigma_energy: float,
        cov_t__energy: float = 0.0,
    ) -> None:
        """
        Sets `twiss_alpha_z`, `twiss_beta_z`, and `twiss_norm_emit_z` from
        standard physical beam quantities.

        Note
        ----
        Requires that `sigma_t * sigma_energy >= abs(cov_t__energy)`.

        Parameters
        ----------
        sigma_t : float
            Initial RMS bunch duration (s)
        sigma_energy : float
            Initial RMS energy spread (eV)
        cov_t__energy : float
            Initial <t, energy>  (eV*s)
        """

        if sigma_t * sigma_energy < abs(cov_t__energy):
            raise ValueError(
                f"sigma_t * sigma_energy ({sigma_t * sigma_energy} eV*s) must be >= abs(cov_t__energy) ({abs(cov_t__energy)} eV*s)"
            )

        fref = self.reference_frequency
        scale_z = self.twiss_mismatch_z
        scale_e_z = self.twiss_mismatch_e_z

        sig_t = sigma_t / scale_z
        sig_e = sigma_energy / scale_e_z
        cov = cov_t__energy / (scale_z * scale_e_z)

        emit = np.sqrt(
            (fref * sig_t * 360) ** 2 * (sig_e / 1e6) ** 2
            - (fref * cov / 1e6 * 360) ** 2
        )
        if emit <= 0.0:
            raise ValueError("Calculated `twiss_norm_emit_z` <= 0.0")

        alpha_z = -cov * fref * 360 / 1e6 / emit

        if alpha_z == 0.0:
            # twiss_alpha_z must be nonzero
            alpha_z = 1e-9

        self.twiss_norm_emit_z = emit
        self.twiss_alpha_z = alpha_z
        self.twiss_beta_z = (fref * sig_t * 360) ** 2 / emit

    def plot(
        self,
        *,
        ax: matplotlib.axes.Axes | None = None,
        bounds: tuple[float, float] | None = None,
        include_labels: bool = True,
        include_markers: bool = True,
        include_marker_labels: bool | None = None,
        figsize: tuple[int, int] = (6, 2),
    ):
        """
        Make a matplotlib plot of the lattice layout.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes object on which to draw the layout. If None, a new figure and axes
            are created.
        bounds : (float, float), optional
            Lower and upper bounds for z position. Defaults to None.
        include_labels : bool, optional
            Whether to include element labels in the plot. Defaults to True.
        include_markers : bool, optional
            If True, include zero length markers in the plot. Default is True.
        include_marker_labels : bool, optional
            If True, include labels for markers when `include_markers` is set.
            Default is `include_labels`.
        figsize : tuple of int, optional
            Size of the figure in inches (width, height) when a new figure is created.
            Defaults to (6, 2).

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plotted layout.
        """
        from .plot import plot_layout

        if include_marker_labels is None:
            include_marker_labels = include_labels

        return plot_layout(
            by_z=self.by_z,
            ax=ax,
            bounds=bounds,
            include_labels=include_labels,
            include_markers=include_markers,
            include_marker_labels=include_marker_labels,
            figsize=figsize,
        )

    @property
    def bounds(self):
        """
        Calculate the phase bounds based on the Twiss settings and distribution type.

        Returns
        -------
        (xmin, xmax)
        (ymin, ymax)
        (zmin, zmax)
        """
        sig11x = self.twiss_alpha_x * self.twiss_mismatch_x
        # sig22x = self.twiss_beta_x * self.twiss_mismatch_px
        sig11y = self.twiss_alpha_y * self.twiss_mismatch_y
        # sig22y = self.twiss_beta_y * self.twiss_mismatch_py
        sig11z = self.twiss_alpha_z * self.twiss_mismatch_z
        # sig22z = self.twiss_beta_z * self.twiss_mismatch_e_z

        xy_factor, z_factor = {
            DistributionType.uniform: (np.sqrt(3.0), np.sqrt(3.0)),
            DistributionType.gauss: (4.0, 4.0),
            DistributionType.waterBag: (np.sqrt(8.0), np.sqrt(8.0)),
            DistributionType.semiGauss: (np.sqrt(5.0), np.sqrt(5.0)),
            DistributionType.unknown: (2.0, 2.0),  # read in or fortran bug?
            DistributionType.kV: (2.0, np.sqrt(3.0)),
        }.get(self.distribution, (np.sqrt(3.0), np.sqrt(3.0)))

        emit_x_sqr = self.twiss_norm_emit_x * self.twiss_norm_emit_x
        emit_y_sqr = self.twiss_norm_emit_y * self.twiss_norm_emit_y
        emit_z_sqr = self.twiss_norm_emit_z * self.twiss_norm_emit_z

        xmin = self.twiss_offset_x - xy_factor * sig11x / np.sqrt(1.0 - emit_x_sqr)
        xmax = self.twiss_offset_x + xy_factor * sig11x / np.sqrt(1.0 - emit_x_sqr)
        ymin = self.twiss_offset_y - xy_factor * sig11y / np.sqrt(1.0 - emit_y_sqr)
        ymax = self.twiss_offset_y + xy_factor * sig11y / np.sqrt(1.0 - emit_y_sqr)
        zmin = self.twiss_offset_phase_z - z_factor * sig11z / np.sqrt(1.0 - emit_z_sqr)
        zmax = self.twiss_offset_phase_z + z_factor * sig11z / np.sqrt(1.0 - emit_z_sqr)

        if self.boundary_type == BoundaryType.trans_round_longi_open:
            xmin = 0.0
            xmax = self.radius_x
            ymin = 0.0
            ymax = 4 * np.pi / 2.0
        elif self.boundary_type == BoundaryType.trans_round_longi_period:
            xmin = 0.0
            xmax = self.radius_x
            ymin = 0.0
            ymax = 4 * np.pi / 2.0
            zmin = -self.z_period_size / 2
            zmax = self.z_period_size / 2

        return (
            (float(xmin), float(xmax)),
            (float(ymin), float(ymax)),
            (float(zmin), float(zmax)),
        )

    # @property
    # def LOWERs(self) -> ElementListProxy[ELEMENT]:
    #     """List of all ELEMENT instances."""
    #     return self.by_element.get(ELEMENT, ElementListProxy())
    #
    # @property
    # def LOWER(self) -> ELEMENT:
    #     """Get the sole ELEMENT if it is defined."""
    #     return self._get_only_one(ELEMENT)
    #

    @property
    def drifts(self) -> ElementListProxy[Drift]:
        """List of all Drift instances."""
        return self.by_element.get(Drift, ElementListProxy())

    @property
    def drift(self) -> Drift:
        """Get the sole Drift if it is defined."""
        return self._get_only_one(Drift)

    @property
    def quadrupoles(self) -> ElementListProxy[Quadrupole]:
        """List of all Quadrupole instances."""
        return self.by_element.get(Quadrupole, ElementListProxy())

    @property
    def quadrupole(self) -> Quadrupole:
        """Get the sole Quadrupole if it is defined."""
        return self._get_only_one(Quadrupole)

    @property
    def constant_focusings(self) -> ElementListProxy[ConstantFocusing]:
        """List of all ConstantFocusing instances."""
        return self.by_element.get(ConstantFocusing, ElementListProxy())

    @property
    def constant_focusing(self) -> ConstantFocusing:
        """Get the sole ConstantFocusing if it is defined."""
        return self._get_only_one(ConstantFocusing)

    @property
    def solenoids(self) -> ElementListProxy[Solenoid]:
        """List of all Solenoid instances."""
        return self.by_element.get(Solenoid, ElementListProxy())

    @property
    def solenoid(self) -> Solenoid:
        """Get the sole Solenoid if it is defined."""
        return self._get_only_one(Solenoid)

    @property
    def dipoles(self) -> ElementListProxy[Dipole]:
        """List of all Dipole instances."""
        return self.by_element.get(Dipole, ElementListProxy())

    @property
    def dipole(self) -> Dipole:
        """Get the sole Dipole if it is defined."""
        return self._get_only_one(Dipole)

    @property
    def multipoles(self) -> ElementListProxy[Multipole]:
        """List of all Multipole instances."""
        return self.by_element.get(Multipole, ElementListProxy())

    @property
    def multipole(self) -> Multipole:
        """Get the sole Multipole if it is defined."""
        return self._get_only_one(Multipole)

    @property
    def wigglers(self) -> ElementListProxy[Wiggler]:
        """List of all wiggler instances."""
        return self.by_element.get(Wiggler, ElementListProxy())

    @property
    def wiggler(self) -> Wiggler:
        """Get the sole Wiggler if it is defined."""
        return self._get_only_one(Wiggler)

    @property
    def dtls(self) -> ElementListProxy[DTL]:
        """List of all DTL instances."""
        return self.by_element.get(DTL, ElementListProxy())

    @property
    def dtl(self) -> DTL:
        """Get the sole DTL if it is defined."""
        return self._get_only_one(DTL)

    @property
    def ccdtls(self) -> ElementListProxy[CCDTL]:
        """List of all CCDTL instances."""
        return self.by_element.get(CCDTL, ElementListProxy())

    @property
    def ccdtl(self) -> CCDTL:
        """Get the sole CCDTL if it is defined."""
        return self._get_only_one(CCDTL)

    @property
    def ccls(self) -> ElementListProxy[CCL]:
        """List of all CCL instances."""
        return self.by_element.get(CCL, ElementListProxy())

    @property
    def ccl(self) -> CCL:
        """Get the sole CCL if it is defined."""
        return self._get_only_one(CCL)

    @property
    def superconducting_cavitys(self) -> ElementListProxy[SuperconductingCavity]:
        """List of all SuperconductingCavity instances."""
        return self.by_element.get(SuperconductingCavity, ElementListProxy())

    @property
    def superconducting_cavity(self) -> SuperconductingCavity:
        """Get the sole SuperconductingCavity if it is defined."""
        return self._get_only_one(SuperconductingCavity)

    @property
    def solenoid_with_rf_cavitys(self) -> ElementListProxy[SolenoidWithRFCavity]:
        """List of all SolenoidWithRFCavity instances."""
        return self.by_element.get(SolenoidWithRFCavity, ElementListProxy())

    @property
    def solenoid_with_rf_cavity(self) -> SolenoidWithRFCavity:
        """Get the sole SolenoidWithRFCavity if it is defined."""
        return self._get_only_one(SolenoidWithRFCavity)

    @property
    def traveling_wave_rf_cavitys(self) -> ElementListProxy[TravelingWaveRFCavity]:
        """List of all TravelingWaveRFCavity instances."""
        return self.by_element.get(TravelingWaveRFCavity, ElementListProxy())

    @property
    def traveling_wave_rf_cavity(self) -> TravelingWaveRFCavity:
        """Get the sole TravelingWaveRFCavity if it is defined."""
        return self._get_only_one(TravelingWaveRFCavity)

    @property
    def user_defined_rf_cavitys(self) -> ElementListProxy[UserDefinedRFCavity]:
        """List of all UserDefinedRFCavity instances."""
        return self.by_element.get(UserDefinedRFCavity, ElementListProxy())

    @property
    def user_defined_rf_cavity(self) -> UserDefinedRFCavity:
        """Get the sole UserDefinedRFCavity if it is defined."""
        return self._get_only_one(UserDefinedRFCavity)

    @property
    def shift_centroids(self) -> ElementListProxy[ShiftCentroid]:
        """List of all ShiftCentroid instances."""
        return self.by_element.get(ShiftCentroid, ElementListProxy())

    @property
    def shift_centroid(self) -> ShiftCentroid:
        """Get the sole ShiftCentroid if it is defined."""
        return self._get_only_one(ShiftCentroid)

    @property
    def write_fulls(self) -> ElementListProxy[WriteFull]:
        """List of all WriteFull instances."""
        return self.by_element.get(WriteFull, ElementListProxy())

    @property
    def write_full(self) -> WriteFull:
        """Get the sole WriteFull if it is defined."""
        return self._get_only_one(WriteFull)

    @property
    def density_profile_inputs(self) -> ElementListProxy[DensityProfileInput]:
        """List of all DensityProfileInput instances."""
        return self.by_element.get(DensityProfileInput, ElementListProxy())

    @property
    def density_profile_input(self) -> DensityProfileInput:
        """Get the sole DensityProfileInput if it is defined."""
        return self._get_only_one(DensityProfileInput)

    @property
    def density_profiles(self) -> ElementListProxy[DensityProfile]:
        """List of all DensityProfile instances."""
        return self.by_element.get(DensityProfile, ElementListProxy())

    @property
    def density_profile(self) -> DensityProfile:
        """Get the sole DensityProfile if it is defined."""
        return self._get_only_one(DensityProfile)

    @property
    def projection_2ds(self) -> ElementListProxy[Projection2D]:
        """List of all Projection2D instances."""
        return self.by_element.get(Projection2D, ElementListProxy())

    @property
    def projection_2d(self) -> Projection2D:
        """Get the sole Projection2D if it is defined."""
        return self._get_only_one(Projection2D)

    @property
    def density_3ds(self) -> ElementListProxy[Density3D]:
        """List of all Density3D instances."""
        return self.by_element.get(Density3D, ElementListProxy())

    @property
    def density_3d(self) -> Density3D:
        """Get the sole Density3D if it is defined."""
        return self._get_only_one(Density3D)

    @property
    def write_phase_space_infos(self) -> ElementListProxy[WritePhaseSpaceInfo]:
        """List of all WritePhaseSpaceInfo instances."""
        return self.by_element.get(WritePhaseSpaceInfo, ElementListProxy())

    @property
    def write_phase_space_info(self) -> WritePhaseSpaceInfo:
        """Get the sole WritePhaseSpaceInfo if it is defined."""
        return self._get_only_one(WritePhaseSpaceInfo)

    @property
    def write_slice_infos(self) -> ElementListProxy[WriteSliceInfo]:
        """List of all WriteSliceInfo instances."""
        return self.by_element.get(WriteSliceInfo, ElementListProxy())

    @property
    def write_slice_info(self) -> WriteSliceInfo:
        """Get the sole WriteSliceInfo if it is defined."""
        return self._get_only_one(WriteSliceInfo)

    @property
    def scale_mismatch_particle_6d_coordinatess(
        self,
    ) -> ElementListProxy[ScaleMismatchParticle6DCoordinates]:
        """List of all ScaleMismatchParticle6DCoordinates instances."""
        return self.by_element.get(
            ScaleMismatchParticle6DCoordinates, ElementListProxy()
        )

    @property
    def scale_mismatch_particle_6d_coordinates(
        self,
    ) -> ScaleMismatchParticle6DCoordinates:
        """Get the sole ScaleMismatchParticle6DCoordinates if it is defined."""
        return self._get_only_one(ScaleMismatchParticle6DCoordinates)

    @property
    def collimate_beams(self) -> ElementListProxy[CollimateBeam]:
        """List of all CollimateBeam instances."""
        return self.by_element.get(CollimateBeam, ElementListProxy())

    @property
    def collimate_beam(self) -> CollimateBeam:
        """Get the sole CollimateBeam if it is defined."""
        return self._get_only_one(CollimateBeam)

    @property
    def toggle_space_charges(self) -> ElementListProxy[ToggleSpaceCharge]:
        """List of all ToggleSpaceCharge instances."""
        return self.by_element.get(ToggleSpaceCharge, ElementListProxy())

    @property
    def toggle_space_charge(self) -> ToggleSpaceCharge:
        """Get the sole ToggleSpaceCharge if it is defined."""
        return self._get_only_one(ToggleSpaceCharge)

    @property
    def rotate_beams(
        self,
    ) -> ElementListProxy[RotateBeam]:
        """List of all RotateBeam instances."""
        return self.by_element.get(RotateBeam, ElementListProxy())

    @property
    def rotate_beam(
        self,
    ) -> RotateBeam:
        """Get the sole RotateBeam if it is defined."""
        return self._get_only_one(RotateBeam)

    @property
    def beam_shifts(self) -> ElementListProxy[BeamShift]:
        """List of all BeamShift instances."""
        return self.by_element.get(BeamShift, ElementListProxy())

    @property
    def beam_shift(self) -> BeamShift:
        """Get the sole BeamShift if it is defined."""
        return self._get_only_one(BeamShift)

    @property
    def beam_energy_spreads(self) -> ElementListProxy[BeamEnergySpread]:
        """List of all BeamEnergySpread instances."""
        return self.by_element.get(BeamEnergySpread, ElementListProxy())

    @property
    def beam_energy_spread(self) -> BeamEnergySpread:
        """Get the sole BeamEnergySpread if it is defined."""
        return self._get_only_one(BeamEnergySpread)

    @property
    def shift_beam_centroids(self) -> ElementListProxy[ShiftBeamCentroid]:
        """List of all ShiftBeamCentroid instances."""
        return self.by_element.get(ShiftBeamCentroid, ElementListProxy())

    @property
    def shift_beam_centroid(self) -> ShiftBeamCentroid:
        """Get the sole ShiftBeamCentroid if it is defined."""
        return self._get_only_one(ShiftBeamCentroid)

    @property
    def integrator_type_switchs(self) -> ElementListProxy[IntegratorTypeSwitch]:
        """List of all IntegratorTypeSwitch instances."""
        return self.by_element.get(IntegratorTypeSwitch, ElementListProxy())

    @property
    def integrator_type_switch(self) -> IntegratorTypeSwitch:
        """Get the sole IntegratorTypeSwitch if it is defined."""
        return self._get_only_one(IntegratorTypeSwitch)

    @property
    def beam_kicker_by_rf_nonlinearitys(
        self,
    ) -> ElementListProxy[BeamKickerByRFNonlinearity]:
        """List of all BeamKickerByRFNonlinearity instances."""
        return self.by_element.get(BeamKickerByRFNonlinearity, ElementListProxy())

    @property
    def beam_kicker_by_rf_nonlinearity(self) -> BeamKickerByRFNonlinearity:
        """Get the sole BeamKickerByRFNonlinearity if it is defined."""
        return self._get_only_one(BeamKickerByRFNonlinearity)

    @property
    def rfcavity_structure_wakefields(
        self,
    ) -> ElementListProxy[RfcavityStructureWakefield]:
        """List of all RfcavityStructureWakefield instances."""
        return self.by_element.get(RfcavityStructureWakefield, ElementListProxy())

    @property
    def rfcavity_structure_wakefield(self) -> RfcavityStructureWakefield:
        """Get the sole RfcavityStructureWakefield if it is defined."""
        return self._get_only_one(RfcavityStructureWakefield)

    @property
    def energy_modulations(self) -> ElementListProxy[EnergyModulation]:
        """List of all EnergyModulation instances."""
        return self.by_element.get(EnergyModulation, ElementListProxy())

    @property
    def energy_modulation(self) -> EnergyModulation:
        """Get the sole EnergyModulation if it is defined."""
        return self._get_only_one(EnergyModulation)

    @property
    def kick_beam_using_multipoles(self) -> ElementListProxy[KickBeamUsingMultipole]:
        """List of all KickBeamUsingMultipole instances."""
        return self.by_element.get(KickBeamUsingMultipole, ElementListProxy())

    @property
    def kick_beam_using_multipole(self) -> KickBeamUsingMultipole:
        """Get the sole KickBeamUsingMultipole if it is defined."""
        return self._get_only_one(KickBeamUsingMultipole)

    @property
    def halt_executions(self) -> ElementListProxy[HaltExecution]:
        """List of all HaltExecution instances."""
        return self.by_element.get(HaltExecution, ElementListProxy())

    @property
    def halt_execution(self) -> HaltExecution:
        """Get the sole HaltExecution if it is defined."""
        return self._get_only_one(HaltExecution)
