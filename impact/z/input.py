from __future__ import annotations

from abc import abstractmethod
import logging
import pathlib
import shlex
from typing import ClassVar, Literal, cast
from collections.abc import Sequence

import numpy as np
import pydantic
from lume import tools as lume_tools
from typing_extensions import Protocol, runtime_checkable

from impact.z.particles import ImpactZParticles

from . import parsers
from .constants import (
    BoundaryType,
    DiagnosticType,
    DistributionZType,
    GPUFlag,
    IntegratorType,
    MultipoleType,
    OutputZType,
    RFCavityCoordinateType,
    RFCavityDataMode,
)
from .types import AnyPath, BaseModel, NDArray, PydanticParticleGroup

input_element_by_id = {}
logger = logging.getLogger(__name__)


class InputElementMetadata(BaseModel):
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


class InputElement(BaseModel):
    _impactz_metadata_: ClassVar[InputElementMetadata]

    def __init_subclass__(
        cls,
        element_id: int,
        has_input_file: bool = False,
        has_output_file: bool = False,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)

        assert isinstance(element_id, int)
        assert (
            element_id not in input_element_by_id
        ), f"Duplicate element ID {element_id}"
        input_element_by_id[element_id] = cls
        cls._impactz_metadata_ = InputElementMetadata(
            element_id=element_id,
            has_input_file=has_input_file,
            has_output_file=has_output_file,
        )

    @classmethod
    def class_information(cls):
        return cls._impactz_metadata_

    @staticmethod
    def from_line(line: str | parsers.InputLine):
        if isinstance(line, str):
            parts = parsers.parse_input_line(line)
        else:
            parts = line

        type_idx = parts[3]
        ele_cls = input_element_by_id[type_idx]

        # if ele_cls is Drift and len(parts) == 7:
        #     # TODO: a known bit of 'extra' data in the examples
        #     # patching in a hotfix here, but we may adjust later...
        #     parts = parts[:5]
        if len(parts) > len(ele_cls.model_fields):
            raise ValueError(
                f"Too many input elements for {ele_cls.__name__}: "
                f"expected {len(ele_cls.model_fields)} at most, got {len(parts)}"
            )

        kwargs = dict(zip(ele_cls.model_fields, parts))
        return ele_cls(**kwargs)

    def to_line(self, *, with_description: bool = True) -> str:
        def as_string(v: float | int):
            if isinstance(v, float):
                return f"{v:g}"
            return str(v)

        attr_to_value = {
            attr: as_string(getattr(self, attr)) for attr in self.model_fields
        }

        line = " ".join(attr_to_value.values())
        if not with_description:
            return f"{line} /"

        desc = f"! {type(self).__name__}: " + " ".join(attr_to_value)
        return f"{desc}\n{line} /"

    @property
    def input_filename(self) -> str | None:
        file_id = getattr(self, "file_id", None)
        if file_id is not None and file_id >= 0:
            return f"rfdata{int(file_id)}.in"

        return None


class Drift(InputElement, element_id=0):
    """
    Drift element.

    Parameters
    ----------
    length : float
        Length of the drift element in meters.
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

    Parameters
    ----------
    length : float
        The length of the quadrupole, given in meters.
    steps : int
        Number of kicks. Usually indicated as "steps" for the quadrupole.
    map_steps : int
        Number of map steps. Typically, `map_steps` is set to 1 for a quadrupole.
    B1 : float
        The gradient of the quadrupole magnetic field, measured in Tesla per meter.
    file_id : int
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
    B1: float = 0.0
    file_id: int = 0
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
        if self.file_id < 0:
            return None
        return f"rfdata{int(self.file_id)}.in"


class ConstantFocusing(InputElement, element_id=2):
    """
    3D constant focusing.

    Parameters
    ----------
    length : float
        The length of the focusing element in meters.
    steps : int
        Number of steps.
    map_steps : int
        Number of map steps.
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

    Parameters
    ----------
    length : float
        The effective length of the solenoid in meters, including
        two linear fringe regions and a flat top region.
    steps : int
        The number of simulation steps in the solenoid.
    map_steps : int
        The number of map steps.
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
    file_id: int = 0
    radius: float = 0.0
    misalignment_error_x: float = 0.0
    misalignment_error_y: float = 0.0
    rotation_error_x: float = 0.0
    rotation_error_y: float = 0.0
    rotation_error_z: float = 0.0


class Dipole(InputElement, element_id=4, has_input_file=True):
    """
    Represents a dipole element used in beam simulations.

    Parameters
    ----------
    x_field_strength : float, optional
        Field strength in the x direction.
    y_field_strength : float, optional
        Field strength in the y direction.
    file_id : float, optional
        File ID: < 100 uses t integration; > 100 but < 200 uses z map + csr wake.
    radius : float, optional
        Radius of the dipole.
    dx : float, optional
        Displacement in the x direction (unused).
    dy : float, optional
        Displacement in the y direction (unused).
    angle_x : float, optional
        Angle in the x direction (unused).
    angle_y : float, optional
        Angle in the y direction (unused).
    angle_z : float, optional
        Angle in the z direction (unused).
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
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[4] = 4

    x_field_strength: float = 0.0
    y_field_strength: float = 0.0
    file_id: float = 0.0
    radius: float = 0.0
    dx: float = 0.0  # unused
    dy: float = 0.0  # unused
    angle_x: float = 0.0  # unused
    angle_y: float = 0.0  # unused
    angle_z: float = 0.0  # unused
    misalignment_error_x: float = 0.0
    misalignment_error_y: float = 0.0
    rotation_error_x: float = 0.0
    rotation_error_y: float = 0.0
    rotation_error_z: float = 0.0

    # Docs indicate the following parameters, but the code is different:
    # angle: float = 0.0
    # k1: float = 0.0
    # input_switch: int = 0
    # half_gap: float = 0.0
    # entrance_angle: float = 0.0
    # exit_angle: float = 0.0
    # entrance_curvature: float = 0.0
    # exit_curvature: float = 0.0
    # fringe_field: float = 0.0


class Multipole(InputElement, element_id=5, has_input_file=True):
    """
    Represents a multipole element used in beam simulations.

    Parameters
    ----------
    multipole_type : MultipoleType
        The type of multipole element, sextupole, octupole, or decapole.
    field_strength : float, optional
        The strength of the magnetic field.
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

    # TODO untested
    multipole_type: MultipoleType
    field_strength: float = 0.0
    file_id: float = 0.0
    radius: float = 0.0
    misalignment_error_x: float = 0.0
    misalignment_error_y: float = 0.0
    rotation_error_x: float = 0.0
    rotation_error_y: float = 0.0
    rotation_error_z: float = 0.0


class DTL(InputElement, element_id=101):
    """
    Discrete-Transmission-Line element with specified parameters.

    Parameters
    ----------
    length : float
        The physical length of the DTL in meters.
    steps : int
        Number of steps used in the simulation.
    map_steps : int
        Number of map steps used in the simulation.
    field_scaling : float
        Scaling factor for the electrical/magnetic field.
    rf_frequency : float
        RF frequency in Hertz.
    theta0 : float
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
    theta0: float = 0.0
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


class CCDTL(InputElement, element_id=102):
    """
    A CCDTL (Cell-Coupled Drift Tube Linac) input element represented by its parameters.

    Attributes
    ----------
    length : float
        Length of the CCDTL in meters.
    steps : int
        Number of computational steps.
    map_steps : int
        Number of map steps.
    field_scaling : float
        Field scaling factor.
    rf_frequency : float
        RF frequency in Hertz.
    theta0 : float
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

    Class Attributes
    ----------------
    type: str
        Type of the element, set as 'ccdtl'.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[102] = 102

    field_scaling: float = 0.0
    rf_frequency: float = 0.0
    theta0: float = 0.0  # theta0
    file_id: float = 0.0
    radius: float = 0.0
    misalignment_error_x: float = 0.0
    misalignment_error_y: float = 0.0
    rotation_error_x: float = 0.0
    rotation_error_y: float = 0.0
    rotation_error_z: float = 0.0


class CCL(InputElement, element_id=103):
    """
    CCL input element with specific parameters.

    Attributes
    ----------
    length : float
        Length of the element in meters.
    steps : int
        Number of steps.
    map_steps : int
        Number of map steps.
    field_scaling : float
        Field scaling factor.
    rf_frequency : float
        RF frequency in Hertz.
    theta0 : float
        Driven phase in degrees.
    file_id : float
        Input field ID. If ID < 0, use the simple sinusoidal model
        (only works for the map integrator, phase is the design phase
        with 0 for maximum energy gain).
    radius : float
        Radius of the element in meters.
    x_misalignment : float
        X-axis misalignment error in meters.
    y_misalignment : float
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
    theta0: float = 0.0  # driven phase
    file_id: float = 0.0
    radius: float = 0.0
    x_misalignment: float = 0.0
    y_misalignment: float = 0.0
    rotation_error_x: float = 0.0
    rotation_error_y: float = 0.0
    rotation_error_z: float = 0.0


class SuperconductingCavity(InputElement, element_id=104, has_input_file=True):
    """
    Attributes
    ----------
    length : float
        Length of the superconducting cavity in meters.
    steps : int
        Number of steps for the superconducting kick.
    map_steps : int
        Number of map steps.
    scale : float
        Field scaling factor.
    frequency : float
        RF frequency in Hz.
    phase : float
        Driven phase in degrees (design phase with 0 for maximum energy gain).
    file_id : int
        Input field ID (if ID < 0, only works for the map integrator).
    radius : float
        Radius in meters.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[104] = 104

    scale: float = 0.0
    frequency: float = 0.0
    phase: float = 0.0  # theta0
    file_id: int = 0
    radius: float = 0.0

    # TODO not in the docs:
    x_misalignment: float = 0.0
    y_misalignment: float = 0.0
    rotation_error_x: float = 0.0
    rotation_error_y: float = 0.0
    rotation_error_z: float = 0.0


class SolenoidWithRFCavity(InputElement, element_id=105):
    """
    A solenoid with an RF cavity.

    Parameters
    ----------
    length : float
        The length of the solenoid in meters.
    steps : int
        The number of steps.
    map_steps : int
        The number of map steps.
    field_scaling : float
        The field scaling factor.
    rf_frequency : float
        The RF frequency in Hertz.
    theta0 : float
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
    aperture_size_for_wk : float
        The aperture size for wakefield computations.
    gap_size_for_wk : float
        The gap size for the wake field.
    length_for_wk : float
        The length for wake, indicating RF structure wakefield should be turned on if this value is greater than zero.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[105] = 105

    field_scaling: float  # field scaling factor
    rf_frequency: float  # RF frequency in Hz
    theta0: float  # driven phase in degrees
    file_id: float  # input field ID
    radius: float  # radius in meters
    misalignment_error_x: float  # x misalignment error in meters
    misalignment_error_y: float  # y misalignment error in meters
    rotation_error_x: float  # x rotation error in radians
    rotation_error_y: float  # y rotation error in radians
    rotation_error_z: float  # z rotation error in radians
    bz0: float  # Bz0 in Tesla
    aperture_size_for_wk: float  # aperture size for wakefield
    gap_size_for_wk: float  # gap size for wake
    length_for_wk: float  # length for wake, RF structure wakefield turned on if > 0


class TravelingWaveRFCavity(InputElement, element_id=106, has_input_file=True):
    """
    Traveling Wave RF Cavity element.

    This element represents a traveling wave RF cavity specified by several
    parameters that define its physical and operational characteristics.

    Attributes
    ----------
    length : float
        The length of the cavity in meters.
    steps : int
        The number of steps through the cavity.
    map_steps : int
        The number of map steps through the cavity.
    field_scaling : float
        Scaling factor for the field.
    rf_frequency : float
        RF frequency, in Hertz.
    theta0 : float
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
    theta1 : float
        Phase difference B and A (pi - beta * d).
    aperture_size : float
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
    theta0: float = 0.0  # driven_phase
    file_id: float = 0.0  # file_id
    radius: float = 0.0  # radius
    misalignment_error_x: float = 0.0
    misalignment_error_y: float = 0.0
    rotation_error_x: float = 0.0
    rotation_error_y: float = 0.0
    rotation_error_z: float = 0.0
    theta1: float = 0.0  # phase diff
    aperture_size: float = 0.0
    gap_size: float = 0.0
    length_for_wakefield: float = 0.0


class UserDefinedRFCavity(InputElement, element_id=110):
    """
    A user-defined RF cavity element in the simulation.

    EMfld in IMPACT-Z.

    Parameters
    ----------
    length : float
        Length of the RF cavity in meters.
    steps : int
        Number of steps.
    map_steps : int
        Number of map steps.
    field_scaling : float
        Scaling factor for the field.
    rf_frequency : float
        RF frequency in Hertz.
    theta0 : float
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
    theta0: float = 0.0  # driven phase
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
    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-1] = -1


class WriteFull(InputElement, element_id=-2, has_output_file=True):
    """
    Write the particle distribution into a fort.N file.

    Parameters
    ----------
    length : float
        Unused.
    steps : int
        Unused.
    map_steps : int
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

    def __init__(self, file_id: int | None = None, sample_frequency: int = 0, **kwargs):
        if file_id is not None:
            kwargs["map_steps"] = file_id

        super().__init__(**kwargs, sample_frequency=sample_frequency)

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-2] = -2
    unused_2: float = 0.0
    sample_frequency: int = 0

    @property
    def file_id(self) -> int:
        return self.map_steps

    @file_id.setter
    def file_id(self, value) -> None:
        self.map_steps = value


class DensityProfileInput(InputElement, element_id=-3):
    """
    Class to represent the density profile input parameters.

    Attributes
    ----------
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
    Class to write the density along R, X, Y into files such as Xprof2.data, Yprof2.data, RadDens2.data.

    Attributes
    ----------
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
    Class representing the 3D density input element.

    Attributes
    ----------
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
    Class to write the 6D phase space information and local computation
    domain information into files fort.1000, fort.1001, fort.1002, ...,
    fort.(1000+Nprocessor-1). This function is used for restart purposes.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-7] = -7


class WriteSliceInfo(InputElement, element_id=-8, has_output_file=True):
    """
    Write slice information into file fort.{file_id} using specific slices.

    Attributes
    ----------
    file_id : int
        Element file id.
    slices : float
        Number of slices.
    alphaX : float
        Twiss parameter alphaX at the location.
    betaX : float
        Twiss parameter betaX at the location (m).
    alphaY : float
        Twiss parameter alphaY at the location.
    betaY : float
        Twiss parameter betaY at the location (m).
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-8] = -8

    slices: float = 0.0
    alphaX: float = 0.0
    betaX: float = 0.0
    alphaY: float = 0.0
    betaY: float = 0.0

    @property
    def file_id(self) -> int:
        return self.map_steps

    @file_id.setter
    def file_id(self, value) -> None:
        self.map_steps = value


class ScaleMismatchParticle6DCoordinates(InputElement, element_id=-10):
    """
    Scale/mismatch the particle 6D coordinates.

    Parameters
    ----------
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


class CollimateBeamWithRectangularAperture(InputElement, element_id=-13):
    """
    Collimate the beam with transverse rectangular aperture sizes.

    Attributes
    ----------
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


class RotateBeamWithRespectToLongitudinalAxis(InputElement, element_id=-18):
    """
    Rotate the beam with respect to the longitudinal axis.

    Attributes
    ----------
    x : float
        The x-coordinate.
    y : float
        The y-coordinate.
    z : float
        The z-coordinate.
    id : int
        The id value.
    radius : float
        The radius in meters.
    rotation_angle : float
        The rotation angle in radians.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-18] = -18

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    id: int = 0
    radius: float = 0.0
    rotation_angle: float = 0.0


class BeamShift(InputElement, element_id=-19):
    """
    BeamShift shifts the beam longitudinally to the bunch centroid.

    Parameters
    ----------
    shift : float
        The amount to shift the beam longitudinally so that <dt>=<dE>=0.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-19] = -19
    shift: float = 0.0


class BeamEnergySpread(InputElement, element_id=-20):
    """
    Class representing a beam energy spread input element.

    Attributes
    ----------
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
    Class to switch the integrator type using the "bmpstp" value (the 3rd number of the line).

    Attributes
    ----------
    beam_centroid_6D : float
        Shift the beam centroid in 6D phase space.
    radius : float
        Radius (in meters), not used.
    linear_map_integrator : float
        Use linear map integrator.
    nonlinear_lorentz_integrator : float
        Use the nonlinear Lorentz integrator for complicated external fields where transfer maps are not available.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-25] = -25

    beam_centroid_6D: float = 0.0
    radius: float = 0.0
    linear_map_integrator: float = 0.0
    nonlinear_lorentz_integrator: float = 0.0


class BeamKickerByRFNonlinearity(InputElement, element_id=-40):
    """
    Beam kicker element that applies a longitudinal kick to the beam by the RF nonlinearity.

    Attributes
    ----------
    vmax : float
        Maximum voltage in volts (V).
    phi0 : float
        Initial phase offset in degrees.
    harm : int
        Harmonic number with respect to the reference frequency.
    radius : float
        Radius in meters (not used).
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-40] = -40

    vmax: float = 0.0
    phi0: float = 0.0
    harm: int = 0
    radius: float = 0.0


class RfcavityStructureWakefield(InputElement, element_id=-41, has_input_file=True):
    """
    Class representing the read-in RF cavity structure wakefield.

    Parameters
    ----------
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-41] = -41

    not_used: float = 1.0
    file_id: float = 0
    # TODO -1.0 RF off, 1.0 RF on, < 10 no transverse wakefield effects included
    enable_wakefield: float = 0.0


class EnergyModulation(InputElement, element_id=-52):
    """
    Class representing the energy modulation (emulate laser heater).

    Attributes
    ----------
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
    Kick the beam using thin lens multipole.

    Parameters
    ----------
    param1 : float
        First parameter, "1.0" not used.
    param2 : float
        Second parameter, 0.0 - dipole (k0), 1.0 - quad. (k1), 2.0 - sext. (k2),
        3.0 - oct. (k3), 4.0 - dec. (k4), 5.0 - dodec. (k5).
    param3 : float
        Third parameter.
    param4 : float
        Fourth parameter.
    param5 : float
        Fifth parameter.
    param6 : float
        Sixth parameter.
    param7 : float
        Seventh parameter.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-55] = -55

    param1: float = 0.0
    param2: float = 0.0
    param3: float = 0.0
    param4: float = 0.0
    param5: float = 0.0
    param6: float = 0.0
    param7: float = 0.0


class HaltExecution(InputElement, element_id=-99):
    """
    Halt execution at this point in the input file.

    This is useful if you have a big file and want to run part-way through it without deleting a lot of lines.
    """

    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    type_id: Literal[-99] = -99


class ImpactZInput(BaseModel):
    """Input settings for an IMPACT-Z run."""

    initial_particles: PydanticParticleGroup | None = None

    # Line 1
    ncpu_y: int = 0
    ncpu_z: int = 0
    gpu: GPUFlag = GPUFlag.disabled

    # Line 2
    seed: int = 0
    n_particle: int = 0
    integrator_type: IntegratorType = IntegratorType.linear
    err: int = 1
    diagnostic_type: DiagnosticType = DiagnosticType.at_given_time
    output_z: OutputZType = OutputZType.extended

    # Line 3
    nx: int = 0
    ny: int = 0
    nz: int = 0
    boundary_type: BoundaryType = BoundaryType.trans_open_longi_open
    radius_x: float = 0.0
    radius_y: float = 0.0
    z_period_size: float = 0.0

    # Line 4
    distribution: DistributionZType = DistributionZType.uniform
    restart: int = 0
    subcycle: int = 0
    nbunch: int = 0

    particle_list: list[int] = []

    current_list: list[float] = []

    charge_over_mass_list: list[float] = []

    twiss_alpha_x: float = 0.0
    twiss_beta_x: float = 1.0
    twiss_norm_emit_x: float = 1e-6
    twiss_mismatch_x: float = 1.0
    twiss_mismatch_px: float = 1.0
    twiss_offset_x: float = 0.0
    twiss_offset_px: float = 0.0

    twiss_alpha_y: float = 0.0
    twiss_beta_y: float = 1.0
    twiss_norm_emit_y: float = 1e-6
    twiss_mismatch_y: float = 1.0
    twiss_mismatch_py: float = 1.0
    twiss_offset_y: float = 0.0
    twiss_offset_py: float = 0.0

    twiss_alpha_z: float = 0.0
    twiss_beta_z: float = 1.0
    twiss_norm_emit_z: float = 1e-6
    twiss_mismatch_z: float = 1.0
    twiss_mismatch_e_z: float = 1.0
    twiss_offset_phase_z: float = 0.0
    twiss_offset_energy_z: float = 0.0

    average_current: float = 1.0
    initial_kinetic_energy: float = 0.0
    reference_particle_mass: float = 0.0
    reference_particle_charge: float = 0.0
    reference_frequency: float = 0.0
    initial_phase_ref: float = 0.0

    lattice: list[InputElement] = []
    filename: pathlib.Path | None = pydantic.Field(default=None, exclude=True)

    file_data: dict[str, NDArray] = pydantic.Field(default={}, repr=False)

    @classmethod
    def from_file(cls, filename: pathlib.Path | str) -> ImpactZInput:
        sections = parsers.read_input_file(filename)
        return cls._from_input_sections(sections, filename=filename)

    @classmethod
    def from_contents(
        cls, contents: str, filename: AnyPath | None = None
    ) -> ImpactZInput:
        sections = parsers.parse_input_lines(contents)
        return cls._from_input_sections(sections, filename=filename)

    @classmethod
    def _from_input_sections(
        cls,
        sections: list[parsers.InputFileSection],
        filename: AnyPath | None,
    ) -> ImpactZInput:
        data = sum((sect.data for sect in sections), [])
        res = cls(filename=pathlib.Path(filename) if filename else None)

        # Casts here are to satisfy the linter. Actual validation will be handled by pydantic.
        if len(data[0]) >= 3:
            # GPU flag is written by the Python GUI but not actually read out
            # by IMPACT-Z.
            res.ncpu_y, res.ncpu_z, res.gpu = cast(
                tuple[int, int, GPUFlag], data[0][:3]
            )
        else:
            res.ncpu_y, res.ncpu_z = cast(tuple[int, int], data[0][:2])
        (
            res.seed,
            res.n_particle,
            res.integrator_type,
            res.err,
            res.output_z,
        ) = cast(tuple[int, int, IntegratorType, int, OutputZType], data[1][:5])
        (
            res.nx,
            res.ny,
            res.nz,
            res.boundary_type,
            res.radius_x,
            res.radius_y,
            res.z_period_size,
        ) = cast(tuple[int, int, int, BoundaryType, float, float, float], data[2][:8])
        (
            res.distribution,
            res.restart,
            res.subcycle,
            res.nbunch,
        ) = cast(tuple[DistributionZType, int, int, int], data[3][:4])

        res.particle_list = [int(v) for v in data[4]]
        res.current_list = [float(v) for v in data[5]]
        res.charge_over_mass_list = [float(v) for v in data[6]]

        (
            res.twiss_alpha_x,
            res.twiss_beta_x,
            res.twiss_norm_emit_x,
            res.twiss_mismatch_x,
            res.twiss_mismatch_px,
            res.twiss_offset_x,
            res.twiss_offset_px,
        ) = data[7]
        (
            res.twiss_alpha_y,
            res.twiss_beta_y,
            res.twiss_norm_emit_y,
            res.twiss_mismatch_y,
            res.twiss_mismatch_py,
            res.twiss_offset_y,
            res.twiss_offset_py,
        ) = data[8]
        (
            res.twiss_alpha_z,
            res.twiss_beta_z,
            res.twiss_norm_emit_z,
            res.twiss_mismatch_z,
            res.twiss_mismatch_e_z,
            res.twiss_offset_phase_z,
            res.twiss_offset_energy_z,
        ) = data[9]

        (
            res.average_current,
            res.initial_kinetic_energy,
            res.reference_particle_mass,
            res.reference_particle_charge,
            res.reference_frequency,
            res.initial_phase_ref,
        ) = data[10]

        if filename is not None:
            work_dir = pathlib.Path(filename).parent
        else:
            work_dir = None

        res.lattice = []
        for idx, lattice_line in enumerate(data[11:], start=1):
            ele = InputElement.from_line(lattice_line)
            res.lattice.append(ele)

            if ele.input_filename and work_dir is not None:
                if not ele.class_information().has_input_file or not isinstance(
                    ele, HasInputFile
                ):
                    continue

                ele_file_id = int(ele.file_id)
                ext_data_fn = work_dir / ele.input_filename
                try:
                    res.file_data[str(ele_file_id)] = parsers.sections_to_ndarray(
                        parsers.read_input_file(ext_data_fn)
                    )
                except FileNotFoundError:
                    logger.warning(
                        f"Referenced file in lattice element {idx} (of type {type(ele).__name__}) "
                        f"does not exist in: {ext_data_fn}"
                    )

        return res

    def to_contents(
        self,
        header="Written by LUME-ImpactZ",
        include_gpu: bool = False,
    ) -> str:
        def stringify_list(lst: Sequence[float | int]):
            return " ".join(str(v) for v in lst)

        if include_gpu:
            gpu = f" {int(self.gpu)}"
        else:
            gpu = ""

        lattice = "\n".join(ele.to_line() for ele in self.lattice)
        return f"""
! {header}
! ncpu_y ncpu_z
{self.ncpu_y} {self.ncpu_z}{gpu}
! seed n_particle integrator_type err output_z
{self.seed} {self.n_particle} {int(self.integrator_type)} {self.err} {int(self.output_z)}
! nx ny nz boundary_type radius_x radius_y z_period_size
{self.nx} {self.ny} {self.nz} {self.boundary_type} {self.radius_x} {self.radius_y} {self.z_period_size}
! distribution restart subcycle nbunch
{self.distribution} {self.restart} {self.subcycle} {self.nbunch}
! particle_list
{stringify_list(self.particle_list)}
! current_list
{stringify_list(self.current_list)}
! charge_over_mass_list
{stringify_list(self.charge_over_mass_list)}
! twiss_alpha_x twiss_beta_x twiss_norm_emit_x twiss_mismatch_x twiss_mismatch_px twiss_offset_x twiss_offset_px
{self.twiss_alpha_x} {self.twiss_beta_x} {self.twiss_norm_emit_x} {self.twiss_mismatch_x} {self.twiss_mismatch_px} {self.twiss_offset_x} {self.twiss_offset_px}
! twiss_alpha_y twiss_beta_y twiss_norm_emit_y twiss_mismatch_y twiss_mismatch_py twiss_offset_y twiss_offset_py
{self.twiss_alpha_y} {self.twiss_beta_y} {self.twiss_norm_emit_y} {self.twiss_mismatch_y} {self.twiss_mismatch_py} {self.twiss_offset_y} {self.twiss_offset_py}
! twiss_alpha_z twiss_beta_z twiss_norm_emit_z twiss_mismatch_z twiss_mismatch_e_z twiss_offset_phase_z twiss_offset_energy_z
{self.twiss_alpha_z} {self.twiss_beta_z} {self.twiss_norm_emit_z} {self.twiss_mismatch_z} {self.twiss_mismatch_e_z} {self.twiss_offset_phase_z} {self.twiss_offset_energy_z}
! average_current initial_kinetic_energy reference_particle_mass reference_particle_charge reference_frequency initial_phase_ref
{self.average_current} {self.initial_kinetic_energy} {self.reference_particle_mass} {self.reference_particle_charge} {self.reference_frequency} {self.initial_phase_ref}
! ** lattice **
{lattice}
        """.strip()

    def write(
        self, workdir: AnyPath, error_if_missing: bool = False
    ) -> list[pathlib.Path]:
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
        if self.initial_particles:
            # TODO cmayes cathode_kinetic_energy_ref?
            particles_path = workdir / "particle.in"
            iz_particles = ImpactZParticles.from_particle_group(
                self.initial_particles,
                reference_frequency=self.reference_frequency,
                reference_kinetic_energy=self.initial_kinetic_energy,
            )
            iz_particles.write_impact(particles_path)
            # TODO: support this in openpmd-beamphysics
            # self.initial_particles.write_impact(
            #     str(particles_path),
            #     cathode_kinetic_energy_ref=self.initial_kinetic_energy,
            # )
            extra_paths.append(particles_path)

        for ele in self.lattice:
            if not ele.class_information().has_input_file or not isinstance(
                ele, HasInputFile
            ):
                continue

            fn = ele.input_filename
            if fn is not None:
                file_id = ele.file_id
                try:
                    data = self.file_data[file_id]
                except KeyError:
                    if error_if_missing:
                        raise FileNotFoundError(f"Missing input file: {fn}")
                    logger.warning(f"File unavailable: {file_id}")
                else:
                    np.savetxt(workdir / fn, data)

        return [input_file_path, *extra_paths]

    def write_run_script(
        self,
        path: pathlib.Path,
        command_prefix: str = "ImpactZexe",
    ) -> None:
        path.parent.mkdir(exist_ok=True)
        with open(path, mode="w") as fp:
            print(shlex.join(shlex.split(command_prefix)), file=fp)
        lume_tools.make_executable(str(path))
