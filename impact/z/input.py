from __future__ import annotations
import ast
import pathlib
from typing import Literal, Sequence
import pydantic

from .constants import (
    BoundaryType,
    DiagnosticType,
    DistributionZType,
    GPUFlag,
    IntegratorType,
    OutputZType,
)


class BaseModel(pydantic.BaseModel, extra="forbid", validate_assignment=True):
    pass


input_element_by_id = {}


class InputElement(BaseModel):
    length: float = 0.0
    steps: int = 0
    map_steps: int = 0
    # next must be type_id

    def __init_subclass__(cls, element_id: int, **kwargs):
        super().__init_subclass__(**kwargs)

        assert isinstance(element_id, int)
        assert (
            element_id not in input_element_by_id
        ), f"Duplicate element ID {element_id}"
        input_element_by_id[element_id] = cls

    @staticmethod
    def from_line(line: str | InputLine):
        if isinstance(line, str):
            parts = parse_input_line(line)
        else:
            parts = line

        type_idx = parts[3]
        ele_cls = input_element_by_id[type_idx]

        if len(parts) > len(ele_cls.model_fields):
            raise ValueError(
                f"Too many input elements for {ele_cls.__name__}: "
                f"expected {len(ele_cls.model_fields)} at most, got {len(parts)}"
            )

        kwargs = dict(zip(ele_cls.model_fields, parts))
        return ele_cls(**kwargs)


class Drift(InputElement, element_id=0):
    """
    Drift element.

    Parameters
    ----------
    length : float
        Length of the drift element in meters. Example: 0.0620822983935m.
    steps : int
        Number of space-charge kicks through the beamline element. Each
        "step" consists of a half-step, a space-charge kick, and another half-step.
    map_steps : int
        Number of "map steps". Each half-step involves computing a map for that
        half-element which is computed by numerical integration.
    radius : float
        Radius of the pipe, in meters. Default example value is 1 m.
    """

    type_id: Literal[0]
    radius: float = 0.0
    unused_0: float = 0.0  # TODO undocumented/unused?
    unused_1: float = 0.0


class Quadrupole(InputElement, element_id=1):
    """
    A quadrupole element.

    Parameters
    ----------
    length : float
        The length of the quadrupole, given in meters, with a typical value of 0.05 m.
    steps : int
        Number of kicks. Usually indicated as "steps" for the quadrupole.
    map_steps : int
        Number of map steps. Typically, `map_steps` is set to 1 for a quadrupole.
    B1 : float
        The gradient of the quadrupole magnetic field, measured in Tesla per meter.
        A typical value is 16.4423850936 T/m.
    input_file_id : int
        An ID for the input gradient file. Determines profile behavior:
        if greater than 0, a fringe field profile is read; if less than -10,
        a linear transfer map of an undulator is used; if between -10 and 0,
        it's the k-value linear transfer map; if equal to 0, it uses the linear
        transfer map with the gradient.
    radius : float
        The radius of the quadrupole, measured in meters.
    dx : float, optional
        The x-direction misalignment error, given in meters. Defaults to 0.0 if
        not specified.
    dy : float, optional
        The y-direction misalignment error, given in meters. Defaults to 0.0 if
        not specified.
    rotation_error_x : float, optional
        Rotation error in radians.
    rotation_error_y : float, optional
        Rotation error in radians.
    rotation_error_z : float, optional
        Rotation error in radians.
    """

    type_id: Literal[1]
    B1: float = 0.0
    input_file_id: int = 0
    radius: float = 0.0
    dx: float = 0.0
    dy: float = 0.0

    rotation_error_x: float = 0.0
    rotation_error_y: float = 0.0
    rotation_error_z: float = 0.0


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

    type_id: Literal[2]
    kx0_squared: float = 0.0
    ky0_squared: float = 0.0
    kz0_squared: float = 0.0
    radius: float = 0.0


class Solenoid(InputElement, element_id=3):
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
    input_field_file_id : float
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
    misalignment_error_z : float
        Rotation error in the z-direction in radians.
    """

    type_id: Literal[3]
    Bz0: float = 0.0
    input_field_file_id: float = 0.0
    radius: float = 0.0
    misalignment_error_x: float = 0.0
    misalignment_error_y: float = 0.0
    rotation_error_x: float = 0.0
    y_rotation_error: float = 0.0
    misalignment_error_z: float = 0.0


class Dipole(InputElement, element_id=4):
    """
    Represents a dipole element used in beam simulations.

    Parameters:
    ----------
    length : float
        Length of the dipole in meters.
    steps : int
        The number of "steps" for tracking particles within the dipole.
    map_steps : int
        The number of "map steps" used in the simulation.
    angle : float
        Bending angle of the dipole in radians.
    k1 : float
        Quadrupole component (focusing strength) of the dipole.
    input_switch : int
        An input switch; if greater than 200, it indicates inclusion of 1D CSR.
    half_gap : float
        Half gap of the dipole in meters.
    entrance_angle : float
        Entrance pole face angle in radians.
    exit_angle : float
        Exit pole face angle in radians.
    entrance_curvature : float
        Curvature of the entrance face.
    exit_curvature : float
        Curvature of the exit face.
    fringe_field : float
        Integrated fringe field of the dipole.

    """

    type_id: Literal[4]
    angle: float = 0.0
    k1: float = 0.0
    input_switch: int = 0
    half_gap: float = 0.0
    entrance_angle: float = 0.0
    exit_angle: float = 0.0
    entrance_curvature: float = 0.0
    exit_curvature: float = 0.0
    fringe_field: float = 0.0

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
    driven_phase : float
        Driven phase in degrees.
    input_field_id : float
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

    type_id: Literal[101]
    field_scaling: float = 0.0
    rf_frequency: float = 0.0
    driven_phase: float = 0.0
    input_field_id: float = 0.0
    radius: float = 0.0
    quad1_length: float = 0.0
    quad1_gradient: float = 0.0
    quad2_length: float = 0.0
    quad2_gradient: float = 0.0
    misalignment_error_x: float = 0.0
    misalignment_error_y: float = 0.0
    rotation_error_x: float = 0.0
    rotation_error_y: float = 0.0
    rotation_error_z: float = 0.0
    displacement_x: float = 0.0
    displacement_y: float = 0.0
    rotation_error_rf_x: float = 0.0
    rotation_error_rf_y: float = 0.0
    rotation_error_rf_z: float = 0.0


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
    driven_phase : float
        Driven phase in degrees.
    input_field_id : float
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

    type_id: Literal[102]
    field_scaling: float = 0.0
    rf_frequency: float = 0.0
    driven_phase: float = 0.0
    input_field_id: float = 0.0
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
    driven_phase : float
        Driven phase in degrees.
    input_field_id : float
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

    type_id: Literal[103]
    field_scaling: float = 0.0
    rf_frequency: float = 0.0
    driven_phase: float = 0.0
    input_field_id: float = 0.0
    radius: float = 0.0
    x_misalignment: float = 0.0
    y_misalignment: float = 0.0
    rotation_error_x: float = 0.0
    rotation_error_y: float = 0.0
    rotation_error_z: float = 0.0


class SuperconductingCavity(InputElement, element_id=104):
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
    input_file_id : int
        Input field ID (if ID < 0, only works for the map integrator).
    radius : float
        Radius in meters.
    """

    type_id: Literal[104]
    scale: float = 0.0
    frequency: float = 0.0
    phase: float = 0.0
    input_file_id: int = 0
    radius: float = 0.0


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
    driven_phase : float
        The driven phase in degrees.
    input_field_id : float
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

    type_id: Literal[105]
    field_scaling: float  # field scaling factor
    rf_frequency: float  # RF frequency in Hz
    driven_phase: float  # driven phase in degrees
    input_field_id: float  # input field ID
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


class TravelingWaveRFCavity(InputElement, element_id=106):
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
    driven_phase : float
        Driven phase in degrees.
    input_field_id : float
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

    type_id: Literal[106]
    field_scaling: float = 0.0  # scale
    rf_frequency: float = 0.0  # rf freq
    theta0: float = 0.0  # theta0
    input_field_id: float = 0.0  # file_id
    radius: float = 0.0  # radius
    misalignment_error_x: float = 0.0
    misalignment_error_y: float = 0.0
    rotation_error_x: float = 0.0
    rotation_error_y: float = 0.0
    rotation_error_z: float = 0.0
    theta1: float = 0.0
    aperture_size: float = 0.0
    gap_size: float = 0.0
    length_for_wakefield: float = 0.0


#  1.48524 10 20 106
# 1.0 field_scaling
# 700.0e6 rf freq
# 30. driven phase
# 1.0 field id
# 0.014 rad
# 0. misx
# 0.misy
# 0. rotx
# 0. roty
# 0. rotz
# 0.5 phase diff
# 0. aperture size
# 0. gap size
# 0. length for wk
# /
# Traveling wave RF cavity, length=1.48524m, 10 "steps", 20 "map steps", field scaling=1.0, RF
# frequency=700.0e6, driven phase=30.0 degree, input field ID=1.0, radius=0.014m, x misalignment
# error=0.0m, y misalignment error=0.0m, rotation error x, y, z=0.0, 0., 0. rad, (pi - beta * d) phase
# difference B and A, 0. "aperture size for wakefield", 0. "gap size for wk", 0. "length for wk". RF
# structure wakefield only turned with length of wk>0.


class UserDefinedRFCavity(InputElement, element_id=110):
    """
    A user-defined RF cavity element in the simulation.

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
    input_field_id : float
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

    type_id: Literal[110]
    field_scaling: float = 0.0
    rf_frequency: float = 0.0
    driven_phase: float = 0.0
    input_field_id: float = 0.0
    radius_x: float = 0.0
    radius_y: float = 0.0
    misalignment_error_x: float = 0.0
    misalignment_error_y: float = 0.0
    rotation_error_x: float = 0.0
    rotation_error_y: float = 0.0
    rotation_error_z: float = 0.0
    data_mode: float = 0.0
    coordinate_type: float = 0.0


class ShiftCentroid(InputElement, element_id=-1):
    type_id: Literal[-1]


class WriteFull(InputElement, element_id=-2):
    """
    Write the particle distribution into a fort.N file.

    Parameters
    ----------
    file_id : int
        Identifier for the file to which particles will be written.

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

    type_id: Literal[-2]
    unused_2: float = 0.0
    sample_frequency: int = 0


InputLine = Sequence[float | int]


class InputFileSection(BaseModel):
    comments: list[str] = []
    data: list[InputLine] = []


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

    type_id: Literal[-3]

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

    type_id: Literal[-4]

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

    type_id: Literal[-5]

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

    type_id: Literal[-6]

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

    type_id: Literal[-7]


class WriteSliceInfo(InputElement, element_id=-8):
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

    type_id: Literal[-8]
    slices: float = 0.0
    alphaX: float = 0.0
    betaX: float = 0.0
    alphaY: float = 0.0
    betaY: float = 0.0

    @property
    def file_id(self) -> int:
        return self.map_steps

    @file_id.setter
    def file_id(self, value) -> int:
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

    type_id: Literal[-10]
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

    type_id: Literal[-13]
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

    type_id: Literal[-18]
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

    type_id: Literal[-19]
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

    type_id: Literal[-20]
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

    type_id: Literal[-21]

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

    type_id: Literal[-25]
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

    type_id: Literal[-40]
    vmax: float = 0.0
    phi0: float = 0.0
    harm: int = 0
    radius: float = 0.0


class RfcavityStructureWakefield(InputElement, element_id=-41):
    """
    Class representing the read-in RF cavity structure wakefield.

    Parameters
    ----------
    """

    type_id: Literal[-41]
    not_used: float = 1.0
    file_id: int = 0
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

    type_id: Literal[-52]

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

    type_id: Literal[-55]
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

    type_id: Literal[-99]


def parse_input_line(line: str) -> list[InputFileSection]:
    line = line.replace("D", "E").replace("d", "e")  # fortran float style
    parts = line.split()
    if "/" in parts:
        parts = parts[: parts.index("/")]
    return [ast.literal_eval(value) for value in parts]


def parse_input_lines(lines: Sequence[str]) -> list[InputFileSection]:
    section = InputFileSection()
    sections = [section]
    last_comment = True
    for line in lines:
        if line.startswith("!"):
            if not last_comment and section is not None:
                section = InputFileSection()
                sections.append(section)

            section.comments.append(line.lstrip("! "))
        else:
            parts = parse_input_line(line)
            if parts:
                section.data.append(parts)

    return sections


def read_input_file(filename):
    with open(filename, "rt") as fp:
        return parse_input_lines(fp.read().splitlines())


class TwissXorY(
    BaseModel,
):
    alpha: float = 0.0
    beta: float = 0.0
    emit: float = 0.0
    mismatch: float = 0.0
    mismatch_p: float = 0.0
    offset: float = 0.0
    offset_p: float = 0.0

    @classmethod
    def from_file(
        cls,
        alpha: float = 0.0,
        beta: float = 0.0,
        emit: float = 0.0,
        mismatch: float = 0.0,
        mismatch_p: float = 0.0,
        offset: float = 0.0,
        offset_p: float = 0.0,
    ):
        return cls(
            alpha=alpha,
            beta=beta,
            emit=emit,
            mismatch=mismatch,
            mismatch_p=mismatch_p,
            offset=offset,
            offset_p=offset_p,
        )


class TwissZ(BaseModel):
    alpha: float = 0.0
    beta: float = 0.0
    emit: float = 0.0
    mismatch: float = 0.0
    mismatch_e: float = 0.0
    offset_phase: float = 0.0
    offset_energy: float = 0.0

    @classmethod
    def from_file(
        cls,
        alpha: float = 0.0,
        beta: float = 0.0,
        emit: float = 0.0,
        mismatch: float = 0.0,
        mismatch_e: float = 0.0,
        offset_phase: float = 0.0,
        offset_energy: float = 0.0,
    ):
        return cls(
            alpha=alpha,
            beta=beta,
            emit=emit,
            mismatch=mismatch,
            mismatch_e=mismatch_e,
            offset_phase=offset_phase,
            offset_energy=offset_energy,
        )


class ImpactZInput(BaseModel):
    ncpu_y: int = 0
    ncpu_z: int = 0
    gpu: GPUFlag = GPUFlag.disabled

    dim: int = 0
    np: int = 0
    integrator_type: IntegratorType = IntegratorType.linear
    err: int = 0
    diagnostic_type: DiagnosticType = DiagnosticType.at_given_time
    output_z: OutputZType = OutputZType.standard

    ngx: int = 0
    ngy: int = 0
    ngz: int = 0
    boundary_type: BoundaryType = BoundaryType.trans_open_longi_open
    x_rad: float = 0.0
    y_rad: float = 0.0
    z_period_size: float = 0.0

    distribution_z: DistributionZType = DistributionZType.uniform
    restart: int = 0
    subcycle: int = 0
    nbunch: int = 0

    particle_list: list[int] = []

    current: list[float] = []

    charge: list[float] = []

    twiss_x: TwissXorY = TwissXorY()
    twiss_y: TwissXorY = TwissXorY()
    twiss_z: TwissZ = TwissZ()

    current_averaged: float = 0.0
    initial_kinetic_energy: float = 0.0
    particle_mass: float = 0.0
    particle_charge: float = 0.0
    scaling_frequency: float = 0.0
    initial_phase_ref: float = 0.0

    elements: list[InputElement] = []

    @classmethod
    def from_file(cls, filename: pathlib.Path | str):
        sections = read_input_file(filename)
        data = sum((sect.data for sect in sections), [])
        res = cls()

        if len(data[0]) == 3:
            res.ncpu_y, res.ncpu_z, res.gpu = data[0]
        else:
            res.ncpu_y, res.ncpu_z = data[0]
        (
            res.dim,
            res.np,
            res.integrator_type,
            res.err,
            res.output_z,
        ) = data[1]
        (
            res.ngx,
            res.ngy,
            res.ngz,
            res.boundary_type,
            res.x_rad,
            res.y_rad,
            res.z_period_size,
        ) = data[2]
        (
            res.distribution_z,
            res.restart,
            res.subcycle,
            res.nbunch,
        ) = data[3]

        res.particle_list = data[4]
        res.current = data[5]
        res.charge = data[6]

        res.twiss_x = TwissXorY.from_file(*data[7])
        res.twiss_y = TwissXorY.from_file(*data[8])
        res.twiss_z = TwissZ.from_file(*data[9])

        (
            res.current_averaged,
            res.initial_kinetic_energy,
            res.particle_mass,
            res.particle_charge,
            res.scaling_frequency,
            res.initial_phase_ref,
        ) = data[10]

        res.elements = []
        for lattice_line in data[11:]:
            ele = InputElement.from_line(lattice_line)
            res.elements.append(ele)

        return res
