from __future__ import annotations

from pmd_beamphysics.units import e_charge, known_unit, mec2, pmd_unit
from typing import Annotated
from .types import NDArray

# Patch these into the lookup dict.
known_unit["mec2"] = pmd_unit("m_ec^2", mec2 * e_charge, "energy")

known_unit["m^{-1}"] = pmd_unit("1/m", 1, (-1, 0, 0, 0, 0, 0, 0))
known_unit["m^{-2}"] = pmd_unit("1/m^2", 1, (-2, 0, 0, 0, 0, 0, 0))
known_unit["{s}"] = known_unit["s"]
known_unit["ev"] = known_unit["eV"]

pmd_MeV = pmd_unit("MeV", 1e6 * known_unit["eV"].unitSI, known_unit["eV"].unitDimension)

Amperes = Annotated[float, {"units": known_unit["A"]}]
Meters = Annotated[float, {"units": known_unit["m"]}]
Meter_Rad = Annotated[float, {"units": known_unit["m"] * known_unit["rad"]}]
Radians = Annotated[float, {"units": known_unit["rad"]}]
Degrees = Annotated[float, {"units": known_unit["degree"]}]
MeV = Annotated[float, {"units": pmd_MeV}]
Unitless = Annotated[float, {"units": known_unit["1"]}]
Seconds = Annotated[float, {"units": known_unit["s"]}]

AmperesArray = Annotated[NDArray, {"units": known_unit["A"]}]
DegreesArray = Annotated[NDArray, {"units": known_unit["degree"]}]
Meter_RadArray = Annotated[NDArray, {"units": known_unit["m"] * known_unit["rad"]}]
MetersArray = Annotated[NDArray, {"units": known_unit["m"]}]
RadiansArray = Annotated[NDArray, {"units": known_unit["rad"]}]
SecondsArray = Annotated[NDArray, {"units": known_unit["s"]}]
UnitlessArray = Annotated[NDArray, {"units": known_unit["1"]}]
eVArray = Annotated[NDArray, {"units": known_unit["eV"]}]
eV_c_Array = Annotated[NDArray, {"units": known_unit["eV/c"]}]
