from pmd_beamphysics.units import e_charge, known_unit, mec2, pmd_unit

# Patch these into the lookup dict.
known_unit["mec2"] = pmd_unit("m_ec^2", mec2 * e_charge, "energy")

for key in ["field_energy", "pulse_energy"]:
    known_unit[key] = known_unit["J"]
known_unit["peak_power"] = known_unit["W"]
known_unit["m^{-1}"] = pmd_unit("1/m", 1, (-1, 0, 0, 0, 0, 0, 0))
known_unit["m^{-2}"] = pmd_unit("1/m^2", 1, (-2, 0, 0, 0, 0, 0, 0))
known_unit["{s}"] = known_unit["s"]
known_unit["ev"] = known_unit["eV"]
