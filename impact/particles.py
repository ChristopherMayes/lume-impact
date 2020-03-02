


import scipy.constants

m_e = scipy.constants.value('electron mass energy equivalent in MeV')*1e6
m_p = scipy.constants.value('proton mass energy equivalent in MeV')*1e6
c_light = 299792458
e_charge = scipy.constants.e

import numpy as np

SPECIES_MASS = {
    'electron': m_e,
    'proton': m_p
    
}


def identify_species(mass_eV, charge_sign):
    """
    Simple function to identify a species based on its mass in eV and charge sign.
    
    Finds species:
        'electron'
        'positron'
    
    TODO: more species
    
    """
    m = round(mass_eV*1e-2)/1e-2
    if m == 511000.0:
        if charge_sign == 1:
            return 'positron'
        if charge_sign == -1:
            return 'electron'
    if m == 938272100.0:
        if charge_sign == 1:
            return 'proton'
        
    raise Exception(f'Cannot identify species with mass {mass_eV} eV and charge {charge_sign} e')

