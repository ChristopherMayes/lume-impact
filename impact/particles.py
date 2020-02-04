


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



def impact_particles_to_particle_data(tout, species='electron', time=0, macrocharge=0):
    """
    Convert impact particles to a standard form 
    
    """
    
    mc2 = SPECIES_MASS[species]
    
    data = {}
    
    n_particle = len(tout['x'])
    
    data['x'] = tout['x']
    data['y'] = tout['y']
    data['z'] = tout['z']
    factor = c_light**2 /e_charge # kg -> eV
    data['px'] = tout['GBx']*mc2
    data['py'] = tout['GBy']*mc2
    data['pz'] = tout['GBz']*mc2
    
    data['t'] = np.full(n_particle, time)
    data['status'] = np.full(n_particle, 1)
    if macrocharge == 0:
        weight = 1/n_particle
    else:
        weight = abs(macrocharge)
    data['weight'] =  np.full(n_particle, weight) 
    
    
    
    data['species'] = species
    data['n_particle'] = n_particle
    return data
    
    
    
def write_bmad_particles_ascii(particles, filename='test.beam0', charge=250e-12, mean_z=0, t_center=0, ix_ele=0, ix_bunch=1, species='electron'):
    n = len(particles)
    header = f'{ix_ele}\n{ix_bunch}\n{n}\nBEGIN_BUNCH\n{species}\n{charge}\n{mean_z}\n{t_center}'
    footer = 'END_BUNCH'
    
    np.savetxt(filename, particles, header=header, footer=footer, comments='')
    print('Written:', filename)    