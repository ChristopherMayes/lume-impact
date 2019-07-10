#import numpy as np

from .parsers import header_lines
from .lattice import lattice_lines

def write_impact_input(filePath, header, eles):
    """
    Write
    
    Note that the filename ultimately needs to be ImpactT.in
    
    """
    lines =  header_lines(header) + lattice_lines(eles)
    with open(filePath, 'w') as f:
        for line in lines:
            f.write(line+'\n')



def write_impact_particles_h5(h5, particle_data, name=None, total_charge=1.0, speciesType='electron'):
    # Write particle data at a screen in openPMD BeamPhysics format
    # https://github.com/DavidSagan/openPMD-standard/blob/EXT_BeamPhysics/EXT_BeamPhysics.md

    if not name:
        g = h5
    else:
        g = h5.create_group(name)

    g.attrs['speciesType'] = speciesType

    g.attrs['totalCharge'] = total_charge

    n_particle = len(particle_data['x'])

    # Position
    g['position/x']=particle_data['x'] # in meters
    g['position/y']=particle_data['y']
    g['position/z']=particle_data['z']
    g['position'].attrs['unitSI'] = 1.0
    g['position'].attrs['unitDimension']=(1., 0., 0., 0., 0., 0., 0.) # m

    # momenta
    g['momentum/x']=particle_data['GBx'] # gamma*beta_x
    g['momentum/y']=particle_data['GBy'] # gamma*beta_y
    g['momentum/z']=particle_data['GBz'] # gamma*beta_z
    g['momentum'].attrs['unitSI']= 2.73092449e-22 # m_e *c in kg*m / s
    g['momentum'].attrs['unitDimension']=(1., 1., -1., 0., 0., 0., 0.) # kg*m / s
    
    # Constant records
    
    # Weights. All particles should have the same weight (macro charge)
    weight = total_charge / n_particle
    g2 = g.create_group('weight')
    g2.attrs['value']  = weight
    g2.attrs['shape'] = (n_particle)
    g2.attrs['unitSI'] = 1.0
    g2.attrs['unitDimension'] = (0., 0., 1, 1., 0., 0., 0.) # Amp*s = Coulomb    


