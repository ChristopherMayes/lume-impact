#import numpy as np

def write_impact_particles_h5(h5, particle_data, name=None, speciesType='electron'):
    # Write particle data at a screen in openPMD BeamPhysics format
    # https://github.com/DavidSagan/openPMD-standard/blob/EXT_BeamPhysics/EXT_BeamPhysics.md

    if not name:
        g = h5
    else:
        g = h5.create_group(name)

    g.attrs['speciesType'] = speciesType

    #macrocharge = screen_data['q']*screen_data['nmacro']
    #g.attrs['totalCharge'] = np.sum(macrocharge)

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


#     # Time
#    g['time'] = screen_data['t']
#    g['time'].attrs['unitSI'] = 1.0 # s
#    g['time'].attrs['unitDimension'] = (0., 0., 1., 0., 0., 0., 0.) # s
#
#    # Weights
#    weights = abs(screen_data['nmacro']*screen_data['q']) 
#    if len(set(weights)) == 1:
#        # Constant record
#        g2 = g.create_group('weight')
#        g2.attrs['value']  = weights[0]
#        g2.attrs['shape'] = (len(weights))
#        g2.attrs['unitSI'] = 1.0
#        g2.attrs['unitDimension'] = (0., 0., 1, 1., 0., 0., 0.) # Amp*s = Coulomb
#    else: 
#        # Unique weights
#        g['weight'] = weights
#        g['weight'].attrs['unitSI'] = 1.0
#        g['weight'].attrs['unitDimension']=(0., 0., 1, 1., 0., 0., 0.) # Amp*s = Coulomb
#

